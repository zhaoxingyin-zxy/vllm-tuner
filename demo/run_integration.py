"""
demo/run_integration.py
-----------------------
End-to-end integration test for vLLM-Tuner WITHOUT real hardware.

What is mocked:
  - RemoteEnv  : SSH calls return preset OK responses
  - Brain      : Claude API replaced by a deterministic mock that cycles configs
  - Hardware   : AscendObserver returns synthetic stats
  - Actor      : restart_service / stop_service are no-ops (server already up)

What is REAL (uses real HTTP):
  - ThroughputSkill / LatencySkill  -> sends requests to the local mock server
  - Reporter                        -> writes real results.tsv + best_config.md
  - Orchestrator                    -> runs the full Phase 2a + 2b loop
  - Runner                          -> executes all skills sequentially

Usage:
  # Terminal 1 - start mock server
  python demo/mock_server.py --port 18000

  # Terminal 2 - run integration test
  python demo/run_integration.py

  OR run everything in one shot (server auto-managed):
  python demo/run_integration.py --auto-server
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import MagicMock

# --- ensure repo root is on sys.path ---------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.mock_server import MockVLLMHandler  # local mock HTTP handler


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_mock_remote():
    """RemoteEnv mock: SSH run() always returns OK."""
    remote = MagicMock()
    remote.run.return_value = MagicMock(stdout="OK\n", stderr="", returncode=0)
    remote.read_log_tail.return_value = "[mock log] Service started OK\n"
    remote.run_background.return_value = None
    remote.close.return_value = None
    return remote


def _make_mock_actor(remote):
    """Actor mock: start/stop/restart are no-ops since server is already up."""
    actor = MagicMock()
    actor.remote = remote
    actor.fast_fail_check.return_value = True
    actor.restart_service.return_value = None
    actor.stop_service.return_value = None
    actor.start_service.return_value = None
    return actor


def _make_mock_observer():
    """Hardware observer mock: returns synthetic Ascend NPU stats."""
    from vllm_tuner.hardware.base import HardwareStats
    observer = MagicMock()
    observer.get_stats.return_value = HardwareStats(
        hbm_used_mb=20000,
        hbm_total_mb=65536,
        hbm_util_pct=30.5,
        aicore_util_pct=72.0,
        power_w=95.0,
        temp_c=52.0,
        health="OK",
    )
    observer.is_healthy.return_value = True
    return observer


class _MockBrain:
    """
    Deterministic Brain that cycles through a fixed list of Phase 2a / 2b configs.
    No Claude API calls -- works without ANTHROPIC_API_KEY.
    """
    _CONFIGS_2A = [
        {"block_size": 16, "gpu_memory_utilization": 0.85, "max_num_seqs": 128},
        {"block_size": 8,  "gpu_memory_utilization": 0.80, "max_num_seqs": 64},
        {"block_size": 32, "gpu_memory_utilization": 0.90, "max_num_seqs": 256},
    ]
    _CONFIGS_2B = [
        {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0},
        {"temperature": 0.8, "top_p": 0.90, "top_k": -1, "repetition_penalty": 1.05},
        {"temperature": 0.6, "top_p": 1.0,  "top_k": 20, "repetition_penalty": 1.1},
    ]
    _idx_2a = 0
    _idx_2b = 0

    def decide_next_config(self, sweep_result, history, hw_stats,
                           phase, param_space, seen_hashes):
        if phase == "2a":
            cfg = self._CONFIGS_2A[self._idx_2a % len(self._CONFIGS_2A)]
            self._idx_2a += 1
        else:
            cfg = self._CONFIGS_2B[self._idx_2b % len(self._CONFIGS_2B)]
            self._idx_2b += 1

        return {
            "next_config": cfg,
            "reasoning": f"[mock] cycling config for phase {phase}",
            "confidence": 0.9,
            "skip_reason": None,
            "optimization_focus": "speed" if phase == "2a" else "accuracy",
        }

    def diagnose(self, logs, error, attempt_history):
        return {
            "diagnosis": "[mock] no real diagnosis",
            "fix_commands": [],
            "adjusted_param": {},
        }


# --------------------------------------------------------------------------
# Minimal in-process TunerConfig (avoids need for a real YAML file)
# --------------------------------------------------------------------------

def _make_cfg(server_port: int, save_dir: str):
    """Build a TunerConfig-shaped object without loading YAML."""
    cfg = MagicMock()
    cfg.remote.host = "127.0.0.1"
    cfg.remote.port = 22
    cfg.remote.user = "mock"
    cfg.remote.vllm_port = server_port
    cfg.remote.working_dir = "/tmp/mock_workspace"
    cfg.hardware.type = "ascend"
    cfg.hardware.npu_id = 0
    cfg.hardware.chip_id = 0
    cfg.framework = "vllm"
    cfg.model.hf_url = "https://huggingface.co/Qwen/Qwen2-7B"
    cfg.model.local_path = "/tmp/mock_workspace/models/Qwen2-7B"
    cfg.model.hf_token = ""
    cfg.evaluation.script = "/evals/run.py"
    cfg.evaluation.data = "/datasets/"
    cfg.evaluation.metric = "accuracy"
    cfg.evaluation.metric_direction = "maximize"
    cfg.evaluation.sample_size = 5
    cfg.evaluation.timeout_seconds = 30
    cfg.evaluation.skills = ["throughput", "latency"]  # skip task_metric/memory for mock
    cfg.optimization.phase_2a.max_rounds = 3
    cfg.optimization.phase_2a.patience = 2
    cfg.optimization.phase_2a.parameters = {
        "block_size": [8, 16, 32],
        "gpu_memory_utilization": [0.80, 0.85, 0.90],
        "max_num_seqs": [64, 128, 256],
    }
    cfg.optimization.phase_2b.max_rounds = 3
    cfg.optimization.phase_2b.patience = 2
    cfg.optimization.phase_2b.baseline = {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0
    }
    cfg.optimization.phase_2b.parameters = {
        "temperature": [0.6, 0.7, 0.8],
        "top_p": [0.90, 0.95, 1.0],
    }
    cfg.save_dir = save_dir
    return cfg


# --------------------------------------------------------------------------
# Main integration runner
# --------------------------------------------------------------------------

def run_integration(server_port: int, save_dir: str):
    from vllm_tuner.reporter import Reporter
    from vllm_tuner.runner import Runner
    from vllm_tuner.orchestrator import Orchestrator
    from vllm_tuner.skills.throughput import ThroughputSkill
    from vllm_tuner.skills.latency import LatencySkill

    server_url = f"http://127.0.0.1:{server_port}/v1"
    health_url = f"http://127.0.0.1:{server_port}/health"

    print(f"\n{'-'*55}")
    print("  vLLM-Tuner  Integration Test  (Mock Mode)")
    print(f"{'-'*55}")
    print(f"  Server URL : {server_url}")
    print(f"  Save dir   : {save_dir}")
    print(f"{'-'*55}\n")

    # -- verify mock server is reachable --
    import requests
    for attempt in range(10):
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print(f"[OK] Mock server healthy at {health_url}")
                break
        except Exception:
            pass
        if attempt == 9:
            print(f"[FAIL] Cannot reach mock server at {health_url}. Is it running?")
            sys.exit(1)
        time.sleep(0.5)

    # -- build components --
    cfg = _make_cfg(server_port, save_dir)
    remote = _make_mock_remote()
    actor = _make_mock_actor(remote)
    observer = _make_mock_observer()
    brain = _MockBrain()

    # Only HTTP-based skills (no SSH needed)
    skills = [
        ThroughputSkill(concurrency=2, num_requests=4, input_len=64),
        LatencySkill(num_requests=3, input_len=64),
    ]
    reporter = Reporter(save_dir=save_dir)
    runner = Runner(skills=skills, server_url=server_url)

    orch = Orchestrator(
        config=cfg,
        actor=actor,
        brain=brain,
        runner=runner,
        reporter=reporter,
        sweep_result={},
        observer=observer,
    )

    # -- Phase 2a --
    print("--- Phase 2a: Infra Tuning (mock Brain, real HTTP) ---")
    best_infra = orch.run_phase_2a()
    print(f"\n[OK] Phase 2a done. Best infra config: {best_infra}")

    # -- Phase 2b --
    print("\n--- Phase 2b: Accuracy Tuning (mock Brain, real HTTP) ---")
    best_gen = orch.run_phase_2b(infra_config=best_infra)
    print(f"\n[OK] Phase 2b done. Best gen config: {best_gen}")

    # -- Phase 3: Report --
    print("\n--- Phase 3: Report ---")
    report_path = reporter.generate_best_report(
        model_name="Qwen2-7B",
        hardware_name="ASCEND",
        metric_direction="maximize",
    )
    print(f"[OK] Report written: {report_path}")

    # -- Verify outputs --
    tsv_path = Path(save_dir) / "results.tsv"
    rows = reporter.load_all()
    print(f"\n{'-'*55}")
    print(f"  Results summary")
    print(f"{'-'*55}")
    print(f"  Rows in results.tsv : {len(rows)}")
    keeps = [r for r in rows if r['status'] == 'KEEP']
    crashes = [r for r in rows if r['status'] == 'CRASH']
    print(f"  KEEP                : {len(keeps)}")
    print(f"  DISCARD             : {len(rows) - len(keeps) - len(crashes)}")
    print(f"  CRASH               : {len(crashes)}")
    if rows:
        sample = rows[0]
        print(f"\n  Sample row (round 1):")
        for k in ("round", "phase", "config_hash", "throughput", "latency_p99", "status"):
            print(f"    {k:15s}: {sample.get(k, '-')}")

    print(f"\n  best_config.md      : {report_path}")
    print(f"\n{'-'*55}")
    print("  ?  Integration test PASSED")
    print(f"{'-'*55}\n")
    return rows


def _start_server_thread(port: int):
    """Start mock server in a background daemon thread."""
    server = HTTPServer(("127.0.0.1", port), MockVLLMHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # Give it a moment to bind
    time.sleep(0.3)
    print(f"[MockServer] Auto-started on port {port} (background thread)")
    return server


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM-Tuner end-to-end integration test")
    parser.add_argument("--port", type=int, default=18000,
                        help="Mock server port (default: 18000)")
    parser.add_argument("--save-dir", default=None,
                        help="Results output dir (default: temp dir, auto-cleaned)")
    parser.add_argument("--auto-server", action="store_true",
                        help="Start mock server automatically in a background thread")
    parser.add_argument("--keep-results", action="store_true",
                        help="Don't delete the save-dir after the test")
    args = parser.parse_args()

    use_tmp = args.save_dir is None
    save_dir = args.save_dir or tempfile.mkdtemp(prefix="vllm_tuner_integration_")

    if args.auto_server:
        _start_server_thread(args.port)

    try:
        run_integration(server_port=args.port, save_dir=save_dir)
    finally:
        if use_tmp and not args.keep_results:
            shutil.rmtree(save_dir, ignore_errors=True)
        elif args.keep_results or not use_tmp:
            print(f"[Results kept at] {save_dir}")
