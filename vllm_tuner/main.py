from __future__ import annotations
import argparse
import os
import sys

from vllm_tuner.config import load_config
from vllm_tuner.remote_env import RemoteEnv
from vllm_tuner.frameworks import get_framework
from vllm_tuner.hardware import get_observer
from vllm_tuner.actor import Actor
from vllm_tuner.shipit import ShipIt
from vllm_tuner.sweep import BenchmarkSweep
from vllm_tuner.brain import Brain
from vllm_tuner.runner import Runner
from vllm_tuner.reporter import Reporter
from vllm_tuner.orchestrator import Orchestrator
from vllm_tuner.skills import ThroughputSkill, LatencySkill, TaskMetricSkill, MemorySkill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-tuner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Full optimization pipeline")
    run_p.add_argument("--config", required=True, help="Path to tuner_config.yaml")
    run_p.add_argument(
        "--server-url", default=None,
        help="Skip Phase 0 (deploy) and Phase 1 (sweep). Use this pre-deployed server URL directly.",
    )

    deploy_p = sub.add_parser("deploy", help="Deploy only (Ship-It)")
    deploy_p.add_argument("--config", required=True)
    deploy_p.add_argument("--use-best-config", default=None,
                          help="Read best infra+gen params from this report file")
    return parser


def _build_skills(cfg, remote, observer):
    skill_map = {
        "throughput": ThroughputSkill(concurrency=4, num_requests=20, input_len=256),
        "latency": LatencySkill(num_requests=10, input_len=256),
        "task_metric": TaskMetricSkill(
            remote=remote,
            script=cfg.evaluation.script,
            data_dir=cfg.evaluation.data,
            sample_size=cfg.evaluation.sample_size,
            timeout_seconds=cfg.evaluation.timeout_seconds,
        ),
        "memory": MemorySkill(observer=observer),
    }
    return [skill_map[s] for s in cfg.evaluation.skills if s in skill_map]


def _load_best_infra_from_report(report_path: str, save_dir: str):
    """
    Parse best_config.md to find the 'Best Throughput' config hash,
    then look it up in results.tsv (config_json column) to get the full param dict.
    Returns the infra param dict, or None if not found.
    """
    import re
    import csv
    import json
    from pathlib import Path

    report = Path(report_path).read_text(encoding="utf-8")
    m = re.search(r"### \u2462 Best Throughput.*?Config hash: (\w+)", report, re.DOTALL)
    if not m:
        return None
    target_hash = m.group(1)

    tsv_path = Path(save_dir) / "results.tsv"
    if not tsv_path.exists():
        return None

    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("config_hash") == target_hash and row.get("phase") == "2a":
                config_json = row.get("config_json", "")
                if config_json and config_json != "-":
                    try:
                        return json.loads(config_json)
                    except json.JSONDecodeError:
                        pass
    return None


def cmd_run(args):
    cfg = load_config(args.config)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    server_url = args.server_url  # None means full pipeline; set means skip Phase 0+1

    remote = RemoteEnv(cfg.remote)
    framework = get_framework(cfg.framework, host=cfg.remote.host, port=cfg.remote.vllm_port)
    observer = get_observer(cfg.hardware.type, remote,
                            npu_id=cfg.hardware.npu_id, chip_id=cfg.hardware.chip_id)
    actor = Actor(remote=remote, framework=framework,
                  work_dir=cfg.remote.working_dir,
                  host=cfg.remote.host, port=cfg.remote.vllm_port)
    brain = Brain(api_key=api_key)
    reporter = Reporter(save_dir=cfg.save_dir)

    if server_url is None:
        # Phase 0: Ship-It
        print("\n─── Phase 0: Ship-It ───")
        ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg)
        default_infra = {
            "block_size": cfg.optimization.phase_2a.parameters["block_size"][0],
            "gpu_memory_utilization": cfg.optimization.phase_2a.parameters.get(
                "gpu_memory_utilization", [0.85])[0],
        }
        ship.run(default_infra)

        # Phase 1: Sweep
        print("\n─── Phase 1: Sweep ───")
        api_base = framework.get_api_base()
        sweep = BenchmarkSweep(
            server_url=api_base,
            health_url=framework.get_health_endpoint(),
            concurrency_levels=cfg.sweep.concurrency_levels,
            input_lengths=cfg.sweep.input_lengths,
            requests_per_cell=cfg.sweep.requests_per_cell,
            request_timeout=cfg.sweep.request_timeout_seconds,
        )
        sweep_result = sweep.run()
        print(f"[Sweep] Best: {sweep_result.get('best_throughput')}")
        server_url = api_base
    else:
        print(f"\n[--server-url] Skipping Phase 0 (deploy) and Phase 1 (sweep).")
        print(f"[--server-url] Using pre-deployed server: {server_url}")
        sweep_result = {}

    skills = _build_skills(cfg, remote, observer)
    runner = Runner(skills=skills, server_url=server_url)

    # Phase 2a + 2b
    orch = Orchestrator(
        config=cfg, actor=actor, brain=brain,
        runner=runner, reporter=reporter,
        sweep_result=sweep_result, observer=observer,
    )
    print("\n─── Phase 2a: Infra Tuning ───")
    best_infra = orch.run_phase_2a()

    print("\n─── Phase 2b: Accuracy Tuning ───")
    orch.run_phase_2b(infra_config=best_infra)

    # Phase 3: Report
    print("\n─── Phase 3: Report ───")
    model_name = cfg.model.hf_url.split("/")[-1]
    report_path = reporter.generate_best_report(
        model_name=model_name,
        hardware_name=cfg.hardware.type.upper(),
        metric_direction=cfg.evaluation.metric_direction,
    )
    print(f"[✓] Report saved: {report_path}")
    print(f"[✓] Service: {server_url}")
    remote.close()


def cmd_deploy(args):
    cfg = load_config(args.config)
    remote = RemoteEnv(cfg.remote)
    framework = get_framework(cfg.framework, host=cfg.remote.host, port=cfg.remote.vllm_port)
    actor = Actor(remote=remote, framework=framework,
                  work_dir=cfg.remote.working_dir,
                  host=cfg.remote.host, port=cfg.remote.vllm_port)
    ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg)

    infra = {
        "block_size": cfg.optimization.phase_2a.parameters["block_size"][0],
        "gpu_memory_utilization": cfg.optimization.phase_2a.parameters.get(
            "gpu_memory_utilization", [0.85])[0],
    }

    if args.use_best_config:
        print(f"[Deploy] Loading best config from {args.use_best_config}")
        best = _load_best_infra_from_report(args.use_best_config, cfg.save_dir)
        if best:
            infra.update(best)
        else:
            print("[Deploy] Could not parse best infra params; using config defaults.")

    ship.run(infra)
    remote.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "deploy":
        cmd_deploy(args)


if __name__ == "__main__":
    main()
