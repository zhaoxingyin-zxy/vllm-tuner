# tests/test_orchestrator.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.orchestrator import Orchestrator

def make_orchestrator():
    cfg = MagicMock()
    cfg.model.local_path = "/models/Qwen2-7B"
    cfg.remote.host = "1.2.3.4"
    cfg.remote.vllm_port = 8000
    cfg.evaluation.metric_direction = "maximize"
    cfg.optimization.phase_2a.max_rounds = 3
    cfg.optimization.phase_2a.patience = 2
    cfg.optimization.phase_2a.parameters = {"block_size": [8, 16]}
    cfg.optimization.phase_2b.max_rounds = 3
    cfg.optimization.phase_2b.patience = 2
    cfg.optimization.phase_2b.baseline = {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0
    }
    cfg.optimization.phase_2b.parameters = {"temperature": [0.7, 0.8]}
    cfg.evaluation.skills = ["throughput", "latency"]

    actor = MagicMock()
    actor.fast_fail_check.return_value = True

    brain = MagicMock()
    brain.decide_next_config.return_value = {
        "next_config": {"block_size": 8},
        "reasoning": "test",
        "confidence": 0.9,
        "skip_reason": None,
        "optimization_focus": "speed",
    }

    runner = MagicMock()
    runner.run_all.return_value = {
        "throughput": 900.0, "latency_p99": 130.0,
        "task_metric": None, "memory_pct": 0.8
    }

    reporter = MagicMock()
    reporter.append_row.return_value = True
    reporter.load_all.return_value = []

    return Orchestrator(
        config=cfg, actor=actor, brain=brain,
        runner=runner, reporter=reporter,
        sweep_result={}, observer=MagicMock()
    )

def test_phase_2a_runs_at_most_max_rounds():
    orch = make_orchestrator()
    best = orch.run_phase_2a()
    assert orch.actor.restart_service.call_count <= 3

def test_phase_2a_returns_best_infra_config():
    orch = make_orchestrator()
    best = orch.run_phase_2a()
    assert "block_size" in best or best == {}
