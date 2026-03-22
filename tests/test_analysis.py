# tests/test_analysis.py
import pytest
from vllm_tuner.analysis import is_pareto_dominated, get_pareto_frontier, select_bests

def make_rows(data):
    return [
        {"phase": r[0], "throughput": str(r[1]), "task_metric": str(r[2]),
         "latency_p99": str(r[3]), "status": "KEEP",
         "config_hash": f"h{i}", "round": str(i)}
        for i, r in enumerate(data)
    ]

def test_pareto_dominated_maximize():
    candidate = {"throughput": 800.0, "task_metric": 0.60}
    others = [{"throughput": 900.0, "task_metric": 0.65}]
    assert is_pareto_dominated(candidate, others, "maximize") is True

def test_pareto_not_dominated_tradeoff():
    candidate = {"throughput": 1000.0, "task_metric": 0.60}
    others = [{"throughput": 800.0, "task_metric": 0.70}]
    # neither dominates the other (tradeoff)
    assert is_pareto_dominated(candidate, others, "maximize") is False

def test_pareto_dominated_minimize():
    # lower task_metric is better for minimize
    candidate = {"throughput": 800.0, "task_metric": 0.50}
    others = [{"throughput": 900.0, "task_metric": 0.40}]
    assert is_pareto_dominated(candidate, others, "minimize") is True

def test_get_pareto_frontier_returns_non_dominated():
    rows = make_rows([
        ("2b", 800, 0.60, 150),  # dominated
        ("2b", 1000, 0.60, 130), # on frontier (best tps)
        ("2b", 800, 0.70, 150),  # on frontier (best accuracy)
    ])
    frontier = get_pareto_frontier(rows, "maximize")
    assert len(frontier) == 2

def test_select_bests_from_phase_2a():
    rows = make_rows([
        ("2a", 820, "-", 180),
        ("2a", 1050, "-", 145),
        ("2a", 950, "-", 200),
    ])
    # Fix task_metric for 2a rows to "-"
    for r in rows:
        r["task_metric"] = "-"
    bests = select_bests(rows)
    assert bests["best_latency"]["latency_p99"] == "145"
    assert bests["best_throughput"]["throughput"] == "1050"
