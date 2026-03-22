# vllm_tuner/analysis.py
from __future__ import annotations


def is_pareto_dominated(candidate: dict, others: list, metric_direction: str) -> bool:
    """
    Returns True if candidate is dominated by any row in others.
    For maximize: higher throughput and task_metric is better.
    For minimize: higher throughput, LOWER task_metric is better.
    """
    def not_worse_metric(other_val, cand_val):
        if metric_direction == "maximize":
            return other_val >= cand_val
        else:  # minimize
            return other_val <= cand_val

    def strictly_better_metric(other_val, cand_val):
        if metric_direction == "maximize":
            return other_val > cand_val
        else:
            return other_val < cand_val

    cand_tps = candidate["throughput"]
    cand_metric = candidate["task_metric"]

    for other in others:
        if other is candidate:
            continue
        o_tps = other["throughput"]
        o_metric = other["task_metric"]

        tps_not_worse = o_tps >= cand_tps
        metric_not_worse = not_worse_metric(o_metric, cand_metric)
        tps_better = o_tps > cand_tps
        metric_better = strictly_better_metric(o_metric, cand_metric)

        if tps_not_worse and metric_not_worse and (tps_better or metric_better):
            return True
    return False


def get_pareto_frontier(phase2b_rows: list, metric_direction: str) -> list:
    """Only Phase 2b rows with numeric task_metric. Returns non-dominated configs."""
    valid = []
    for r in phase2b_rows:
        if r.get("phase") != "2b" or r.get("task_metric") == "-":
            continue
        if r.get("status") not in ("KEEP", "DISCARD"):
            continue
        try:
            valid.append({
                **r,
                "throughput": float(r["throughput"]),
                "task_metric": float(r["task_metric"]),
            })
        except (ValueError, TypeError):
            continue

    return [r for r in valid if not is_pareto_dominated(r, valid, metric_direction)]


def select_bests(all_rows: list) -> dict:
    """Select best_latency and best_throughput from Phase 2a KEEP rows."""
    phase2a = [
        r for r in all_rows
        if r.get("phase") == "2a" and r.get("status") == "KEEP"
        and r.get("throughput") not in (None, "-")
        and r.get("latency_p99") not in (None, "-")
    ]
    if not phase2a:
        return {"best_latency": None, "best_throughput": None}

    best_lat = min(phase2a, key=lambda r: float(r["latency_p99"]))
    best_tps = max(phase2a, key=lambda r: float(r["throughput"]))
    return {"best_latency": best_lat, "best_throughput": best_tps}
