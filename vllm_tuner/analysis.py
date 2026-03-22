from __future__ import annotations


def is_pareto_dominated(candidate: dict, others: list[dict], metric_direction: str) -> bool:
    """Return True if candidate is dominated by any row in others (throughput vs task_metric)."""
    raise NotImplementedError


def get_pareto_frontier(phase2b_rows: list[dict], metric_direction: str) -> list[dict]:
    """Return non-dominated Phase 2b configs. metric_direction: maximize or minimize."""
    raise NotImplementedError


def select_bests(all_rows: list[dict]) -> dict:
    """Select best_latency and best_throughput from Phase 2a KEEP rows."""
    raise NotImplementedError
