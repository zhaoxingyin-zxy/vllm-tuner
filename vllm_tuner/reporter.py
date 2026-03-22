import csv
import hashlib
import json
import os
from pathlib import Path


def compute_config_hash(config: dict) -> str:
    """MD5 of sorted JSON params, first 6 hex chars."""
    raise NotImplementedError


FIELDNAMES = [
    "round", "phase", "config_hash", "config_json", "throughput", "latency_p99",
    "task_metric", "memory_pct", "status", "reasoning"
]


class Reporter:
    """Write results.tsv rows with dedup, generate best_config.md report."""

    def __init__(self, save_dir: str):
        pass

    def append_row(
        self,
        round_num: int,
        phase: str,
        config: dict,
        throughput: float | None,
        latency_p99: float | None,
        task_metric: float | None,
        memory_pct: float | None,
        status: str,
        reasoning: str,
    ) -> bool:
        """Append TSV row; skip and return False if config_hash already seen."""
        raise NotImplementedError

    def load_all(self) -> list[dict]:
        """Load all rows from results.tsv as list of dicts."""
        raise NotImplementedError

    def generate_best_report(
        self,
        model_name: str,
        hardware_name: str,
        metric_direction: str,
    ) -> str:
        """Generate best_config.md with best latency, throughput, accuracy, Pareto rec."""
        raise NotImplementedError
