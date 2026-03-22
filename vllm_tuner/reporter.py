# vllm_tuner/reporter.py
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


def compute_config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:6]


FIELDNAMES = [
    "round", "phase", "config_hash", "config_json", "throughput", "latency_p99",
    "task_metric", "memory_pct", "status", "reasoning"
]


class Reporter:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = self.save_dir / "results.tsv"
        self._seen_hashes: set[str] = set()
        # Load existing hashes if file exists
        if self.tsv_path.exists():
            for row in self.load_all():
                self._seen_hashes.add(row["config_hash"])

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
        """Append a row. Returns False (skipped) if config_hash already seen."""
        h = compute_config_hash(config)
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)

        write_header = not self.tsv_path.exists()
        with open(self.tsv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "round": round_num,
                "phase": phase,
                "config_hash": h,
                "config_json": json.dumps(config, sort_keys=True),
                "throughput": f"{throughput:.1f}" if throughput is not None else "-",
                "latency_p99": f"{latency_p99:.0f}" if latency_p99 is not None else "-",
                "task_metric": f"{task_metric:.4f}" if task_metric is not None else "-",
                "memory_pct": f"{memory_pct:.2f}" if memory_pct is not None else "-",
                "status": status,
                "reasoning": reasoning,
            })
        return True

    def load_all(self) -> list[dict]:
        if not self.tsv_path.exists():
            return []
        with open(self.tsv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def generate_best_report(
        self,
        model_name: str,
        hardware_name: str,
        metric_direction: str,
    ) -> str:
        """Generate best_config.md with best latency, throughput, accuracy, Pareto rec."""
        raise NotImplementedError
