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
        """Generate best_config.md. Returns path to written file."""
        from vllm_tuner.analysis import get_pareto_frontier, select_bests
        rows = self.load_all()
        bests = select_bests(rows)
        frontier = get_pareto_frontier(rows, metric_direction)

        lines = [
            "# vLLM-Tuner Optimization Report",
            f"## Model: {model_name} | Hardware: {hardware_name}",
            "",
        ]

        b_lat = bests.get("best_latency")
        b_tps = bests.get("best_throughput")
        if b_lat:
            lines += [
                f"### Best Latency (Round {b_lat['round']})",
                f"Config hash: {b_lat['config_hash']}",
                f"P99 latency: {b_lat['latency_p99']}ms",
                f"Config: {b_lat.get('config_json', '-')}",
                "",
            ]
        if b_tps:
            lines += [
                f"### Best Throughput (Round {b_tps['round']})",
                f"Config hash: {b_tps['config_hash']}",
                f"Throughput: {b_tps['throughput']} tok/s",
                f"Config: {b_tps.get('config_json', '-')}",
                "",
            ]

        phase2b = [
            r for r in rows
            if r.get("phase") == "2b" and r.get("status") == "KEEP"
            and r.get("task_metric") not in (None, "-")
        ]
        if phase2b:
            best_acc = (max if metric_direction == "maximize" else min)(
                phase2b, key=lambda r: float(r["task_metric"])
            )
            lines += [
                f"### Best Accuracy (Round {best_acc['round']})",
                f"Config hash: {best_acc['config_hash']}",
                f"Task metric: {best_acc['task_metric']}",
                f"Config: {best_acc.get('config_json', '-')}",
                "",
            ]

        if frontier:
            rec = frontier[0]
            lines += [
                "### Pareto Recommendation",
                f"Round {rec['round']}: throughput={rec['throughput']:.0f} tok/s, "
                f"metric={rec['task_metric']:.4f}",
                f"Config: {rec.get('config_json', '-')}",
                "",
            ]

        if not b_lat and not b_tps and not phase2b:
            lines += ["(No results to report yet.)", ""]

        report_path = self.save_dir / "best_config.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)
