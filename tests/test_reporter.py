# tests/test_reporter.py
import os, csv
import pytest
from vllm_tuner.reporter import Reporter, compute_config_hash

def test_config_hash_is_deterministic():
    cfg = {"block_size": 8, "gpu_memory_utilization": 0.85}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash({"gpu_memory_utilization": 0.85, "block_size": 8})
    assert h1 == h2
    assert len(h1) == 6

def test_config_hash_differs_for_different_configs():
    h1 = compute_config_hash({"block_size": 8})
    h2 = compute_config_hash({"block_size": 16})
    assert h1 != h2

def test_reporter_writes_tsv_row(tmp_path):
    tsv = tmp_path / "results.tsv"
    reporter = Reporter(save_dir=str(tmp_path))
    reporter.append_row(
        round_num=1, phase="2a",
        config={"block_size": 8, "gpu_memory_utilization": 0.85},
        throughput=820.1, latency_p99=180.0,
        task_metric=None, memory_pct=0.72,
        status="KEEP", reasoning="baseline"
    )
    assert tsv.exists()
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["status"] == "KEEP"
    assert rows[0]["phase"] == "2a"
    assert rows[0]["task_metric"] == "-"

def test_reporter_skips_duplicate_hash(tmp_path):
    reporter = Reporter(save_dir=str(tmp_path))
    cfg = {"block_size": 8}
    reporter.append_row(1, "2a", cfg, 820, 180, None, 0.7, "KEEP", "r1")
    result = reporter.append_row(2, "2a", cfg, 850, 170, None, 0.7, "KEEP", "r2")
    assert result is False  # skipped duplicate

def test_reporter_load_all_rounds(tmp_path):
    reporter = Reporter(save_dir=str(tmp_path))
    reporter.append_row(1, "2a", {"block_size": 8}, 820, 180, None, 0.72, "KEEP", "b")
    reporter.append_row(2, "2a", {"block_size": 16}, 900, 160, None, 0.80, "KEEP", "opt")
    rows = reporter.load_all()
    assert len(rows) == 2
    assert rows[0]["round"] == "1"

def test_reporter_stores_config_json(tmp_path):
    """config_json column enables --use-best-config round-trip."""
    import json
    reporter = Reporter(save_dir=str(tmp_path))
    cfg = {"block_size": 8, "gpu_memory_utilization": 0.88}
    reporter.append_row(1, "2a", cfg, 1050, 115, None, 0.83, "KEEP", "best")
    rows = reporter.load_all()
    assert "config_json" in rows[0]
    restored = json.loads(rows[0]["config_json"])
    assert restored == cfg
