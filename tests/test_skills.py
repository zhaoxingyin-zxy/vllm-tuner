# tests/test_skills.py
import json
import pytest
from unittest.mock import MagicMock, patch
from vllm_tuner.skills.throughput import ThroughputSkill
from vllm_tuner.skills.latency import LatencySkill
from vllm_tuner.skills.task_metric import TaskMetricSkill
from vllm_tuner.skills.memory import MemorySkill

MOCK_RESPONSE = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"completion_tokens": 10}
}

def make_mock_post(response_json=MOCK_RESPONSE, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_json
    return resp

def test_throughput_returns_tokens_per_second():
    with patch("vllm_tuner.skills.throughput.requests.post") as mock_post:
        mock_post.return_value = make_mock_post()
        skill = ThroughputSkill(concurrency=2, num_requests=4, input_len=128)
        tps = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
        assert tps > 0

def test_latency_returns_p99():
    with patch("vllm_tuner.skills.latency.requests.post") as mock_post:
        mock_post.return_value = make_mock_post()
        skill = LatencySkill(num_requests=5, input_len=128)
        result = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
        assert "p50_ms" in result
        assert "p99_ms" in result
        assert result["p99_ms"] >= result["p50_ms"]

def test_task_metric_parses_json_stdout():
    remote = MagicMock()
    remote.run.return_value = MagicMock(
        stdout='{"metric": 0.678}\n', stderr="", returncode=0
    )
    skill = TaskMetricSkill(
        remote=remote,
        script="/evals/run.py",
        data_dir="/data/",
        sample_size=50,
        timeout_seconds=60,
    )
    val = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
    assert abs(val - 0.678) < 1e-6

def test_task_metric_raises_on_bad_stdout():
    remote = MagicMock()
    remote.run.return_value = MagicMock(stdout="not json", stderr="", returncode=0)
    skill = TaskMetricSkill(remote=remote, script="/e.py", data_dir="/d/",
                             sample_size=10, timeout_seconds=60)
    with pytest.raises(ValueError, match="metric"):
        skill.measure("http://1.2.3.4:8000/v1", gen_params={})

def test_memory_skill_returns_hbm_pct():
    observer = MagicMock()
    from vllm_tuner.hardware.base import HardwareStats
    observer.get_stats.return_value = HardwareStats(
        hbm_used_mb=3161, hbm_total_mb=65536, hbm_util_pct=4.8,
        aicore_util_pct=87, power_w=93.6, temp_c=40, health="OK"
    )
    from vllm_tuner.skills.memory import MemorySkill
    skill = MemorySkill(observer=observer)
    pct = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
    assert abs(pct - 4.8) < 0.01
