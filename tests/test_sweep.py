# tests/test_sweep.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.sweep import BenchmarkSweep

def test_sweep_builds_matrix():
    with patch("vllm_tuner.sweep.requests.post") as mock_post, \
         patch("vllm_tuner.sweep.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "hi"}}],
                          "usage": {"completion_tokens": 10}}
        )
        sweep = BenchmarkSweep(
            server_url="http://1.2.3.4:8000/v1",
            health_url="http://1.2.3.4:8000/health",
            concurrency_levels=[1, 2],
            input_lengths=[128, 256],
            requests_per_cell=2,
            request_timeout=10,
        )
        result = sweep.run()
    assert "matrix" in result
    assert "1" in result["matrix"]
    assert "128" in result["matrix"]["1"]
    assert result["matrix"]["1"]["128"]["status"] == "OK"

def test_sweep_marks_oom_on_health_failure():
    call_count = [0]

    def post_side_effect(url, **kwargs):
        call_count[0] += 1
        if call_count[0] > 2:
            raise Exception("timeout")
        return MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "x"}}],
                          "usage": {"completion_tokens": 5}}
        )

    with patch("vllm_tuner.sweep.requests.post", side_effect=post_side_effect), \
         patch("vllm_tuner.sweep.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(status_code=200),  # healthy at start
            Exception("down"),           # health check after timeout → OOM
        ]
        sweep = BenchmarkSweep(
            server_url="http://1.2.3.4:8000/v1",
            health_url="http://1.2.3.4:8000/health",
            concurrency_levels=[1, 4],
            input_lengths=[128, 512],
            requests_per_cell=2,
            request_timeout=5,
        )
        result = sweep.run()
    # At least one OOM cell exists
    has_oom = any(
        cell.get("status") == "OOM"
        for conc_data in result["matrix"].values()
        for cell in conc_data.values()
    )
    assert has_oom
