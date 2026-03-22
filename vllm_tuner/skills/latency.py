import time
import requests
import statistics
from vllm_tuner.skills.base import EvalSkill


class LatencySkill(EvalSkill):
    """Measure P50/P99 latency (ms) via sequential requests."""

    name = "latency"

    def __init__(self, num_requests: int = 10, input_len: int = 256):
        self.num_requests = num_requests
        self.input_len = input_len

    def measure(self, server_url: str, gen_params: dict) -> dict:
        """Return dict with p50_ms and p99_ms."""
        prompt = "x " * self.input_len
        payload = {"model": "default", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 64, **gen_params}
        latencies = []
        for _ in range(self.num_requests):
            start = time.time()
            requests.post(f"{server_url}/chat/completions", json=payload, timeout=60)
            latencies.append((time.time() - start) * 1000)
        latencies.sort()
        return {
            "p50_ms": statistics.median(latencies),
            "p99_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[-1],
        }
