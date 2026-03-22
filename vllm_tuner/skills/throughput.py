import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm_tuner.skills.base import EvalSkill


class ThroughputSkill(EvalSkill):
    """Measure throughput (tokens/s) via concurrent HTTP requests."""

    name = "throughput"

    def __init__(self, concurrency: int = 4, num_requests: int = 20, input_len: int = 256):
        self.concurrency = concurrency
        self.num_requests = num_requests
        self.input_len = input_len

    def _single_request(self, url: str, gen_params: dict) -> float:
        """Send one request, return tokens/s."""
        prompt = "x " * self.input_len
        payload = {"model": "default", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 64, **gen_params}
        start = time.time()
        resp = requests.post(f"{url}/chat/completions", json=payload, timeout=60)
        elapsed = time.time() - start
        tokens = resp.json().get("usage", {}).get("completion_tokens", 1)
        return tokens / max(elapsed, 0.001)

    def measure(self, server_url: str, gen_params: dict) -> float:
        """Return average tokens/s across concurrent requests."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._single_request, server_url, gen_params)
                       for _ in range(self.num_requests)]
            results = [f.result() for f in as_completed(futures)]
        return sum(results) / len(results)
