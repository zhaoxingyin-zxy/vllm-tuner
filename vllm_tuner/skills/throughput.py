import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm_tuner.skills.base import EvalSkill


class ThroughputSkill(EvalSkill):
    """Measure throughput (tokens/s) via concurrent HTTP requests."""

    name = "throughput"

    def __init__(self, concurrency: int = 4, num_requests: int = 20, input_len: int = 256):
        pass

    def measure(self, server_url: str, gen_params: dict) -> float:
        """Return average tokens/s across concurrent requests."""
        raise NotImplementedError

    def _single_request(self, url: str, gen_params: dict) -> float:
        """Send one request, return tokens/s."""
        raise NotImplementedError
