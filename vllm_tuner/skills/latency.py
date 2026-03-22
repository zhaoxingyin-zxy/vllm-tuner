import time
import requests
import statistics
from vllm_tuner.skills.base import EvalSkill


class LatencySkill(EvalSkill):
    """Measure P50/P99 latency (ms) via sequential requests."""

    name = "latency"

    def __init__(self, num_requests: int = 10, input_len: int = 256):
        pass

    def measure(self, server_url: str, gen_params: dict) -> dict:
        """Return dict with p50_ms and p99_ms."""
        raise NotImplementedError
