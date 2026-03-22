import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


class BenchmarkSweep:
    """Phase 1: concurrency x input_len matrix sweep without restarting service."""

    def __init__(
        self,
        server_url: str,
        health_url: str,
        concurrency_levels: list[int],
        input_lengths: list[int],
        requests_per_cell: int,
        request_timeout: int,
    ):
        pass

    def run(self) -> dict:
        """Execute full sweep matrix, return results with OOM boundary and recommendations."""
        raise NotImplementedError

    def _is_healthy(self) -> bool:
        """Check if inference service is alive via GET /health."""
        raise NotImplementedError

    def _single_request(self, input_len: int) -> tuple[float, float]:
        """Send one request, return (latency_ms, tokens_per_sec)."""
        raise NotImplementedError

    def _measure_cell(self, concurrency: int, input_len: int) -> dict:
        """Measure one (concurrency, input_len) cell, detect OOM/ERROR/OK."""
        raise NotImplementedError

    def _recommend(self, matrix: dict, oom_boundary: dict | None) -> dict:
        """Recommend max_num_seqs and block_size search space from sweep results."""
        raise NotImplementedError
