# vllm_tuner/sweep.py
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
import requests


class BenchmarkSweep:
    def __init__(
        self,
        server_url: str,
        health_url: str,
        concurrency_levels: list,
        input_lengths: list,
        requests_per_cell: int,
        request_timeout: int,
    ):
        self.server_url = server_url
        self.health_url = health_url
        self.concurrency_levels = concurrency_levels
        self.input_lengths = input_lengths
        self.requests_per_cell = requests_per_cell
        self.request_timeout = request_timeout

    def _is_healthy(self) -> bool:
        try:
            return requests.get(self.health_url, timeout=5).status_code == 200
        except Exception:
            return False

    def _single_request(self, input_len: int) -> tuple:
        """Returns (latency_ms, tokens_per_sec)."""
        prompt = "x " * input_len
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
        }
        start = time.time()
        resp = requests.post(
            f"{self.server_url}/chat/completions",
            json=payload,
            timeout=self.request_timeout,
        )
        elapsed = time.time() - start
        tokens = resp.json().get("usage", {}).get("completion_tokens", 1)
        return elapsed * 1000, tokens / max(elapsed, 0.001)

    def _measure_cell(self, concurrency: int, input_len: int) -> dict:
        latencies, tps_list = [], []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(self._single_request, input_len)
                for _ in range(self.requests_per_cell)
            ]
            for future in as_completed(futures, timeout=self.request_timeout * 2):
                try:
                    lat, tps = future.result()
                    latencies.append(lat)
                    tps_list.append(tps)
                except Exception:
                    # Check if service crashed
                    if not self._is_healthy():
                        return {"status": "OOM"}
                    return {"status": "ERROR"}

        if not latencies:
            return {"status": "ERROR"}

        sorted_lat = sorted(latencies)
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[-1]
        return {
            "status": "OK",
            "p99_ms": round(p99, 1),
            "tps": round(sum(tps_list) / len(tps_list), 1),
        }

    def run(self) -> dict:
        matrix = {}
        best_tps = 0
        best_cell = {}
        oom_boundary = None

        for conc in self.concurrency_levels:
            matrix[str(conc)] = {}
            oom_hit = False

            for input_len in self.input_lengths:
                if oom_hit:
                    matrix[str(conc)][str(input_len)] = {"status": "SKIPPED"}
                    continue

                cell = self._measure_cell(conc, input_len)
                matrix[str(conc)][str(input_len)] = cell

                if cell["status"] == "OOM":
                    oom_hit = True
                    if oom_boundary is None:
                        oom_boundary = {"concurrency": conc, "input_len": input_len}
                elif cell["status"] == "OK" and cell.get("tps", 0) > best_tps:
                    best_tps = cell["tps"]
                    best_cell = {"concurrency": conc, "input_len": input_len, "tps": best_tps}

        return {
            "matrix": matrix,
            "oom_boundary": oom_boundary,
            "best_throughput": best_cell,
            "recommended_search_space": self._recommend(matrix, oom_boundary),
        }

    def _recommend(self, matrix: dict, oom_boundary) -> dict:
        safe_concurrencies = [
            int(c) for c, cells in matrix.items()
            if any(v.get("status") == "OK" for v in cells.values())
        ]
        return {
            "max_num_seqs": safe_concurrencies[:3] if safe_concurrencies else [1],
            "block_size": [8, 16, 32],
            "note": f"OOM boundary: {oom_boundary}" if oom_boundary else "No OOM observed",
        }
