# vllm_tuner/actor.py
import time
import requests


class Actor:
    def __init__(self, remote, framework, work_dir: str, host: str, port: int):
        self.remote = remote
        self.framework = framework
        self.work_dir = work_dir
        self.host = host
        self.port = port

    def stop_service(self):
        """Gracefully stop the inference service and free the port."""
        # Get PID and process name for the port
        result = self.remote.run(f"lsof -ti:{self.port} -sTCP:LISTEN", timeout=10)
        pid = result.stdout.strip()
        if not pid:
            return  # Nothing running

        # Verify process name is safe to kill
        name_result = self.remote.run(f"ps -p {pid} -o comm=", timeout=10)
        proc_name = name_result.stdout.strip().lower()
        safe_names = ("python", "vllm", "lmdeploy", "sglang")
        if not any(n in proc_name for n in safe_names):
            raise RuntimeError(
                f"Refusing to kill PID {pid} (process: {proc_name!r}). "
                "Not a recognized inference server process."
            )

        self.remote.run(f"kill -15 {pid}", timeout=10)
        time.sleep(3)
        # Force kill if still alive
        check = self.remote.run(f"ps -p {pid} -o pid=", timeout=10)
        if check.stdout.strip():
            self.remote.run(f"kill -9 {pid}", timeout=10)

    def start_service(self, model_path: str, infra_config: dict, health_timeout: int = 120):
        """Launch service with given infra_config, wait for health OK."""
        cmd = self.framework.build_start_cmd(model_path, infra_config)
        log_file = f"{self.work_dir}/vllm.log"
        self.remote.run_background(cmd, log_file)
        self._wait_for_health(timeout=health_timeout)

    def restart_service(self, model_path: str, infra_config: dict, health_timeout: int = 120):
        self.stop_service()
        time.sleep(2)
        self.start_service(model_path, infra_config, health_timeout)

    def fast_fail_check(self, num_probes: int = 3) -> bool:
        """Send 3 test requests. Return True only if all succeed."""
        health_url = self.framework.get_health_endpoint()
        try:
            r = requests.get(health_url, timeout=10)
            if r.status_code != 200:
                return False
        except Exception:
            return False

        api_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        payload = {"model": "default",
                   "messages": [{"role": "user", "content": "Hello"}],
                   "max_tokens": 5}
        for _ in range(num_probes):
            try:
                resp = requests.post(api_url, json=payload, timeout=30)
                if resp.status_code != 200:
                    return False
                if not resp.json().get("choices"):
                    return False
            except Exception:
                return False
        return True

    def _wait_for_health(self, timeout: int = 120):
        health_url = self.framework.get_health_endpoint()
        for _ in range(timeout):
            try:
                if requests.get(health_url, timeout=3).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(
            f"Service did not become healthy within {timeout}s. "
            f"Check {self.work_dir}/vllm.log on the remote machine."
        )
