import time
import requests


class Actor:
    """Start/stop/restart vLLM service and run fast-fail health check."""

    def __init__(self, remote, framework, work_dir: str, host: str, port: int):
        pass

    def stop_service(self):
        """Gracefully stop the inference service, verify process name before killing."""
        raise NotImplementedError

    def start_service(self, model_path: str, infra_config: dict, health_timeout: int = 120):
        """Launch service in background and wait for /health to return 200."""
        raise NotImplementedError

    def restart_service(self, model_path: str, infra_config: dict, health_timeout: int = 120):
        """Stop then start the service with new infra_config."""
        raise NotImplementedError

    def fast_fail_check(self, num_probes: int = 3) -> bool:
        """Send 3 test requests; return True only if all succeed."""
        raise NotImplementedError

    def _wait_for_health(self, timeout: int = 120):
        """Poll /health every second until 200 or timeout."""
        raise NotImplementedError
