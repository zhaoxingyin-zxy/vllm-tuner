import fabric
from vllm_tuner.config import RemoteConfig


class RemoteEnv:
    """Single persistent SSH connection via fabric with auto-reconnect."""

    def __init__(self, config: RemoteConfig):
        pass

    def run(self, cmd: str, timeout: int = 60):
        """Execute command, return Result with .stdout .stderr .returncode."""
        raise NotImplementedError

    def run_background(self, cmd: str, log_file: str):
        """Non-blocking background execution, stdout+stderr redirected to log_file."""
        raise NotImplementedError

    def read_log_tail(self, log_file: str, lines: int = 50) -> str:
        """Return last N lines of remote log file."""
        raise NotImplementedError

    def close(self):
        """Close the SSH connection."""
        raise NotImplementedError
