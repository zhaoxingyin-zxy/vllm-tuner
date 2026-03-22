import fabric
from vllm_tuner.config import RemoteConfig


class RemoteEnv:
    """Single persistent SSH connection via fabric with auto-reconnect."""

    def __init__(self, config: RemoteConfig):
        self.config = config
        self.conn = fabric.Connection(
            host=config.host,
            port=config.port,
            user=config.user,
            connect_kwargs={"key_filename": config.key_file},
        )

    def run(self, cmd: str, timeout: int = 60):
        """Execute command, return Result with .stdout .stderr .returncode."""
        return self.conn.run(cmd, warn=True, hide=True, timeout=timeout)

    def run_background(self, cmd: str, log_file: str):
        """Non-blocking background execution, stdout+stderr redirected to log_file."""
        self.conn.run(f"nohup {cmd} > {log_file} 2>&1 &", warn=True, hide=True)

    def read_log_tail(self, log_file: str, lines: int = 50) -> str:
        """Return last N lines of remote log file."""
        result = self.run(f"tail -{lines} {log_file}")
        return result.stdout

    def close(self):
        """Close the SSH connection."""
        self.conn.close()
