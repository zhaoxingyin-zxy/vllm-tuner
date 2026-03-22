# vllm_tuner/docker_manager.py
from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass
class ExecResult:
    """Return type of exec_run — mirrors fabric Result contract."""
    stdout: str
    stderr: str
    returncode: int


class DockerManager:
    """Encapsulates all docker CLI operations issued over SSH via RemoteEnv."""

    _SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_./:@-]+$')

    def __init__(self, remote, docker_cfg, hw_cfg):
        self.remote = remote
        self.cfg = docker_cfg
        self.hw = hw_cfg
        self._validate_config()

    def _validate_config(self):
        """Reject config values that could inject shell metacharacters."""
        for field_name, value in [
            ("container_name", self.cfg.container_name),
            ("image", self.cfg.image),
        ]:
            if not self._SAFE_PATTERN.match(value):
                raise ValueError(
                    f"DockerConfig.{field_name} contains unsafe characters: {value!r}. "
                    "Only alphanumerics, _, ., /, :, @, - are allowed."
                )

    def pull(self):
        """Pull image; run docker login first if registry is set."""
        if self.cfg.registry:
            result = self.remote.run(f"docker login {self.cfg.registry}", timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"docker login failed: {result.stderr}")
        result = self.remote.run(f"docker pull {self.cfg.image}", timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"docker pull failed: {result.stderr}")

    def is_image_present(self) -> bool:
        """Return True if image already exists locally on the remote host."""
        result = self.remote.run(f"docker images -q {self.cfg.image}", timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"docker images failed: {result.stderr}")
        return bool(result.stdout.strip())

    def is_container_running(self) -> bool:
        """Return True if container is currently running."""
        result = self.remote.run(
            f"docker ps --filter name=^/{self.cfg.container_name}$ --filter status=running -q",
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker ps failed: {result.stderr}")
        return bool(result.stdout.strip())

    def run_container(self, model_host_dir: str, vllm_port: int):
        """
        Remove any stale container with the same name, then start a fresh one.
        Container runs `sleep infinity` — vLLM is started separately via exec_background.
        No --rm flag: container is removed explicitly by stop_container() at session end.
        """
        # Remove stale container (running or stopped) if it exists
        stale = self.remote.run(
            f"docker ps -a --filter name={self.cfg.container_name} -q", timeout=15
        )
        if stale.stdout.strip():
            result = self.remote.run(f"docker rm -f {self.cfg.container_name}", timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"docker rm -f failed: {result.stderr}")

        cmd = (
            f"docker run -d"
            f" --name {self.cfg.container_name}"
            f" {self._device_flags()}"
            f" --shm-size {self.cfg.shm_size}"
            f" -v {model_host_dir}:{model_host_dir}:ro"
            f" -p {vllm_port}:{vllm_port}"
        )
        if self.cfg.extra_flags:
            cmd += f" {self.cfg.extra_flags}"
        cmd += f" {self.cfg.image} sleep infinity"

        result = self.remote.run(cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")

    def exec_background(self, cmd: str, log_file: str):
        """Run cmd inside the container as a background process."""
        # Escape single quotes in cmd and log_file before embedding in sh -c '...'
        safe_cmd = cmd.replace("'", "'\\''")
        safe_log = log_file.replace("'", "'\\''")
        exec_cmd = (
            f"docker exec {self.cfg.container_name} "
            f"sh -c 'nohup {safe_cmd} > {safe_log} 2>&1 &'"
        )
        result = self.remote.run(exec_cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker exec (background) failed: {result.stderr}"
            )

    def exec_run(self, cmd: str, timeout: int = 60) -> ExecResult:
        """Run cmd inside the container, return ExecResult(stdout, stderr, returncode)."""
        # Escape single quotes in cmd before embedding in sh -c '...'
        safe_cmd = cmd.replace("'", "'\\''")
        exec_cmd = f"docker exec {self.cfg.container_name} sh -c '{safe_cmd}'"
        result = self.remote.run(exec_cmd, timeout=timeout)
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )

    def stop_container(self):
        """Stop and remove the container. Errors suppressed (may already be gone)."""
        self.remote.run(f"docker stop {self.cfg.container_name} || true", timeout=30)
        self.remote.run(f"docker rm {self.cfg.container_name} || true", timeout=15)

    def _device_flags(self) -> str:
        """Return device flag string for docker run based on hardware type."""
        if self.hw.type == "ascend":
            if self.cfg.device_index >= 0:
                idx = self.cfg.device_index
            else:
                idx = self.hw.npu_id * 2 + self.hw.chip_id  # dual-die Atlas 300I Duo: 2 dies per card
            return (
                f"--device /dev/davinci{idx}"
                f" --device /dev/davinci_manager"
                f" --device /dev/hisi_hdc"
            )
        else:  # cuda
            return f'--gpus "device={self.hw.npu_id}"'
