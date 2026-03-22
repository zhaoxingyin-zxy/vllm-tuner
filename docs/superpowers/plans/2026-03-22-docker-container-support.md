# Docker Container Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional Docker container support so vLLM runs inside a container on the remote machine, while preserving backward compatibility when no `docker:` block is present in the config.

**Architecture:** A new `DockerManager` class encapsulates all `docker` CLI calls (pull, run, exec, stop). `Actor` and `ShipIt` each gain an optional `docker_manager` parameter; when set they delegate to it, when `None` they use the existing bare-process logic unchanged. `config.py` gains a `DockerConfig` dataclass and `load_config` parses it from YAML.

**Tech Stack:** Python 3.9, dataclasses, `fabric` (SSH via `RemoteEnv`), `docker` CLI on remote machine (called over SSH), `pytest` + `unittest.mock`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm_tuner/config.py` | Modify | Add `DockerConfig` dataclass; add `docker` field to `TunerConfig`; update `load_config` |
| `vllm_tuner/docker_manager.py` | **Create** | All `docker` CLI operations over SSH |
| `vllm_tuner/actor.py` | Modify | Accept `docker_manager=None`; branch `start_service` and `stop_service` |
| `vllm_tuner/shipit.py` | Modify | Accept `docker_manager=None`; branch `check_remote_env`, `auto_fix_env`, `run` |
| `vllm_tuner/main.py` | Modify | Assemble `DockerManager` in both `cmd_run` and `cmd_deploy`; `finally` cleanup |
| `tests/test_docker_manager.py` | **Create** | Unit tests for `DockerManager` |
| `tests/test_actor.py` | Modify | Add Docker-branch tests |
| `tests/test_shipit.py` | Modify | Add Docker-branch tests |

---

## Task 1: Add `DockerConfig` to `config.py`

**Files:**
- Modify: `vllm_tuner/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_docker.py`:

```python
# tests/test_config_docker.py
from vllm_tuner.config import DockerConfig, TunerConfig
import pytest

def test_docker_config_defaults():
    dc = DockerConfig(image="vllm:latest", container_name="vllm_tuner")
    assert dc.shm_size == "8g"
    assert dc.registry == ""
    assert dc.extra_flags == ""
    assert dc.device_index == -1

def test_tuner_config_docker_is_optional():
    # TunerConfig.docker defaults to None — no docker block needed
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(TunerConfig)}
    assert "docker" in fields
    assert fields["docker"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/dev_project/work_proj
python -m pytest tests/test_config_docker.py -v
```

Expected: `ImportError: cannot import name 'DockerConfig'`

- [ ] **Step 3: Add `DockerConfig` to `config.py`**

Insert after the existing imports at the top of `vllm_tuner/config.py`, before `class RemoteConfig`:

```python
@dataclass
class DockerConfig:
    image: str            # e.g. "vllm-ascend:v0.9.0"
    container_name: str   # e.g. "vllm_tuner_serving"
    shm_size: str = "8g"
    registry: str = ""    # private registry prefix; empty = no login
    extra_flags: str = "" # appended verbatim to docker run
    device_index: int = -1  # Ascend /dev/davinciN; -1 = auto (npu_id*2+chip_id)
```

Then update `TunerConfig` — add `docker` field with default `None`:

```python
@dataclass
class TunerConfig:
    remote: RemoteConfig
    hardware: HardwareConfig
    framework: str
    model: ModelConfig
    evaluation: EvaluationConfig
    sweep: SweepConfig
    optimization: OptimizationConfig
    save_dir: str
    docker: DockerConfig = None  # Optional; None = bare-process mode
```

Then update `load_config` — add docker parsing just before the `return TunerConfig(...)` call:

```python
    docker_cfg = None
    if "docker" in raw:
        docker_cfg = DockerConfig(**raw["docker"])

    return TunerConfig(
        remote=remote,
        hardware=hardware,
        framework=raw["framework"],
        model=model,
        evaluation=evaluation,
        sweep=sweep,
        optimization=optimization,
        save_dir=raw["save_dir"],
        docker=docker_cfg,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config_docker.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/config.py tests/test_config_docker.py
git commit -m "feat: add DockerConfig dataclass and optional docker field to TunerConfig"
```

---

## Task 2: Create `docker_manager.py`

**Files:**
- Create: `vllm_tuner/docker_manager.py`
- Create: `tests/test_docker_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_docker_manager.py`:

```python
# tests/test_docker_manager.py
from unittest.mock import MagicMock, call
import pytest
from vllm_tuner.docker_manager import DockerManager, ExecResult
from vllm_tuner.config import DockerConfig, HardwareConfig


def make_mgr(hw_type="ascend", npu_id=0, chip_id=0, device_index=-1, registry=""):
    remote = MagicMock()
    remote.run.return_value = MagicMock(stdout="", stderr="", returncode=0)
    docker_cfg = DockerConfig(
        image="vllm-ascend:latest",
        container_name="vllm_tuner",
        shm_size="8g",
        registry=registry,
        extra_flags="",
        device_index=device_index,
    )
    hw_cfg = HardwareConfig(type=hw_type, npu_id=npu_id, chip_id=chip_id)
    return DockerManager(remote=remote, docker_cfg=docker_cfg, hw_cfg=hw_cfg), remote


def test_is_image_present_true():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="abc123\n", returncode=0)
    assert mgr.is_image_present() is True


def test_is_image_present_false():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="", returncode=0)
    assert mgr.is_image_present() is False


def test_is_container_running_true():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="deadbeef\n", returncode=0)
    assert mgr.is_container_running() is True


def test_is_container_running_false():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="", returncode=0)
    assert mgr.is_container_running() is False


def test_pull_without_registry():
    mgr, remote = make_mgr()
    mgr.pull()
    cmds = [str(c) for c in remote.run.call_args_list]
    assert not any("docker login" in c for c in cmds)
    assert any("docker pull vllm-ascend:latest" in c for c in cmds)


def test_pull_with_registry_calls_login_first():
    mgr, remote = make_mgr(registry="registry.example.com")
    mgr.pull()
    cmds = [str(c) for c in remote.run.call_args_list]
    assert any("docker login registry.example.com" in c for c in cmds)
    assert any("docker pull" in c for c in cmds)
    # login must come before pull
    login_idx = next(i for i, c in enumerate(cmds) if "docker login" in c)
    pull_idx = next(i for i, c in enumerate(cmds) if "docker pull" in c)
    assert login_idx < pull_idx


def test_run_container_removes_stale_then_runs(monkeypatch):
    mgr, remote = make_mgr()
    # Simulate stale container exists: docker ps -a returns a line
    remote.run.side_effect = [
        MagicMock(stdout="old_id\n", returncode=0),  # docker ps -a (stale check)
        MagicMock(stdout="", returncode=0),           # docker rm -f
        MagicMock(stdout="new_id\n", returncode=0),  # docker run
    ]
    mgr.run_container("/models", 8000)
    cmds = [str(c) for c in remote.run.call_args_list]
    assert any("docker rm -f" in c for c in cmds)
    assert any("docker run" in c for c in cmds)
    assert any("sleep infinity" in c for c in cmds)
    assert any("-p 8000:8000" in c for c in cmds)
    assert any("-v /models:/models:ro" in c for c in cmds)


def test_run_container_no_stale():
    mgr, remote = make_mgr()
    remote.run.side_effect = [
        MagicMock(stdout="", returncode=0),          # docker ps -a (no stale)
        MagicMock(stdout="new_id\n", returncode=0),  # docker run
    ]
    mgr.run_container("/models", 8000)
    cmds = [str(c) for c in remote.run.call_args_list]
    assert not any("docker rm -f" in c for c in cmds)
    assert any("docker run" in c for c in cmds)


def test_device_flags_ascend_auto():
    mgr, _ = make_mgr(hw_type="ascend", npu_id=1, chip_id=0, device_index=-1)
    flags = mgr._device_flags()
    # device_index = 1*2+0 = 2
    assert "--device /dev/davinci2" in flags
    assert "--device /dev/davinci_manager" in flags
    assert "--device /dev/hisi_hdc" in flags


def test_device_flags_ascend_explicit():
    mgr, _ = make_mgr(hw_type="ascend", npu_id=1, chip_id=0, device_index=0)
    flags = mgr._device_flags()
    assert "--device /dev/davinci0" in flags


def test_device_flags_cuda():
    mgr, _ = make_mgr(hw_type="cuda", npu_id=2)
    flags = mgr._device_flags()
    assert '--gpus "device=2"' in flags


def test_exec_run_returns_exec_result():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="hello\n", stderr="", returncode=0)
    result = mgr.exec_run("echo hello")
    assert isinstance(result, ExecResult)
    assert result.stdout == "hello\n"
    assert result.returncode == 0
    cmd_str = str(remote.run.call_args)
    assert "docker exec" in cmd_str
    assert "vllm_tuner" in cmd_str


def test_exec_background_calls_docker_exec():
    mgr, remote = make_mgr()
    mgr.exec_background("python -m vllm.entrypoints.openai.api_server", "/workspace/vllm.log")
    cmd_str = str(remote.run.call_args)
    assert "docker exec" in cmd_str
    assert "nohup" in cmd_str
    assert "/workspace/vllm.log" in cmd_str


def test_stop_container_suppresses_errors():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="", returncode=1)  # container not found
    mgr.stop_container()  # must not raise
    cmds = [str(c) for c in remote.run.call_args_list]
    assert any("docker stop" in c for c in cmds)
    assert any("docker rm" in c for c in cmds)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_docker_manager.py -v
```

Expected: `ImportError: No module named 'vllm_tuner.docker_manager'`

- [ ] **Step 3: Create `vllm_tuner/docker_manager.py`**

```python
# vllm_tuner/docker_manager.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ExecResult:
    """Return type of exec_run — mirrors fabric Result contract."""
    stdout: str
    stderr: str
    returncode: int


class DockerManager:
    """Encapsulates all docker CLI operations issued over SSH via RemoteEnv."""

    def __init__(self, remote, docker_cfg, hw_cfg):
        self.remote = remote
        self.cfg = docker_cfg
        self.hw = hw_cfg

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
        return bool(result.stdout.strip())

    def is_container_running(self) -> bool:
        """Return True if container is currently running."""
        result = self.remote.run(
            f"docker ps --filter name={self.cfg.container_name} --filter status=running -q",
            timeout=15,
        )
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
            self.remote.run(f"docker rm -f {self.cfg.container_name}", timeout=30)

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
        # Use double-quoted outer shell to safely handle single quotes in cmd
        # (vLLM pkill patterns contain single quotes)
        safe_cmd = cmd.replace("'", "'\\''")
        exec_cmd = (
            f"docker exec {self.cfg.container_name} "
            f"sh -c 'nohup {safe_cmd} > {log_file} 2>&1 &'"
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
                idx = self.hw.npu_id * 2 + self.hw.chip_id
            return (
                f"--device /dev/davinci{idx}"
                f" --device /dev/davinci_manager"
                f" --device /dev/hisi_hdc"
            )
        else:  # cuda
            return f'--gpus "device={self.hw.npu_id}"'
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_docker_manager.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/docker_manager.py tests/test_docker_manager.py
git commit -m "feat: add DockerManager with pull/run/exec/stop over SSH"
```

---

## Task 3: Extend `Actor` with Docker support

**Files:**
- Modify: `vllm_tuner/actor.py`
- Modify: `tests/test_actor.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_actor.py`. **Important:** place the new import at the **top of the file** alongside the existing imports (not inside a function or mid-file), then append the three helper/test functions at the bottom:

```python
# Add this import at the TOP of tests/test_actor.py, with existing imports:
from vllm_tuner.docker_manager import ExecResult

# Add these functions at the BOTTOM of tests/test_actor.py:
def make_docker_actor():
    from vllm_tuner.docker_manager import DockerManager
    remote = MagicMock()
    framework = MagicMock()
    framework.build_start_cmd.return_value = "python -m vllm.entrypoints.openai.api_server"
    framework.get_health_endpoint.return_value = "http://1.2.3.4:8000/health"
    docker_mgr = MagicMock(spec=DockerManager)
    docker_mgr.exec_run.return_value = ExecResult(stdout="", stderr="", returncode=0)
    return Actor(remote=remote, framework=framework, work_dir="/workspace",
                 host="1.2.3.4", port=8000, docker_manager=docker_mgr), docker_mgr


def test_stop_service_docker_uses_pkill():
    actor, docker_mgr = make_docker_actor()
    # First exec_run: pkill (no output = process gone)
    # Second exec_run: pgrep (empty = process dead)
    docker_mgr.exec_run.side_effect = [
        ExecResult(stdout="", stderr="", returncode=0),   # pkill -f
        ExecResult(stdout="", stderr="", returncode=1),   # pgrep (not found = dead)
    ]
    actor.stop_service()
    cmds = [str(c) for c in docker_mgr.exec_run.call_args_list]
    assert any("pkill" in c for c in cmds)
    # remote.run must NOT be called (no lsof on host)
    actor.remote.run.assert_not_called()


def test_start_service_docker_uses_exec_background():
    actor, docker_mgr = make_docker_actor()
    with patch("vllm_tuner.actor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        actor.start_service("/models/Qwen", {})
    docker_mgr.exec_background.assert_called_once()
    actor.remote.run_background.assert_not_called()


def test_backward_compat_no_docker_manager():
    # Without docker_manager, must still work as before
    actor = make_actor()  # existing helper — no docker_manager
    actor.remote.run.side_effect = [
        MagicMock(stdout="12345\n", returncode=0),
        MagicMock(stdout="vllm\n", returncode=0),
        MagicMock(stdout="", returncode=0),
        MagicMock(stdout="", returncode=0),
    ]
    actor.stop_service()
    calls = [str(c) for c in actor.remote.run.call_args_list]
    assert any("lsof" in c for c in calls)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_actor.py::test_stop_service_docker_uses_pkill tests/test_actor.py::test_start_service_docker_uses_exec_background tests/test_actor.py::test_backward_compat_no_docker_manager -v
```

Expected: FAILED — `Actor.__init__` does not accept `docker_manager`

- [ ] **Step 3: Update `vllm_tuner/actor.py`**

Change `__init__` signature:

```python
def __init__(self, remote, framework, work_dir: str, host: str, port: int,
             docker_manager=None):
    self.remote = remote
    self.framework = framework
    self.work_dir = work_dir
    self.host = host
    self.port = port
    self.docker_manager = docker_manager
```

Replace `stop_service`:

```python
def stop_service(self):
    """Gracefully stop the inference service and free the port."""
    if self.docker_manager:
        # Use pkill inside container — requires only procps (present in Ubuntu images)
        self.docker_manager.exec_run(
            "pkill -f 'vllm.entrypoints' || true", timeout=10
        )
        time.sleep(3)
        check = self.docker_manager.exec_run(
            "pgrep -f 'vllm.entrypoints'", timeout=10
        )
        if check.stdout.strip():
            self.docker_manager.exec_run(
                "pkill -9 -f 'vllm.entrypoints' || true", timeout=10
            )
        return

    # Bare-process mode (unchanged)
    result = self.remote.run(f"lsof -ti:{self.port} -sTCP:LISTEN", timeout=10)
    pid = result.stdout.strip()
    if not pid:
        return
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
    check = self.remote.run(f"ps -p {pid} -o pid=", timeout=10)
    if check.stdout.strip():
        self.remote.run(f"kill -9 {pid}", timeout=10)
```

Replace `start_service`:

```python
def start_service(self, model_path: str, infra_config: dict, health_timeout: int = 120):
    """Launch service with given infra_config, wait for health OK."""
    cmd = self.framework.build_start_cmd(model_path, infra_config)
    log_file = f"{self.work_dir}/vllm.log"
    if self.docker_manager:
        self.docker_manager.exec_background(cmd, log_file)
    else:
        self.remote.run_background(cmd, log_file)
    self._wait_for_health(timeout=health_timeout)
```

- [ ] **Step 4: Run all actor tests**

```bash
python -m pytest tests/test_actor.py -v
```

Expected: All PASSED (existing 3 + new 3 = 6 total)

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/actor.py tests/test_actor.py
git commit -m "feat: Actor accepts docker_manager; Docker branch uses pkill/exec_background"
```

---

## Task 4: Extend `ShipIt` with Docker support

**Files:**
- Modify: `vllm_tuner/shipit.py`
- Modify: `tests/test_shipit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shipit.py`:

```python
from vllm_tuner.docker_manager import DockerManager, ExecResult
from pathlib import Path

def make_docker_shipit():
    from vllm_tuner.docker_manager import DockerManager
    remote = MagicMock()
    actor = MagicMock()
    actor.fast_fail_check.return_value = True
    framework = MagicMock()
    cfg = MagicMock()
    cfg.remote.working_dir = "/workspace"
    cfg.remote.vllm_port = 8000
    cfg.remote.host = "1.2.3.4"
    cfg.hardware.npu_id = 0
    cfg.hardware.chip_id = 0
    cfg.model.hf_url = "https://huggingface.co/Qwen/Qwen2-7B"
    cfg.model.local_path = "/workspace/models/Qwen2-7B"
    cfg.model.hf_token = ""
    docker_mgr = MagicMock(spec=DockerManager)
    docker_mgr.exec_run.return_value = ExecResult(stdout="", stderr="", returncode=0)
    ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg,
                  docker_manager=docker_mgr)
    return ship, docker_mgr


def test_check_env_docker_uses_is_image_present():
    ship, docker_mgr = make_docker_shipit()
    docker_mgr.is_image_present.return_value = True
    # Other remote.run calls: npu-smi usages, npu-smi health, df, lsof, ls model
    ship.remote.run.side_effect = [
        MagicMock(stdout="HBM-Usage(MB): 1000 / 65536\n"),
        MagicMock(stdout="OK\n"),
        MagicMock(stdout="/dev/sda  400G  50G  350G\n"),
        MagicMock(stdout=""),           # lsof (port free)
        MagicMock(stdout="exists\n"),   # ls model
    ]
    env = ship.check_remote_env()
    assert env["vllm_installed"] is True
    docker_mgr.is_image_present.assert_called_once()
    # Must NOT call "python -c 'import vllm'" on host
    host_cmds = [str(c) for c in ship.remote.run.call_args_list]
    assert not any("import vllm" in c for c in host_cmds)


def test_auto_fix_env_docker_pulls_when_image_missing():
    ship, docker_mgr = make_docker_shipit()
    env = {"vllm_installed": False, "port_occupied": False}
    ship.auto_fix_env(env)
    docker_mgr.pull.assert_called_once()


def test_auto_fix_env_docker_port_uses_fuser():
    ship, docker_mgr = make_docker_shipit()
    env = {"vllm_installed": True, "port_occupied": True}
    ship.auto_fix_env(env)
    cmds = [str(c) for c in docker_mgr.exec_run.call_args_list]
    assert any("fuser" in c for c in cmds)


def test_run_docker_starts_container_when_not_running():
    ship, docker_mgr = make_docker_shipit()
    docker_mgr.is_container_running.return_value = False
    ship.remote.run.return_value = MagicMock(stdout="exists\n", returncode=0)
    env = {
        "hbm_total_mb": 65536, "health": "OK",
        "vllm_installed": True, "disk_ok": True,
        "port_occupied": False, "port_pid": "", "model_exists": True,
    }
    with patch.object(ship, "check_remote_env", return_value=env), \
         patch.object(ship, "auto_fix_env"):
        ship.run({})
    docker_mgr.run_container.assert_called_once()


def test_run_docker_skips_container_start_when_already_running():
    ship, docker_mgr = make_docker_shipit()
    docker_mgr.is_container_running.return_value = True
    env = {
        "hbm_total_mb": 65536, "health": "OK",
        "vllm_installed": True, "disk_ok": True,
        "port_occupied": False, "port_pid": "", "model_exists": True,
    }
    with patch.object(ship, "check_remote_env", return_value=env), \
         patch.object(ship, "auto_fix_env"):
        ship.run({})
    docker_mgr.run_container.assert_not_called()


def test_backward_compat_shipit_no_docker():
    # Without docker_manager, check_remote_env still calls "python -c 'import vllm'"
    ship = make_shipit()  # existing helper — no docker_manager
    ship.remote.run.side_effect = [
        MagicMock(stdout="HBM-Usage(MB): 1000 / 65536\n"),
        MagicMock(stdout="OK\n"),
        MagicMock(stdout="", returncode=0),   # python -c 'import vllm'
        MagicMock(stdout="/dev/sda  400G\n"),
        MagicMock(stdout=""),
        MagicMock(stdout="exists\n"),
    ]
    env = ship.check_remote_env()
    host_cmds = [str(c) for c in ship.remote.run.call_args_list]
    assert any("import vllm" in c for c in host_cmds)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_shipit.py::test_check_env_docker_uses_is_image_present tests/test_shipit.py::test_auto_fix_env_docker_pulls_when_image_missing tests/test_shipit.py::test_auto_fix_env_docker_port_uses_fuser tests/test_shipit.py::test_run_docker_starts_container_when_not_running tests/test_shipit.py::test_run_docker_skips_container_start_when_already_running tests/test_shipit.py::test_backward_compat_shipit_no_docker -v
```

Expected: FAILED — `ShipIt.__init__` does not accept `docker_manager`

- [ ] **Step 3: Update `vllm_tuner/shipit.py`**

Change `__init__` signature:

```python
def __init__(self, remote, actor, framework, config, docker_manager=None):
    self.remote = remote
    self.actor = actor
    self.framework = framework
    self.cfg = config
    self.docker_manager = docker_manager
```

Replace `check_remote_env` — add Docker branch for `vllm_installed`, keep rest unchanged:

```python
def check_remote_env(self) -> dict:
    npu_id, chip_id = self.cfg.hardware.npu_id, self.cfg.hardware.chip_id
    port = self.cfg.remote.vllm_port
    work_dir = self.cfg.remote.working_dir

    # HBM
    usages_out = self.remote.run(
        f"npu-smi info -t usages -i {npu_id} -c {chip_id}"
    ).stdout
    m = re.search(r"HBM-Usage\(MB\):\s*(\d+)\s*/\s*(\d+)", usages_out)
    hbm_total = float(m.group(2)) if m else 0.0

    # Health
    health_out = self.remote.run(
        f"npu-smi info -t health -i {npu_id}"
    ).stdout.strip()
    health = health_out.split()[0] if health_out else "UNKNOWN"

    # vLLM installed — Docker: check image; bare: import check
    if self.docker_manager:
        vllm_installed = self.docker_manager.is_image_present()
    else:
        vllm_check = self.remote.run("python -c 'import vllm'")
        vllm_installed = vllm_check.returncode == 0

    # Disk
    df_out = self.remote.run(f"df -h {work_dir}").stdout
    disk_ok = bool(df_out.strip())

    # Port occupant
    lsof_out = self.remote.run(f"lsof -ti:{port} -sTCP:LISTEN").stdout.strip()
    port_occupied = bool(lsof_out)

    # Model exists
    model_check = self.remote.run(f"ls {self.cfg.model.local_path}")
    model_exists = bool(model_check.stdout.strip())

    return {
        "hbm_total_mb": hbm_total,
        "health": health,
        "vllm_installed": vllm_installed,
        "disk_ok": disk_ok,
        "port_occupied": port_occupied,
        "port_pid": lsof_out,
        "model_exists": model_exists,
    }
```

Replace `auto_fix_env`:

```python
def auto_fix_env(self, env: dict):
    if self.docker_manager:
        if not env["vllm_installed"]:
            print("[Fix] Pulling Docker image...")
            self.docker_manager.pull()
        if env["port_occupied"]:
            print(f"[!] Port {self.cfg.remote.vllm_port} occupied, clearing inside container...")
            self.docker_manager.exec_run(
                f"fuser -k {self.cfg.remote.vllm_port}/tcp || true", timeout=10
            )
        return

    # Bare-process mode (unchanged)
    if not env["vllm_installed"]:
        print("[Fix] Installing vllm-ascend...")
        self.remote.run("pip install vllm-ascend", timeout=300)
    if env["port_occupied"]:
        print(f"[!] Port {self.cfg.remote.vllm_port} occupied, clearing...")
        self._clear_port(self.cfg.remote.vllm_port)
    if self.cfg.hardware.type == "ascend":
        self.remote.run(
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh || true"
        )
```

Add `from pathlib import Path` to the **top of `vllm_tuner/shipit.py`** alongside the existing `import re` and `import time` lines:

```python
# vllm_tuner/shipit.py
import re
import time
from pathlib import Path   # ADD THIS LINE
```

Update `run` — add container startup block after `auto_fix_env`:

```python
def run(self, default_infra_config: dict) -> bool:
    """Full deployment pipeline. Returns True on success."""
    print("[Ship-It] Checking remote environment...")
    env = self.check_remote_env()
    print(f"[OK] HBM: {env['hbm_total_mb']}MB, Health: {env['health']}")

    self.auto_fix_env(env)

    if self.docker_manager:
        if not self.docker_manager.is_container_running():
            print("[Docker] Starting container...")
            model_dir = str(Path(self.cfg.model.local_path).parent)
            self.docker_manager.run_container(model_dir, self.cfg.remote.vllm_port)

    if not env["model_exists"]:
        print(f"[Down] Pulling model {self.cfg.model.hf_url}...")
        self.pull_model()
        print("[OK] Model pull complete")

    print("[...] Starting inference service...")
    self.actor.start_service(self.cfg.model.local_path, default_infra_config)

    if not self.actor.fast_fail_check():
        logs = self.remote.read_log_tail(
            f"{self.cfg.remote.working_dir}/vllm.log"
        )
        if hasattr(self, "brain") and self.brain is not None:
            self.self_heal(
                error="fast-fail check failed",
                attempt_history=[],
                brain=self.brain,
                default_infra_config=default_infra_config,
            )
        else:
            raise RuntimeError(f"Service fast-fail check failed. Logs:\n{logs}")

    print("[OK] Service healthy and responding")
    return True
```

- [ ] **Step 4: Run all shipit tests**

```bash
python -m pytest tests/test_shipit.py -v
```

Expected: All PASSED (existing 4 + new 6 = 10 total)

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/shipit.py tests/test_shipit.py
git commit -m "feat: ShipIt accepts docker_manager; Docker branch uses image check, pull, fuser, run_container"
```

---

## Task 5: Wire `DockerManager` into `main.py`

**Files:**
- Modify: `vllm_tuner/main.py`

No new tests needed here — the wiring is a composition of already-tested components. Verified by the integration test in Task 6.

- [ ] **Step 1: Update imports in `main.py`**

Add after existing imports:

```python
from vllm_tuner.docker_manager import DockerManager
```

- [ ] **Step 2: Update `cmd_run`**

After `remote = RemoteEnv(cfg.remote)`, add:

```python
docker_mgr = None
if cfg.docker:
    docker_mgr = DockerManager(remote, cfg.docker, cfg.hardware)
```

Change `Actor(...)` construction:

```python
actor = Actor(remote=remote, framework=framework,
              work_dir=cfg.remote.working_dir,
              host=cfg.remote.host, port=cfg.remote.vllm_port,
              docker_manager=docker_mgr)
```

Change `ShipIt(...)` construction (in Phase 0 block):

```python
ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg,
              docker_manager=docker_mgr)
```

Wrap the **entire body** of `cmd_run` from after the `actor`/`brain`/`reporter` construction down to and including Phase 2a/2b/3 in a try/finally. The full structure of `cmd_run` after the edit:

```python
def cmd_run(args):
    cfg = load_config(args.config)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    server_url = args.server_url

    remote = RemoteEnv(cfg.remote)
    docker_mgr = None                                    # NEW
    if cfg.docker:                                       # NEW
        docker_mgr = DockerManager(remote, cfg.docker, cfg.hardware)  # NEW

    framework = get_framework(cfg.framework, host=cfg.remote.host, port=cfg.remote.vllm_port)
    observer = get_observer(cfg.hardware.type, remote,
                            npu_id=cfg.hardware.npu_id, chip_id=cfg.hardware.chip_id)
    actor = Actor(remote=remote, framework=framework,
                  work_dir=cfg.remote.working_dir,
                  host=cfg.remote.host, port=cfg.remote.vllm_port,
                  docker_manager=docker_mgr)             # NEW param
    brain = Brain(api_key=api_key)
    reporter = Reporter(save_dir=cfg.save_dir)

    try:                                                 # NEW
        if server_url is None:
            print("\n--- Phase 0: Ship-It ---")
            ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg,
                          docker_manager=docker_mgr)     # NEW param
            default_infra = {
                "block_size": cfg.optimization.phase_2a.parameters["block_size"][0],
                "gpu_memory_utilization": cfg.optimization.phase_2a.parameters.get(
                    "gpu_memory_utilization", [0.85])[0],
            }
            ship.run(default_infra)

            print("\n--- Phase 1: Sweep ---")
            api_base = framework.get_api_base()
            sweep = BenchmarkSweep(
                server_url=api_base,
                health_url=framework.get_health_endpoint(),
                concurrency_levels=cfg.sweep.concurrency_levels,
                input_lengths=cfg.sweep.input_lengths,
                requests_per_cell=cfg.sweep.requests_per_cell,
                request_timeout=cfg.sweep.request_timeout_seconds,
            )
            sweep_result = sweep.run()
            print(f"[Sweep] Best: {sweep_result.get('best_throughput')}")
            server_url = api_base
        else:
            print(f"\n[--server-url] Skipping Phase 0 (deploy) and Phase 1 (sweep).")
            print(f"[--server-url] Using pre-deployed server: {server_url}")
            sweep_result = {}

        skills = _build_skills(cfg, remote, observer)
        runner = Runner(skills=skills, server_url=server_url)

        orch = Orchestrator(
            config=cfg, actor=actor, brain=brain,
            runner=runner, reporter=reporter,
            sweep_result=sweep_result, observer=observer,
        )
        print("\n--- Phase 2a: Infra Tuning ---")
        best_infra = orch.run_phase_2a()

        print("\n--- Phase 2b: Accuracy Tuning ---")
        orch.run_phase_2b(infra_config=best_infra)

        print("\n--- Phase 3: Report ---")
        model_name = cfg.model.hf_url.split("/")[-1]
        report_path = reporter.generate_best_report(
            model_name=model_name,
            hardware_name=cfg.hardware.type.upper(),
            metric_direction=cfg.evaluation.metric_direction,
        )
        print(f"[OK] Report saved: {report_path}")
        print(f"[OK] Service: {server_url}")
    finally:                                             # NEW
        if docker_mgr:                                   # NEW
            docker_mgr.stop_container()                  # NEW
        remote.close()                                   # moved here from end of function
```

The key point: **all of Phase 0 through Phase 3 must be inside the `try` block** so that `docker_mgr.stop_container()` runs even if any phase raises an exception.

- [ ] **Step 3: Update `cmd_deploy`**

After `remote = RemoteEnv(cfg.remote)`:

```python
docker_mgr = None
if cfg.docker:
    docker_mgr = DockerManager(remote, cfg.docker, cfg.hardware)
```

Change `Actor(...)`:

```python
actor = Actor(remote=remote, framework=framework,
              work_dir=cfg.remote.working_dir,
              host=cfg.remote.host, port=cfg.remote.vllm_port,
              docker_manager=docker_mgr)
```

Change `ShipIt(...)`:

```python
ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg,
              docker_manager=docker_mgr)
```

Wrap in try/finally:

```python
    try:
        ship.run(infra)
    finally:
        if docker_mgr:
            docker_mgr.stop_container()
        remote.close()
```

Remove the standalone `remote.close()`.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASSED (no regressions)

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/main.py
git commit -m "feat: wire DockerManager into cmd_run and cmd_deploy with finally cleanup"
```

---

## Task 6: Smoke test Docker mode end-to-end

**Files:**
- No new files — use existing `demo/run_integration.py` pattern for reference

This task verifies the Docker code path with mocked Docker calls, using Python directly (no real remote machine needed).

- [ ] **Step 1: Write a quick smoke test**

```python
# Run this as a one-off verify (not added to test suite — no real Docker available in CI)
from unittest.mock import MagicMock, patch
from vllm_tuner.config import DockerConfig, HardwareConfig
from vllm_tuner.docker_manager import DockerManager, ExecResult

remote = MagicMock()
remote.run.return_value = MagicMock(stdout="", stderr="", returncode=0)
docker_cfg = DockerConfig(image="vllm-ascend:latest", container_name="smoke_test")
hw_cfg = HardwareConfig(type="ascend", npu_id=0, chip_id=0)
mgr = DockerManager(remote=remote, docker_cfg=docker_cfg, hw_cfg=hw_cfg)

# Simulate: image not present → pull → run_container → exec_background → stop
remote.run.side_effect = [
    MagicMock(stdout="", returncode=0),          # is_image_present: False
    MagicMock(stdout="", returncode=0),          # pull
    MagicMock(stdout="", returncode=0),          # run_container: ps -a (no stale)
    MagicMock(stdout="abc\n", returncode=0),     # run_container: docker run OK
    MagicMock(stdout="", returncode=0),          # exec_background
    MagicMock(stdout="", returncode=0),          # stop: docker stop
    MagicMock(stdout="", returncode=0),          # stop: docker rm
]
assert mgr.is_image_present() is False
mgr.pull()
mgr.run_container("/models", 8000)
mgr.exec_background("python -m vllm.entrypoints.openai.api_server", "/workspace/vllm.log")
mgr.stop_container()
print("Smoke test PASSED")
```

- [ ] **Step 2: Run the smoke test**

```bash
python -c "
from unittest.mock import MagicMock
from vllm_tuner.config import DockerConfig, HardwareConfig
from vllm_tuner.docker_manager import DockerManager, ExecResult

remote = MagicMock()
docker_cfg = DockerConfig(image='vllm-ascend:latest', container_name='smoke_test')
hw_cfg = HardwareConfig(type='ascend', npu_id=0, chip_id=0)
mgr = DockerManager(remote=remote, docker_cfg=docker_cfg, hw_cfg=hw_cfg)

remote.run.side_effect = [
    MagicMock(stdout='', returncode=0),
    MagicMock(stdout='', returncode=0),
    MagicMock(stdout='', returncode=0),
    MagicMock(stdout='abc', returncode=0),
    MagicMock(stdout='', returncode=0),
    MagicMock(stdout='', returncode=0),
    MagicMock(stdout='', returncode=0),
]
assert mgr.is_image_present() is False
mgr.pull()
mgr.run_container('/models', 8000)
mgr.exec_background('python -m vllm.entrypoints.openai.api_server', '/workspace/vllm.log')
mgr.stop_container()
print('Smoke test PASSED')
"
```

Expected output: `Smoke test PASSED`

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All tests PASSED

- [ ] **Step 4: Final commit**

```bash
git add vllm_tuner/config.py vllm_tuner/docker_manager.py vllm_tuner/actor.py vllm_tuner/shipit.py vllm_tuner/main.py tests/test_config_docker.py tests/test_docker_manager.py tests/test_actor.py tests/test_shipit.py
git commit -m "test: verify Docker mode smoke test passes end-to-end"
```

---

## Summary

| Task | Files | Tests added |
|------|-------|-------------|
| 1. DockerConfig | `config.py` | 2 (test_config_docker.py) |
| 2. DockerManager | `docker_manager.py` | 15 (test_docker_manager.py) |
| 3. Actor Docker | `actor.py` | 3 (test_actor.py) |
| 4. ShipIt Docker | `shipit.py` | 6 (test_shipit.py) |
| 5. main.py wiring | `main.py` | 0 (covered by full suite) |
| 6. Smoke test | — | 0 (manual verify) |

Total new tests: **26** on top of the existing 51.

After all tasks: `python -m pytest tests/ -v` should show **77 PASSED**.
