# Docker Container Support Design

**Date:** 2026-03-22
**Feature:** Run vLLM inside a Docker container on the remote machine
**Status:** Approved

---

## Summary

Add optional Docker container support to vLLM-Tuner. When a `docker:` block is present in `tuner_config.yaml`, ShipIt pulls the image, starts a long-running container (`sleep infinity`), and all vLLM process management happens via `docker exec` inside that container. When the `docker:` block is absent, the system behaves exactly as before (bare nohup process on the host). Supports both Ascend NPU and CUDA GPU via automatic device flag selection.

---

## Decisions

| Question | Decision |
|----------|----------|
| Container lifecycle | Container stays running; only the vLLM process inside is killed/restarted per round |
| Device support | Both Ascend NPU and CUDA GPU, auto-selected from `hardware.type` |
| Model mounting | Host directory mounted read-only into container (`-v host_path:container_path:ro`) |
| Backward compatibility | Fully preserved; `docker:` block is optional |

---

## Architecture

```
tuner_config.yaml
    └── docker: (optional)
            ↓
        main.py
            ├── DockerManager(remote, docker_cfg, hw_cfg)   [new]
            ├── Actor(..., docker_manager=docker_mgr)        [extended]
            └── ShipIt(..., docker_manager=docker_mgr)      [extended]

ShipIt.run()
    ├── check_remote_env()  → docker_manager.is_image_present()
    ├── auto_fix_env()      → docker_manager.pull()  [includes docker login if registry set]
    ├── [NEW] docker_manager.run_container(model_dir, port)  # sleep infinity, no --rm
    └── actor.start_service()
            └── docker_manager.exec_background(vllm_cmd, log_file)

Per Phase 2a round:
    actor.stop_service()    → docker_manager.exec_run("pkill -f 'vllm.entrypoints'")
    actor.start_service()   → docker_manager.exec_background(vllm_cmd, log_file)

Session end (main.py finally — both cmd_run and cmd_deploy):
    docker_manager.stop_container()   # docker stop + docker rm
```

**Note:** `DockerManager` issues all `docker` CLI commands via `remote.run(...)` over SSH — Docker runs on the remote machine, not locally.

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| `vllm_tuner/config.py` | Modified | Add `DockerConfig` dataclass; `TunerConfig.docker: DockerConfig \| None = None`; update `load_config` to parse optional `docker:` YAML block |
| `vllm_tuner/docker_manager.py` | **New** | All `docker` CLI operations |
| `vllm_tuner/actor.py` | Modified | `docker_manager=None` param; 2 branch points in `start_service` / `stop_service` |
| `vllm_tuner/shipit.py` | Modified | `docker_manager=None` param; 3 branch points in `check_remote_env`, `auto_fix_env`, `run` |
| `vllm_tuner/main.py` | Modified | Assemble `DockerManager` in both `cmd_run` and `cmd_deploy`; `finally` block cleans up container |

`frameworks/vllm_framework.py` is **not changed** — `build_start_cmd` continues returning a bare vLLM command; Docker exec wrapping is done in `Actor`.

---

## DockerConfig Schema

```python
@dataclass
class DockerConfig:
    image: str            # e.g. "vllm-ascend:v0.9.0"
    container_name: str   # e.g. "vllm_tuner_serving"
    shm_size: str = "8g"
    registry: str = ""    # private registry prefix; empty = no login
    extra_flags: str = "" # appended verbatim to docker run
    device_index: int = -1  # Ascend /dev/davinciN index; -1 = auto (npu_id*2+chip_id)
```

### YAML example

```yaml
docker:
  image: "vllm-ascend:v0.9.0"
  container_name: "vllm_tuner_serving"
  shm_size: "8g"
  registry: ""
  extra_flags: ""
  device_index: -1   # -1 = auto-compute; set explicitly for single-die cards (e.g. 0)
```

### config.py: load_config update

`load_config` parses the optional `docker:` block:

```python
docker_cfg = None
if "docker" in raw:
    docker_cfg = DockerConfig(**raw["docker"])

return TunerConfig(
    ...
    docker=docker_cfg,
)
```

---

## DockerManager Interface

```python
@dataclass
class ExecResult:
    """Return type of exec_run — mirrors fabric Result contract."""
    stdout: str
    stderr: str
    returncode: int

class DockerManager:
    def __init__(self, remote: RemoteEnv, docker_cfg: DockerConfig, hw_cfg: HardwareConfig): ...

    def pull(self):
        """
        If cfg.registry is non-empty, first runs `docker login <registry>`.
        Then runs `docker pull <image>`. Raises RuntimeError on non-zero exit.
        """

    def is_image_present(self) -> bool:
        """
        docker images -q <image>  — returns True if output is non-empty.
        """

    def is_container_running(self) -> bool:
        """
        docker ps --filter name=<container_name> --filter status=running -q
        Returns True if output is non-empty.
        """

    def run_container(self, model_host_dir: str, vllm_port: int):
        """
        If a container with container_name exists (running or stopped),
        remove it first: docker rm -f <container_name>.

        Then:
        docker run -d \
          --name <container_name> \
          <device_flags> \
          --shm-size <shm_size> \
          -v <model_host_dir>:<model_host_dir>:ro \
          -p <vllm_port>:<vllm_port> \
          <extra_flags> \
          <image> \
          sleep infinity

        Note: --rm is NOT used; container is removed explicitly in stop_container().
        Note: model_host_dir is mounted at the same absolute path inside the container
              so that vLLM command paths remain identical between host and container.
        Note: the image must include /bin/sh and sleep (not distroless).
        Raises RuntimeError on non-zero exit.
        """

    def exec_background(self, cmd: str, log_file: str):
        """
        docker exec <container_name> sh -c 'nohup <cmd> > <log_file> 2>&1 &'
        Raises RuntimeError if docker exec returns non-zero (e.g. container not running).
        """

    def exec_run(self, cmd: str, timeout: int = 60) -> ExecResult:
        """
        docker exec <container_name> sh -c '<cmd>'
        Returns ExecResult with stdout, stderr, returncode.
        This is the drop-in replacement for remote.run() inside Actor.stop_service().
        """

    def stop_container(self):
        """
        docker stop <container_name> || true
        docker rm   <container_name> || true
        Errors are suppressed (container may already be gone).
        """

    def _device_flags(self) -> str:
        """
        Returns device flag string based on hw_cfg.type.
        See Device Flags section below.
        """
```

---

## Device Flags

### Ascend NPU

Device node index = `npu_id * chips_per_card + chip_id`. For Atlas 300I (1 die/card) this equals `npu_id`. For Atlas 300I Duo (2 dies/card) chip_id=1 gives node index 1. The formula used:

```python
device_index = hw_cfg.npu_id * 2 + hw_cfg.chip_id  # assumes dual-die; override via extra_flags for single-die
```

`DockerConfig.device_index` defaults to `-1` (auto). If `-1`, compute from `npu_id * 2 + chip_id`; otherwise use the explicit value. This lets users override for single-die cards (set `device_index: 0`).

Flags:
```
--device /dev/davinci{device_index}
--device /dev/davinci_manager
--device /dev/hisi_hdc
```

### CUDA GPU

```
--gpus '"device={npu_id}"'
```

---

## Actor Changes

```python
class Actor:
    def __init__(self, remote, framework, work_dir, host, port,
                 docker_manager=None):  # NEW optional param
        ...
        self.docker_manager = docker_manager

    def stop_service(self):
        if self.docker_manager:
            # Use pkill inside the container — avoids requiring lsof/ps in the image
            self.docker_manager.exec_run(
                f"pkill -f 'vllm.entrypoints' || true", timeout=10
            )
            time.sleep(3)
            # Force kill if still running
            check = self.docker_manager.exec_run(
                f"pgrep -f 'vllm.entrypoints'", timeout=10
            )
            if check.stdout.strip():
                self.docker_manager.exec_run(
                    "pkill -9 -f 'vllm.entrypoints' || true", timeout=10
                )
        else:
            # existing lsof + kill logic unchanged
            ...

    def start_service(self, model_path, infra_config, health_timeout=120):
        cmd = self.framework.build_start_cmd(model_path, infra_config)
        log_file = f"{self.work_dir}/vllm.log"
        if self.docker_manager:
            self.docker_manager.exec_background(cmd, log_file)
        else:
            self.remote.run_background(cmd, log_file)
        self._wait_for_health(timeout=health_timeout)
```

**Note:** The Docker `stop_service` uses `pkill -f` instead of `lsof + kill`. This requires only `procps` (`pgrep`/`pkill`) in the container image, which all common vLLM images (Ubuntu-based) include.

---

## ShipIt Changes

```python
class ShipIt:
    def __init__(self, remote, actor, framework, config,
                 docker_manager=None):  # NEW optional param
        ...
        self.docker_manager = docker_manager

    def check_remote_env(self) -> dict:
        if self.docker_manager:
            vllm_installed = self.docker_manager.is_image_present()
            # other checks (disk, port, model) still run via self.remote.run
            ...
        else:
            # existing logic unchanged
            ...

    def auto_fix_env(self, env: dict):
        if self.docker_manager:
            if not env["vllm_installed"]:   # image not present
                self.docker_manager.pull()  # includes docker login if registry set
            if env["port_occupied"]:
                # Port is held by a process inside the (possibly existing) container.
                # Use fuser (from psmisc, present in Ubuntu images) — no lsof needed.
                self.docker_manager.exec_run(
                    f"fuser -k {self.cfg.remote.vllm_port}/tcp || true", timeout=10
                )
        else:
            # existing logic unchanged
            ...

    def run(self, default_infra_config: dict) -> bool:
        env = self.check_remote_env()
        self.auto_fix_env(env)

        if self.docker_manager:
            # Start the container (removes stale container first if needed)
            if not self.docker_manager.is_container_running():
                model_dir = str(Path(self.cfg.model.local_path).parent)
                self.docker_manager.run_container(model_dir, self.cfg.remote.vllm_port)

        if not env["model_exists"]:
            self.pull_model()

        self.actor.start_service(self.cfg.model.local_path, default_infra_config)
        ...
```

---

## main.py Changes

Applied to **both** `cmd_run` and `cmd_deploy`:

```python
def cmd_run(args):
    cfg = load_config(args.config)
    remote = RemoteEnv(cfg.remote)

    docker_mgr = None
    if cfg.docker:
        docker_mgr = DockerManager(remote, cfg.docker, cfg.hardware)

    actor = Actor(remote, framework, work_dir, host, port,
                  docker_manager=docker_mgr)
    shipit = ShipIt(remote, actor, framework, cfg,
                    docker_manager=docker_mgr)
    try:
        shipit.run(default_infra_config)
        # ... rest of pipeline unchanged
    finally:
        if docker_mgr:
            docker_mgr.stop_container()
        remote.close()

# cmd_deploy follows the same pattern
```

---

## Backward Compatibility

No `docker:` key in YAML → `cfg.docker is None` → `docker_mgr = None` → all new branches are unreachable → existing behavior is byte-for-byte identical. Verified per code path:

- `Actor.stop_service`: `if self.docker_manager:` is False → existing `lsof + kill` runs
- `Actor.start_service`: `if self.docker_manager:` is False → existing `remote.run_background` runs
- `ShipIt.check_remote_env`: `if self.docker_manager:` is False → existing `python -c 'import vllm'` runs
- `ShipIt.auto_fix_env`: `if self.docker_manager:` is False → existing `pip install` runs
- `ShipIt.run`: `if self.docker_manager:` is False → no container step added

---

## Image Requirements

The Docker image must:
- Include `/bin/sh` and `sleep` (not distroless)
- Include `pkill` / `pgrep` (`procps` package — present in all Ubuntu-based images)
- Include `fuser` (`psmisc` package — present in all Ubuntu-based images; used for port cleanup)
- Have `python -m vllm.entrypoints.openai.api_server` runnable without additional setup

---

## Out of Scope

- Podman / containerd support (future)
- Multi-container setups
- Docker Compose
- Building images (image must already exist or be pullable)
- `docker login` credential management beyond passing `registry` prefix
