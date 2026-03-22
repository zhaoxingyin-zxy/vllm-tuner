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
    ├── auto_fix_env()      → docker_manager.pull()
    ├── [NEW] docker_manager.run_container(model_dir, port)  # sleep infinity
    └── actor.start_service()
            └── docker_manager.exec_background(vllm_cmd, log_file)

Per Phase 2a round:
    actor.stop_service()    → docker_manager.exec_run("lsof -ti:port | xargs kill")
    actor.start_service()   → docker_manager.exec_background(vllm_cmd, log_file)

Session end (main.py finally):
    docker_manager.stop_container()   # docker stop + docker rm
```

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| `vllm_tuner/config.py` | Modified | Add `DockerConfig` dataclass; `TunerConfig.docker: DockerConfig \| None = None` |
| `vllm_tuner/docker_manager.py` | **New** | All `docker` CLI operations |
| `vllm_tuner/actor.py` | Modified | `docker_manager=None` param; 2 branch points in `start_service` / `stop_service` |
| `vllm_tuner/shipit.py` | Modified | `docker_manager=None` param; 3 branch points in `check_remote_env`, `auto_fix_env`, `run` |
| `vllm_tuner/main.py` | Modified | Assemble `DockerManager`; `finally` block cleans up container |

`frameworks/vllm_framework.py` is **not changed** — `build_start_cmd` continues returning a bare vLLM command; Docker exec wrapping is done in `Actor`.

---

## DockerConfig Schema

```python
@dataclass
class DockerConfig:
    image: str           # e.g. "vllm-ascend:v0.9.0"
    container_name: str  # e.g. "vllm_tuner_serving"
    shm_size: str = "8g"
    registry: str = ""   # private registry prefix; empty = no login
    extra_flags: str = "" # appended verbatim to docker run
```

### YAML example

```yaml
docker:
  image: "vllm-ascend:v0.9.0"
  container_name: "vllm_tuner_serving"
  shm_size: "8g"
  registry: ""
  extra_flags: ""
```

---

## DockerManager Interface

```python
class DockerManager:
    def pull(self): ...
    def is_image_present(self) -> bool: ...
    def is_container_running(self) -> bool: ...
    def run_container(self, model_host_dir: str, vllm_port: int): ...
    def exec_background(self, cmd: str, log_file: str): ...
    def exec_run(self, cmd: str, timeout: int = 60): ...
    def stop_container(self): ...
    def _device_flags(self) -> str: ...
```

### Device flags by hardware type

| `hardware.type` | Flags added to `docker run` |
|-----------------|----------------------------|
| `ascend` | `--device /dev/davinci{npu_id} --device /dev/davinci_manager --device /dev/hisi_hdc` |
| `cuda` | `--gpus '"device={npu_id}"'` |

### Container startup command

```
docker run -d --rm \
  --name <container_name> \
  <device_flags> \
  --shm-size <shm_size> \
  -v <model_parent_dir>:<model_parent_dir>:ro \
  -p <vllm_port>:<vllm_port> \
  <extra_flags> \
  <image> \
  sleep infinity
```

---

## Actor Changes

```python
class Actor:
    def __init__(self, remote, framework, work_dir, host, port,
                 docker_manager=None):  # NEW param
        ...
        self.docker_manager = docker_manager

    def stop_service(self):
        run = self.docker_manager.exec_run if self.docker_manager else self.remote.run
        # existing lsof + kill logic unchanged, just uses `run` variable

    def start_service(self, model_path, infra_config, health_timeout=120):
        cmd = self.framework.build_start_cmd(model_path, infra_config)
        log_file = f"{self.work_dir}/vllm.log"
        if self.docker_manager:
            self.docker_manager.exec_background(cmd, log_file)
        else:
            self.remote.run_background(cmd, log_file)
        self._wait_for_health(timeout=health_timeout)
```

---

## ShipIt Changes

```python
class ShipIt:
    def __init__(self, remote, actor, framework, config,
                 docker_manager=None):  # NEW param
        ...
        self.docker_manager = docker_manager

    def check_remote_env(self) -> dict:
        if self.docker_manager:
            # replace "python -c 'import vllm'" with image presence check
            vllm_installed = self.docker_manager.is_image_present()
        else:
            # existing logic unchanged
            ...

    def auto_fix_env(self, env):
        if self.docker_manager:
            if not env["vllm_installed"]:   # image not present
                self.docker_manager.pull()
            if env["port_occupied"]:
                self._clear_port(self.cfg.remote.vllm_port)  # exec_run inside
        else:
            # existing logic unchanged
            ...

    def run(self, default_infra_config):
        ...
        if self.docker_manager:
            if not self.docker_manager.is_container_running():
                model_dir = str(Path(self.cfg.model.local_path).parent)
                self.docker_manager.run_container(model_dir, self.cfg.remote.vllm_port)
        self.actor.start_service(...)
        ...
```

---

## main.py Changes

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
```

---

## Backward Compatibility

No `docker:` key in YAML → `cfg.docker is None` → `docker_mgr = None` → all branches fall through to existing code. Zero behavior change for current users.

---

## Out of Scope

- Podman / containerd support (future)
- Multi-container setups
- Docker Compose
- Building images (image must already exist or be pullable)
