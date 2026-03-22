# vLLM-Tuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that automatically deploys an LLM on a remote Ascend NPU via SSH, sweeps performance boundaries, then iteratively tunes both vLLM startup parameters (throughput) and generation parameters (accuracy) using Claude as the decision engine.

**Architecture:** Five sequential phases — Ship-It (deploy), Sweep (profile), Phase 2a (infra tuning with restart), Phase 2b (generation tuning no restart), Phase 3 (report). All remote operations go through a single `RemoteEnv` SSH connection using `fabric`. Claude (`brain.py`) reads hardware snapshots and round history to make causal decisions rather than statistical search.

**Tech Stack:** Python 3.10+, fabric>=3.0, anthropic SDK, requests, pyyaml, pandas, matplotlib. Remote: vllm-ascend, npu-smi (pre-installed with CANN).

---

## File Map

```
vllm_tuner/
├── main.py                      # CLI entry: run / deploy subcommands
├── config.py                    # Load + validate tuner_config.yaml into dataclasses
├── remote_env.py                # fabric.Connection wrapper
├── shipit.py                    # Phase 0: check → fix → pull → deploy → verify
├── actor.py                     # Start/stop/restart vLLM service, fast-fail check
├── sweep.py                     # Phase 1: concurrency × input_len matrix
├── orchestrator.py              # Phase 2a + 2b loop control
├── brain.py                     # Claude API: decide next config, diagnose failures
├── runner.py                    # Sequential EvalSkill execution per round
├── reporter.py                  # Write results.tsv row, generate best_config.md
├── analysis.py                  # Pareto frontier + matplotlib chart
├── frameworks/
│   ├── base.py                  # InferenceFramework ABC
│   ├── vllm_framework.py        # vllm-ascend launch command
│   ├── lmdeploy_framework.py    # lmdeploy launch command
│   └── sglang_framework.py      # sglang launch command
├── hardware/
│   ├── base.py                  # HardwareObserver ABC
│   ├── ascend.py                # npu-smi snapshot commands
│   └── cuda.py                  # nvidia-smi snapshot commands
├── skills/
│   ├── base.py                  # EvalSkill ABC
│   ├── throughput.py            # ThreadPoolExecutor HTTP benchmark
│   ├── latency.py               # Sequential P50/P99 measurement
│   ├── task_metric.py           # SSH exec eval_script, parse JSON stdout
│   └── memory.py                # npu-smi HBM snapshot
├── tuner_config.yaml            # User config (in .gitignore)
└── .gitignore
```

---

## Task 1: Project Scaffold + Config Loading

**Files:**
- Create: `vllm_tuner/config.py`
- Create: `vllm_tuner/tuner_config.yaml`
- Create: `vllm_tuner/.gitignore`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from vllm_tuner.config import TunerConfig, load_config

def test_load_config_remote(tmp_path):
    yaml_content = """
remote:
  host: 192.168.1.100
  port: 22
  user: ubuntu
  key_file: ~/.ssh/id_rsa
  working_dir: /home/ubuntu/workspace
  vllm_port: 8000
hardware:
  type: ascend
  npu_id: 4
  chip_id: 0
framework: vllm
model:
  hf_url: https://huggingface.co/Qwen/Qwen2-7B
  hf_token: ""
evaluation:
  script: /evals/run_mmlu.py
  data: /datasets/mmlu_sample/
  metric: accuracy
  metric_direction: maximize
  sample_size: 50
  timeout_seconds: 600
  skills: [throughput, latency, task_metric, memory]
sweep:
  concurrency_levels: [1, 2, 4, 8]
  input_lengths: [128, 256, 512]
  requests_per_cell: 10
  request_timeout_seconds: 30
optimization:
  phase_2a:
    max_rounds: 15
    patience: 3
    parameters:
      block_size: [8, 16, 32]
      gpu_memory_utilization: [0.75, 0.80, 0.85, 0.90]
      max_num_seqs: [64, 128, 256]
  phase_2b:
    max_rounds: 15
    patience: 3
    baseline:
      temperature: 1.0
      top_p: 1.0
      top_k: -1
      repetition_penalty: 1.0
    parameters:
      temperature: [0.6, 0.7, 0.8, 1.0]
      top_p: [0.85, 0.90, 0.95, 1.0]
      top_k: [-1, 20, 50, 100]
      repetition_penalty: [1.0, 1.05, 1.1]
save_dir: ./results/
"""
    cfg_file = tmp_path / "tuner_config.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.remote.host == "192.168.1.100"
    assert cfg.remote.vllm_port == 8000
    assert cfg.hardware.type == "ascend"
    assert cfg.hardware.npu_id == 4
    assert cfg.framework == "vllm"
    assert cfg.model.hf_url == "https://huggingface.co/Qwen/Qwen2-7B"
    assert cfg.model.local_path == "/home/ubuntu/workspace/models/Qwen2-7B"
    assert cfg.evaluation.metric_direction == "maximize"
    assert cfg.evaluation.sample_size == 50
    assert cfg.optimization.phase_2a.patience == 3
    assert cfg.optimization.phase_2b.baseline["temperature"] == 1.0
    assert 8 in cfg.optimization.phase_2a.parameters["block_size"]

def test_local_path_derivation(tmp_path):
    """local_path is last path segment of hf_url under working_dir/models/"""
    yaml_content = """
remote:
  host: 1.2.3.4
  port: 22
  user: u
  key_file: ~/.ssh/id_rsa
  working_dir: /workspace
  vllm_port: 8000
hardware: {type: ascend, npu_id: 0, chip_id: 0}
framework: vllm
model:
  hf_url: https://huggingface.co/Qwen/Qwen2-7B-Instruct
  hf_token: ""
evaluation:
  script: /e.py
  data: /d/
  metric: accuracy
  metric_direction: maximize
  sample_size: 10
  timeout_seconds: 60
  skills: [throughput]
sweep:
  concurrency_levels: [1]
  input_lengths: [128]
  requests_per_cell: 1
  request_timeout_seconds: 10
optimization:
  phase_2a: {max_rounds: 1, patience: 1, parameters: {block_size: [8]}}
  phase_2b:
    max_rounds: 1
    patience: 1
    baseline: {temperature: 1.0, top_p: 1.0, top_k: -1, repetition_penalty: 1.0}
    parameters: {temperature: [0.7]}
save_dir: ./results/
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.model.local_path == "/workspace/models/Qwen2-7B-Instruct"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:\dev_project\work_proj
python -m pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'vllm_tuner'`

- [ ] **Step 3: Create package structure and implement config.py**

```python
# vllm_tuner/__init__.py  (empty)

# vllm_tuner/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class RemoteConfig:
    host: str
    port: int
    user: str
    key_file: str
    working_dir: str
    vllm_port: int


@dataclass
class HardwareConfig:
    type: str   # "ascend" or "cuda"
    npu_id: int = 0
    chip_id: int = 0


@dataclass
class ModelConfig:
    hf_url: str
    hf_token: str
    local_path: str = ""

    def __post_init__(self):
        if not self.local_path:
            # Derive: last path segment of hf_url under working_dir/models/
            # working_dir is injected after construction via _set_working_dir
            pass

    def _set_working_dir(self, working_dir: str):
        if not self.local_path:
            model_name = self.hf_url.rstrip("/").split("/")[-1]
            self.local_path = f"{working_dir}/models/{model_name}"


@dataclass
class EvaluationConfig:
    script: str
    data: str
    metric: str
    metric_direction: str   # "maximize" or "minimize"
    sample_size: int
    timeout_seconds: int
    skills: list[str]


@dataclass
class SweepConfig:
    concurrency_levels: list[int]
    input_lengths: list[int]
    requests_per_cell: int
    request_timeout_seconds: int


@dataclass
class PhaseConfig:
    max_rounds: int
    patience: int
    parameters: dict[str, list]


@dataclass
class Phase2bConfig:
    max_rounds: int
    patience: int
    baseline: dict[str, Any]
    parameters: dict[str, list]


@dataclass
class OptimizationConfig:
    phase_2a: PhaseConfig
    phase_2b: Phase2bConfig


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


def load_config(path: str) -> TunerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    remote = RemoteConfig(**raw["remote"])
    hardware = HardwareConfig(**raw["hardware"])
    model = ModelConfig(
        hf_url=raw["model"]["hf_url"],
        hf_token=raw["model"].get("hf_token", ""),
        local_path=raw["model"].get("local_path", ""),
    )
    model._set_working_dir(remote.working_dir)

    evaluation = EvaluationConfig(**raw["evaluation"])

    sweep = SweepConfig(**raw["sweep"])

    p2a_raw = raw["optimization"]["phase_2a"]
    phase_2a = PhaseConfig(
        max_rounds=p2a_raw["max_rounds"],
        patience=p2a_raw["patience"],
        parameters=p2a_raw["parameters"],
    )

    p2b_raw = raw["optimization"]["phase_2b"]
    phase_2b = Phase2bConfig(
        max_rounds=p2b_raw["max_rounds"],
        patience=p2b_raw["patience"],
        baseline=p2b_raw["baseline"],
        parameters=p2b_raw["parameters"],
    )

    optimization = OptimizationConfig(phase_2a=phase_2a, phase_2b=phase_2b)

    return TunerConfig(
        remote=remote,
        hardware=hardware,
        framework=raw["framework"],
        model=model,
        evaluation=evaluation,
        sweep=sweep,
        optimization=optimization,
        save_dir=raw["save_dir"],
    )
```

- [ ] **Step 4: Create tuner_config.yaml and .gitignore**

```yaml
# vllm_tuner/tuner_config.yaml
remote:
  host: 192.168.1.100
  port: 22
  user: ubuntu
  key_file: ~/.ssh/id_rsa
  working_dir: /home/ubuntu/workspace
  vllm_port: 8000
hardware:
  type: ascend
  npu_id: 4
  chip_id: 0
framework: vllm
model:
  hf_url: https://huggingface.co/Qwen/Qwen2-7B
  hf_token: ""
evaluation:
  script: /evals/run_mmlu.py
  data: /datasets/mmlu_sample/
  metric: accuracy
  metric_direction: maximize
  sample_size: 50
  timeout_seconds: 600
  skills: [throughput, latency, task_metric, memory]
sweep:
  concurrency_levels: [1, 2, 4, 8, 16, 32]
  input_lengths: [128, 256, 512, 1024, 2048]
  requests_per_cell: 10
  request_timeout_seconds: 30
optimization:
  phase_2a:
    max_rounds: 15
    patience: 3
    parameters:
      block_size: [8, 16, 32]
      gpu_memory_utilization: [0.75, 0.80, 0.85, 0.90]
      max_num_seqs: [64, 128, 256]
  phase_2b:
    max_rounds: 15
    patience: 3
    baseline:
      temperature: 1.0
      top_p: 1.0
      top_k: -1
      repetition_penalty: 1.0
    parameters:
      temperature: [0.6, 0.7, 0.8, 1.0]
      top_p: [0.85, 0.90, 0.95, 1.0]
      top_k: [-1, 20, 50, 100]
      repetition_penalty: [1.0, 1.05, 1.1]
save_dir: ./results/
```

```
# vllm_tuner/.gitignore
results.tsv
results/
tuner_config.yaml
*.log
__pycache__/
.pytest_cache/
```

- [ ] **Step 5: Install dependencies**

```bash
pip install fabric pyyaml anthropic requests pandas matplotlib pytest
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 2 PASSED

- [ ] **Step 7: Commit**

```bash
git add vllm_tuner/ tests/test_config.py
git commit -m "feat: project scaffold and config loading with local_path derivation"
```

---

## Task 2: RemoteEnv SSH Layer

**Files:**
- Create: `vllm_tuner/remote_env.py`
- Create: `tests/test_remote_env.py`

- [ ] **Step 1: Write the failing tests (using Mock)**

```python
# tests/test_remote_env.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.remote_env import RemoteEnv
from vllm_tuner.config import RemoteConfig

def make_config():
    return RemoteConfig(
        host="192.168.1.100", port=22, user="ubuntu",
        key_file="~/.ssh/id_rsa", working_dir="/workspace", vllm_port=8000
    )

@patch("vllm_tuner.remote_env.fabric")
def test_run_returns_stdout(mock_fabric):
    mock_conn = MagicMock()
    mock_conn.run.return_value = MagicMock(stdout="hello\n", stderr="", returncode=0)
    mock_fabric.Connection.return_value = mock_conn

    env = RemoteEnv(make_config())
    result = env.run("echo hello")
    assert result.stdout == "hello\n"

@patch("vllm_tuner.remote_env.fabric")
def test_run_background_uses_nohup(mock_fabric):
    mock_conn = MagicMock()
    mock_fabric.Connection.return_value = mock_conn

    env = RemoteEnv(make_config())
    env.run_background("python server.py", "/workspace/server.log")
    call_args = mock_conn.run.call_args[0][0]
    assert "nohup" in call_args
    assert "server.log" in call_args

@patch("vllm_tuner.remote_env.fabric")
def test_read_log_tail(mock_fabric):
    mock_conn = MagicMock()
    mock_conn.run.return_value = MagicMock(stdout="last line\n", stderr="")
    mock_fabric.Connection.return_value = mock_conn

    env = RemoteEnv(make_config())
    tail = env.read_log_tail("/workspace/vllm.log", lines=10)
    assert tail == "last line\n"
    cmd = mock_conn.run.call_args[0][0]
    assert "tail -10" in cmd
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_remote_env.py -v
```

Expected: `ModuleNotFoundError: No module named 'vllm_tuner.remote_env'`

- [ ] **Step 3: Implement remote_env.py**

```python
# vllm_tuner/remote_env.py
import fabric
from vllm_tuner.config import RemoteConfig


class RemoteEnv:
    """Single persistent SSH connection via fabric. Handles reconnection automatically."""

    def __init__(self, config: RemoteConfig):
        self.config = config
        self.conn = fabric.Connection(
            host=config.host,
            port=config.port,
            user=config.user,
            connect_kwargs={"key_filename": config.key_file},
        )

    def run(self, cmd: str, timeout: int = 60):
        """Execute command, return Result with .stdout .stderr .returncode"""
        return self.conn.run(cmd, warn=True, hide=True, timeout=timeout)

    def run_background(self, cmd: str, log_file: str):
        """Non-blocking background execution, stdout+stderr → log_file."""
        self.conn.run(f"nohup {cmd} > {log_file} 2>&1 &", warn=True, hide=True)

    def read_log_tail(self, log_file: str, lines: int = 50) -> str:
        result = self.run(f"tail -{lines} {log_file}")
        return result.stdout

    def close(self):
        self.conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_remote_env.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/remote_env.py tests/test_remote_env.py
git commit -m "feat: RemoteEnv SSH layer using fabric persistent connection"
```

---

## Task 3: Framework Abstractions

**Files:**
- Create: `vllm_tuner/frameworks/base.py`
- Create: `vllm_tuner/frameworks/vllm_framework.py`
- Create: `vllm_tuner/frameworks/lmdeploy_framework.py`
- Create: `vllm_tuner/frameworks/sglang_framework.py`
- Create: `vllm_tuner/frameworks/__init__.py`
- Create: `tests/test_frameworks.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frameworks.py
import pytest
from vllm_tuner.frameworks.vllm_framework import VLLMFramework
from vllm_tuner.frameworks.lmdeploy_framework import LMDeployFramework
from vllm_tuner.frameworks.sglang_framework import SGLangFramework

def test_vllm_start_cmd_contains_model_and_params():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    config = {"block_size": 16, "gpu_memory_utilization": 0.85, "max_num_seqs": 128}
    cmd = fw.build_start_cmd("/models/Qwen2-7B", config)
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model /models/Qwen2-7B" in cmd
    assert "--block-size 16" in cmd
    assert "--gpu-memory-utilization 0.85" in cmd
    assert "--max-num-seqs 128" in cmd

def test_vllm_health_endpoint():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    assert fw.get_health_endpoint() == "http://1.2.3.4:8000/health"

def test_vllm_api_base():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    assert fw.get_api_base() == "http://1.2.3.4:8000/v1"

def test_lmdeploy_start_cmd():
    fw = LMDeployFramework(host="1.2.3.4", port=8000)
    cmd = fw.build_start_cmd("/models/Qwen2-7B", {"cache_max_entry_count": 0.8})
    assert "lmdeploy serve api_server" in cmd
    assert "/models/Qwen2-7B" in cmd

def test_lmdeploy_health_endpoint():
    fw = LMDeployFramework(host="1.2.3.4", port=8000)
    assert fw.get_health_endpoint() == "http://1.2.3.4:8000/v1/models"

def test_sglang_start_cmd():
    fw = SGLangFramework(host="1.2.3.4", port=8000)
    cmd = fw.build_start_cmd("/models/Qwen2-7B", {"mem_fraction_static": 0.85})
    assert "sglang.launch_server" in cmd
    assert "--model-path /models/Qwen2-7B" in cmd

def test_framework_factory():
    from vllm_tuner.frameworks import get_framework
    fw = get_framework("vllm", host="1.2.3.4", port=8000)
    assert isinstance(fw, VLLMFramework)
    fw2 = get_framework("lmdeploy", host="1.2.3.4", port=8000)
    assert isinstance(fw2, LMDeployFramework)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_frameworks.py -v
```

- [ ] **Step 3: Implement frameworks**

```python
# vllm_tuner/frameworks/base.py
from abc import ABC, abstractmethod

class InferenceFramework(ABC):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @abstractmethod
    def build_start_cmd(self, model_weights: str, config: dict) -> str: ...

    def get_api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @abstractmethod
    def get_health_endpoint(self) -> str: ...

    @property
    def tunable_infra_params(self) -> dict:
        """Phase 2a tunable parameter ranges. Subclasses should override with framework-specific ranges."""
        return {
            "block_size": [8, 16, 32],
            "gpu_memory_utilization": [0.75, 0.80, 0.85, 0.90],
            "max_num_seqs": [64, 128, 256],
        }
```

Also add `test_vllm_tunable_infra_params` to `tests/test_frameworks.py` (add it to the test file before the commit step):

```python
def test_vllm_tunable_infra_params():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    params = fw.tunable_infra_params
    assert "block_size" in params
    assert "gpu_memory_utilization" in params
    assert "max_num_seqs" in params
    assert 8 in params["block_size"]
```

```python
# vllm_tuner/frameworks/vllm_framework.py
from vllm_tuner.frameworks.base import InferenceFramework

class VLLMFramework(InferenceFramework):
    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server"
            f" --model {model_weights}"
            f" --port {self.port}"
        )
        if "block_size" in config:
            cmd += f" --block-size {config['block_size']}"
        if "gpu_memory_utilization" in config:
            cmd += f" --gpu-memory-utilization {config['gpu_memory_utilization']}"
        if "max_num_seqs" in config:
            cmd += f" --max-num-seqs {config['max_num_seqs']}"
        return cmd

    def get_health_endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/health"
```

```python
# vllm_tuner/frameworks/lmdeploy_framework.py
from vllm_tuner.frameworks.base import InferenceFramework

class LMDeployFramework(InferenceFramework):
    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        cmd = f"lmdeploy serve api_server {model_weights} --server-port {self.port}"
        if "cache_max_entry_count" in config:
            cmd += f" --cache-max-entry-count {config['cache_max_entry_count']}"
        return cmd

    def get_health_endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/v1/models"
```

```python
# vllm_tuner/frameworks/sglang_framework.py
from vllm_tuner.frameworks.base import InferenceFramework

class SGLangFramework(InferenceFramework):
    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        cmd = f"python -m sglang.launch_server --model-path {model_weights} --port {self.port}"
        if "mem_fraction_static" in config:
            cmd += f" --mem-fraction-static {config['mem_fraction_static']}"
        return cmd

    def get_health_endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/health"
```

```python
# vllm_tuner/frameworks/__init__.py
from vllm_tuner.frameworks.vllm_framework import VLLMFramework
from vllm_tuner.frameworks.lmdeploy_framework import LMDeployFramework
from vllm_tuner.frameworks.sglang_framework import SGLangFramework

_REGISTRY = {
    "vllm": VLLMFramework,
    "lmdeploy": LMDeployFramework,
    "sglang": SGLangFramework,
}

def get_framework(name: str, host: str, port: int) -> "InferenceFramework":
    if name not in _REGISTRY:
        raise ValueError(f"Unknown framework: {name}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](host=host, port=port)
```

- [ ] **Step 4: Add `test_vllm_tunable_infra_params` to `tests/test_frameworks.py`** (use the test code shown in Step 3 above)

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_frameworks.py -v
```

Expected: 8 PASSED

- [ ] **Step 6: Commit**

```bash
git add vllm_tuner/frameworks/ tests/test_frameworks.py
git commit -m "feat: InferenceFramework abstractions for vLLM, LMDeploy, SGLang with tunable_infra_params"
```

---

## Task 4: Hardware Observer

**Files:**
- Create: `vllm_tuner/hardware/base.py`
- Create: `vllm_tuner/hardware/ascend.py`
- Create: `vllm_tuner/hardware/cuda.py`
- Create: `vllm_tuner/hardware/__init__.py`
- Create: `tests/test_hardware.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hardware.py
from unittest.mock import MagicMock
from vllm_tuner.hardware.ascend import AscendObserver
from vllm_tuner.hardware import get_observer

def make_remote(outputs: dict):
    """Helper: RemoteEnv mock that returns preset stdout per command substring."""
    remote = MagicMock()
    def run_side_effect(cmd, **kwargs):
        for key, stdout in outputs.items():
            if key in cmd:
                return MagicMock(stdout=stdout, returncode=0)
        return MagicMock(stdout="", returncode=0)
    remote.run.side_effect = run_side_effect
    return remote

def test_ascend_parse_hbm():
    remote = make_remote({
        "usages": "HBM-Usage(MB): 3161 / 65536\n",
        "common": "AICore(%): 87\n",
        "power":  "Power(W): 93.6\n",
        "temp":   "Temp(C): 40\n",
        "health": "OK\n",
    })
    obs = AscendObserver(remote, npu_id=4, chip_id=0)
    stats = obs.get_stats()
    assert stats.hbm_used_mb == 3161
    assert stats.hbm_total_mb == 65536
    assert abs(stats.hbm_util_pct - 4.8) < 0.1
    assert stats.aicore_util_pct == 87.0
    assert stats.power_w == 93.6
    assert stats.temp_c == 40.0
    assert stats.health == "OK"

def test_ascend_is_healthy_ok():
    remote = make_remote({"health": "OK\n", "usages": "HBM-Usage(MB): 1000 / 65536\n",
                           "common": "AICore(%): 50\n", "power": "Power(W): 80\n",
                           "temp": "Temp(C): 60\n"})
    obs = AscendObserver(remote, npu_id=4, chip_id=0)
    assert obs.is_healthy() is True

def test_ascend_is_healthy_warning():
    remote = make_remote({"health": "Warning\n", "usages": "HBM-Usage(MB): 1000 / 65536\n",
                           "common": "AICore(%): 50\n", "power": "Power(W): 80\n",
                           "temp": "Temp(C): 60\n"})
    obs = AscendObserver(remote, npu_id=4, chip_id=0)
    assert obs.is_healthy() is False

def test_get_observer_factory():
    remote = MagicMock()
    obs = get_observer("ascend", remote, npu_id=0, chip_id=0)
    assert isinstance(obs, AscendObserver)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_hardware.py -v
```

- [ ] **Step 3: Implement hardware modules**

```python
# vllm_tuner/hardware/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class HardwareStats:
    hbm_used_mb: float
    hbm_total_mb: float
    hbm_util_pct: float
    aicore_util_pct: float
    power_w: float
    temp_c: float
    health: str  # "OK" / "Warning" / "Alarm" / "Critical"

class HardwareObserver(ABC):
    @abstractmethod
    def get_stats(self) -> HardwareStats: ...

    def is_healthy(self) -> bool:
        return self.get_stats().health == "OK"
```

```python
# vllm_tuner/hardware/ascend.py
import re
from vllm_tuner.hardware.base import HardwareObserver, HardwareStats

class AscendObserver(HardwareObserver):
    def __init__(self, remote, npu_id: int, chip_id: int):
        self.remote = remote
        self.npu_id = npu_id
        self.chip_id = chip_id

    def _run(self, flag: str) -> str:
        result = self.remote.run(
            f"npu-smi info -t {flag} -i {self.npu_id} -c {self.chip_id}"
        )
        return result.stdout

    def _run_health(self) -> str:
        result = self.remote.run(
            f"npu-smi info -t health -i {self.npu_id}"
        )
        return result.stdout

    def get_stats(self) -> HardwareStats:
        usages_out = self._run("usages")
        common_out = self._run("common")
        power_out  = self._run("power")
        temp_out   = self._run("temp")
        health_out = self._run_health()

        # Parse HBM: "HBM-Usage(MB): 3161 / 65536"
        m = re.search(r"HBM-Usage\(MB\):\s*(\d+)\s*/\s*(\d+)", usages_out)
        hbm_used = float(m.group(1)) if m else 0.0
        hbm_total = float(m.group(2)) if m else 1.0
        hbm_pct = round(hbm_used / hbm_total * 100, 1)

        # Parse AICore: "AICore(%): 87"
        m = re.search(r"AICore\(%\):\s*(\d+)", common_out)
        aicore = float(m.group(1)) if m else 0.0

        # Parse Power: "Power(W): 93.6"
        m = re.search(r"Power\(W\):\s*([\d.]+)", power_out)
        power = float(m.group(1)) if m else 0.0

        # Parse Temp: "Temp(C): 40"
        m = re.search(r"Temp\(C\):\s*([\d.]+)", temp_out)
        temp = float(m.group(1)) if m else 0.0

        health = health_out.strip().split()[0] if health_out.strip() else "UNKNOWN"

        return HardwareStats(
            hbm_used_mb=hbm_used,
            hbm_total_mb=hbm_total,
            hbm_util_pct=hbm_pct,
            aicore_util_pct=aicore,
            power_w=power,
            temp_c=temp,
            health=health,
        )
```

```python
# vllm_tuner/hardware/cuda.py
import re
from vllm_tuner.hardware.base import HardwareObserver, HardwareStats

class CUDAObserver(HardwareObserver):
    def __init__(self, remote, device_id: int = 0):
        self.remote = remote
        self.device_id = device_id

    def get_stats(self) -> HardwareStats:
        out = self.remote.run(
            f"nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,"
            f"power.draw,temperature.gpu --format=csv,noheader,nounits "
            f"--id={self.device_id}"
        ).stdout
        parts = [p.strip() for p in out.split(",")]
        used, total, util, power, temp = (float(p) for p in parts)
        pct = round(used / total * 100, 1)
        return HardwareStats(
            hbm_used_mb=used, hbm_total_mb=total, hbm_util_pct=pct,
            aicore_util_pct=util, power_w=power, temp_c=temp, health="OK"
        )
```

```python
# vllm_tuner/hardware/__init__.py
from vllm_tuner.hardware.ascend import AscendObserver
from vllm_tuner.hardware.cuda import CUDAObserver

def get_observer(hw_type: str, remote, npu_id: int = 0, chip_id: int = 0):
    if hw_type == "ascend":
        return AscendObserver(remote, npu_id=npu_id, chip_id=chip_id)
    if hw_type == "cuda":
        return CUDAObserver(remote, device_id=npu_id)
    raise ValueError(f"Unknown hardware type: {hw_type}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_hardware.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/hardware/ tests/test_hardware.py
git commit -m "feat: HardwareObserver with Ascend npu-smi and CUDA nvidia-smi implementations"
```

---

## Task 5: EvalSkill Plugins

**Files:**
- Create: `vllm_tuner/skills/base.py`
- Create: `vllm_tuner/skills/throughput.py`
- Create: `vllm_tuner/skills/latency.py`
- Create: `vllm_tuner/skills/task_metric.py`
- Create: `vllm_tuner/skills/memory.py`
- Create: `vllm_tuner/skills/__init__.py`
- Create: `tests/test_skills.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skills.py
import json
from unittest.mock import MagicMock, patch
from vllm_tuner.skills.throughput import ThroughputSkill
from vllm_tuner.skills.latency import LatencySkill
from vllm_tuner.skills.task_metric import TaskMetricSkill
from vllm_tuner.skills.memory import MemorySkill

MOCK_RESPONSE = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"completion_tokens": 10}
}

def make_mock_post(response_json=MOCK_RESPONSE, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_json
    return resp

def test_throughput_returns_tokens_per_second():
    with patch("vllm_tuner.skills.throughput.requests.post") as mock_post:
        mock_post.return_value = make_mock_post()
        skill = ThroughputSkill(concurrency=2, num_requests=4, input_len=128)
        tps = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
        assert tps > 0

def test_latency_returns_p99():
    with patch("vllm_tuner.skills.latency.requests.post") as mock_post:
        mock_post.return_value = make_mock_post()
        skill = LatencySkill(num_requests=5, input_len=128)
        result = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
        assert "p50_ms" in result
        assert "p99_ms" in result
        assert result["p99_ms"] >= result["p50_ms"]

def test_task_metric_parses_json_stdout():
    remote = MagicMock()
    remote.run.return_value = MagicMock(
        stdout='{"metric": 0.678}\n', stderr="", returncode=0
    )
    skill = TaskMetricSkill(
        remote=remote,
        script="/evals/run.py",
        data_dir="/data/",
        sample_size=50,
        timeout_seconds=60,
    )
    val = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
    assert abs(val - 0.678) < 1e-6

def test_task_metric_raises_on_bad_stdout():
    remote = MagicMock()
    remote.run.return_value = MagicMock(stdout="not json", stderr="", returncode=0)
    skill = TaskMetricSkill(remote=remote, script="/e.py", data_dir="/d/",
                             sample_size=10, timeout_seconds=60)
    with pytest.raises(ValueError, match="metric"):
        skill.measure("http://1.2.3.4:8000/v1", gen_params={})

def test_memory_skill_returns_hbm_pct():
    observer = MagicMock()
    from vllm_tuner.hardware.base import HardwareStats
    observer.get_stats.return_value = HardwareStats(
        hbm_used_mb=3161, hbm_total_mb=65536, hbm_util_pct=4.8,
        aicore_util_pct=87, power_w=93.6, temp_c=40, health="OK"
    )
    from vllm_tuner.skills.memory import MemorySkill
    skill = MemorySkill(observer=observer)
    pct = skill.measure("http://1.2.3.4:8000/v1", gen_params={})
    assert abs(pct - 4.8) < 0.01
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_skills.py -v
```

- [ ] **Step 3: Implement skills**

```python
# vllm_tuner/skills/base.py
from abc import ABC, abstractmethod

class EvalSkill(ABC):
    name: str

    @abstractmethod
    def measure(self, server_url: str, gen_params: dict) -> float | dict:
        """Return scalar or dict of metrics."""
```

```python
# vllm_tuner/skills/throughput.py
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm_tuner.skills.base import EvalSkill

class ThroughputSkill(EvalSkill):
    name = "throughput"

    def __init__(self, concurrency: int = 4, num_requests: int = 20, input_len: int = 256):
        self.concurrency = concurrency
        self.num_requests = num_requests
        self.input_len = input_len

    def _single_request(self, url: str, gen_params: dict) -> float:
        prompt = "x " * self.input_len
        payload = {"model": "default", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 64, **gen_params}
        start = time.time()
        resp = requests.post(f"{url}/chat/completions", json=payload, timeout=60)
        elapsed = time.time() - start
        tokens = resp.json().get("usage", {}).get("completion_tokens", 1)
        return tokens / max(elapsed, 0.001)

    def measure(self, server_url: str, gen_params: dict) -> float:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._single_request, server_url, gen_params)
                       for _ in range(self.num_requests)]
            results = [f.result() for f in as_completed(futures)]
        return sum(results) / len(results)
```

```python
# vllm_tuner/skills/latency.py
import time, requests, statistics
from vllm_tuner.skills.base import EvalSkill

class LatencySkill(EvalSkill):
    name = "latency"

    def __init__(self, num_requests: int = 10, input_len: int = 256):
        self.num_requests = num_requests
        self.input_len = input_len

    def measure(self, server_url: str, gen_params: dict) -> dict:
        prompt = "x " * self.input_len
        payload = {"model": "default", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 64, **gen_params}
        latencies = []
        for _ in range(self.num_requests):
            start = time.time()
            requests.post(f"{server_url}/chat/completions", json=payload, timeout=60)
            latencies.append((time.time() - start) * 1000)
        latencies.sort()
        return {
            "p50_ms": statistics.median(latencies),
            "p99_ms": latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[-1],
        }
```

```python
# vllm_tuner/skills/task_metric.py
import json
from vllm_tuner.skills.base import EvalSkill

class TaskMetricSkill(EvalSkill):
    name = "task_metric"

    def __init__(self, remote, script: str, data_dir: str,
                 sample_size: int, timeout_seconds: int):
        self.remote = remote
        self.script = script
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.timeout_seconds = timeout_seconds

    def measure(self, server_url: str, gen_params: dict) -> float:
        cmd = (
            f"python {self.script}"
            f" --server-url {server_url}"
            f" --data-dir {self.data_dir}"
            f" --sample-size {self.sample_size}"
        )
        result = self.remote.run(cmd, timeout=self.timeout_seconds)
        stdout = result.stdout.strip()
        try:
            data = json.loads(stdout)
            return float(data["metric"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"eval_script stdout must be single-line JSON with 'metric' key. "
                f"Got: {stdout!r}"
            ) from e
```

```python
# vllm_tuner/skills/memory.py
from vllm_tuner.skills.base import EvalSkill

class MemorySkill(EvalSkill):
    name = "memory"

    def __init__(self, observer):
        self.observer = observer

    def measure(self, server_url: str, gen_params: dict) -> float:
        return self.observer.get_stats().hbm_util_pct
```

```python
# vllm_tuner/skills/__init__.py
from vllm_tuner.skills.throughput import ThroughputSkill
from vllm_tuner.skills.latency import LatencySkill
from vllm_tuner.skills.task_metric import TaskMetricSkill
from vllm_tuner.skills.memory import MemorySkill
```

- [ ] **Step 4: Add missing import in test file**

Add `import pytest` at the top of `tests/test_skills.py`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_skills.py -v
```

Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add vllm_tuner/skills/ tests/test_skills.py
git commit -m "feat: EvalSkill plugins - throughput, latency, task_metric, memory"
```

---

## Task 6: Reporter (results.tsv + config_hash)

**Files:**
- Create: `vllm_tuner/reporter.py`
- Create: `tests/test_reporter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reporter.py
import os, csv
import pytest
from vllm_tuner.reporter import Reporter, compute_config_hash

def test_config_hash_is_deterministic():
    cfg = {"block_size": 8, "gpu_memory_utilization": 0.85}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash({"gpu_memory_utilization": 0.85, "block_size": 8})
    assert h1 == h2
    assert len(h1) == 6

def test_config_hash_differs_for_different_configs():
    h1 = compute_config_hash({"block_size": 8})
    h2 = compute_config_hash({"block_size": 16})
    assert h1 != h2

def test_reporter_writes_tsv_row(tmp_path):
    tsv = tmp_path / "results.tsv"
    reporter = Reporter(save_dir=str(tmp_path))
    reporter.append_row(
        round_num=1, phase="2a",
        config={"block_size": 8, "gpu_memory_utilization": 0.85},
        throughput=820.1, latency_p99=180.0,
        task_metric=None, memory_pct=0.72,
        status="KEEP", reasoning="baseline"
    )
    assert tsv.exists()
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["status"] == "KEEP"
    assert rows[0]["phase"] == "2a"
    assert rows[0]["task_metric"] == "-"

def test_reporter_skips_duplicate_hash(tmp_path):
    reporter = Reporter(save_dir=str(tmp_path))
    cfg = {"block_size": 8}
    reporter.append_row(1, "2a", cfg, 820, 180, None, 0.7, "KEEP", "r1")
    result = reporter.append_row(2, "2a", cfg, 850, 170, None, 0.7, "KEEP", "r2")
    assert result is False  # skipped duplicate

def test_reporter_load_all_rounds(tmp_path):
    reporter = Reporter(save_dir=str(tmp_path))
    reporter.append_row(1, "2a", {"block_size": 8}, 820, 180, None, 0.72, "KEEP", "b")
    reporter.append_row(2, "2a", {"block_size": 16}, 900, 160, None, 0.80, "KEEP", "opt")
    rows = reporter.load_all()
    assert len(rows) == 2
    assert rows[0]["round"] == "1"

def test_reporter_stores_config_json(tmp_path):
    """config_json column enables --use-best-config round-trip."""
    import json
    reporter = Reporter(save_dir=str(tmp_path))
    cfg = {"block_size": 8, "gpu_memory_utilization": 0.88}
    reporter.append_row(1, "2a", cfg, 1050, 115, None, 0.83, "KEEP", "best")
    rows = reporter.load_all()
    assert "config_json" in rows[0]
    restored = json.loads(rows[0]["config_json"])
    assert restored == cfg
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_reporter.py -v
```

- [ ] **Step 3: Implement reporter.py**

```python
# vllm_tuner/reporter.py
import csv
import hashlib
import json
import os
from pathlib import Path


def compute_config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:6]


FIELDNAMES = [
    "round", "phase", "config_hash", "config_json", "throughput", "latency_p99",
    "task_metric", "memory_pct", "status", "reasoning"
]


class Reporter:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = self.save_dir / "results.tsv"
        self._seen_hashes: set[str] = set()
        # Load existing hashes if file exists
        if self.tsv_path.exists():
            for row in self.load_all():
                self._seen_hashes.add(row["config_hash"])

    def append_row(
        self,
        round_num: int,
        phase: str,
        config: dict,
        throughput: float | None,
        latency_p99: float | None,
        task_metric: float | None,
        memory_pct: float | None,
        status: str,
        reasoning: str,
    ) -> bool:
        """Append a row. Returns False (skipped) if config_hash already seen."""
        h = compute_config_hash(config)
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)

        write_header = not self.tsv_path.exists()
        with open(self.tsv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "round": round_num,
                "phase": phase,
                "config_hash": h,
                "config_json": json.dumps(config, sort_keys=True),
                "throughput": f"{throughput:.1f}" if throughput is not None else "-",
                "latency_p99": f"{latency_p99:.0f}" if latency_p99 is not None else "-",
                "task_metric": f"{task_metric:.4f}" if task_metric is not None else "-",
                "memory_pct": f"{memory_pct:.2f}" if memory_pct is not None else "-",
                "status": status,
                "reasoning": reasoning,
            })
        return True

    def load_all(self) -> list[dict]:
        if not self.tsv_path.exists():
            return []
        with open(self.tsv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_reporter.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/reporter.py tests/test_reporter.py
git commit -m "feat: Reporter with TSV append, config_hash dedup, config_json column, and load_all"
```

---

## Task 7: Actor (Service Lifecycle)

**Files:**
- Create: `vllm_tuner/actor.py`
- Create: `tests/test_actor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_actor.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.actor import Actor

def make_actor():
    remote = MagicMock()
    framework = MagicMock()
    framework.build_start_cmd.return_value = "python -m vllm.entrypoints.openai.api_server"
    framework.get_health_endpoint.return_value = "http://1.2.3.4:8000/health"
    return Actor(remote=remote, framework=framework, work_dir="/workspace",
                 host="1.2.3.4", port=8000)

def test_fast_fail_passes_when_healthy():
    actor = make_actor()
    with patch("vllm_tuner.actor.requests.post") as mock_post, \
         patch("vllm_tuner.actor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "hi"}}]}
        )
        assert actor.fast_fail_check() is True

def test_fast_fail_fails_when_health_down():
    actor = make_actor()
    with patch("vllm_tuner.actor.requests.get") as mock_get:
        mock_get.side_effect = Exception("connection refused")
        assert actor.fast_fail_check() is False

def test_stop_service_kills_named_process():
    actor = make_actor()
    # Four separate calls: lsof (PID), ps (process name), kill -15, ps (check after kill)
    actor.remote.run.side_effect = [
        MagicMock(stdout="12345\n", returncode=0),   # lsof → PID only
        MagicMock(stdout="vllm\n", returncode=0),    # ps → process name
        MagicMock(stdout="", returncode=0),          # kill -15
        MagicMock(stdout="", returncode=0),          # ps check after kill (dead)
    ]
    actor.stop_service()
    calls = [str(c) for c in actor.remote.run.call_args_list]
    assert any("kill" in c for c in calls)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_actor.py -v
```

- [ ] **Step 3: Implement actor.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_actor.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/actor.py tests/test_actor.py
git commit -m "feat: Actor service lifecycle with safe stop, start, restart, fast-fail check"
```

---

## Task 8: Ship-It (Phase 0)

**Files:**
- Create: `vllm_tuner/shipit.py`
- Create: `tests/test_shipit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shipit.py
from unittest.mock import MagicMock, patch, call
import pytest
from vllm_tuner.shipit import ShipIt

def make_shipit():
    remote = MagicMock()
    actor = MagicMock()
    actor.fast_fail_check.return_value = True
    framework = MagicMock()
    framework.get_health_endpoint.return_value = "http://1.2.3.4:8000/health"
    cfg = MagicMock()
    cfg.remote.working_dir = "/workspace"
    cfg.remote.vllm_port = 8000
    cfg.remote.host = "1.2.3.4"
    cfg.hardware.npu_id = 4
    cfg.hardware.chip_id = 0
    cfg.model.hf_url = "https://huggingface.co/Qwen/Qwen2-7B"
    cfg.model.local_path = "/workspace/models/Qwen2-7B"
    cfg.model.hf_token = ""
    return ShipIt(remote=remote, actor=actor, framework=framework, config=cfg)

def test_check_env_detects_vllm_installed():
    ship = make_shipit()
    # Exactly 6 calls matching check_remote_env() order:
    # 1. npu-smi usages, 2. npu-smi health, 3. python -c 'import vllm',
    # 4. df, 5. lsof (port), 6. ls (model dir)
    ship.remote.run.side_effect = [
        MagicMock(stdout="HBM-Usage(MB): 1000 / 65536\n"),  # npu-smi usages
        MagicMock(stdout="OK\n"),                            # npu-smi health
        MagicMock(stdout="", returncode=0),                  # python -c 'import vllm'
        MagicMock(stdout="/dev/sda  400G  50G  350G\n"),     # df
        MagicMock(stdout=""),                                # lsof (port not occupied)
        MagicMock(stdout="exists\n"),                        # ls model dir
    ]
    env = ship.check_remote_env()
    assert env["hbm_total_mb"] == 65536
    assert env["health"] == "OK"
    assert env["model_exists"] is True

def test_pull_model_constructs_correct_command():
    ship = make_shipit()
    ship.remote.run.return_value = MagicMock(stdout="", returncode=0)
    ship.pull_model()
    calls = [str(c) for c in ship.remote.run.call_args_list]
    assert any("huggingface-cli download" in c for c in calls)
    assert any("Qwen/Qwen2-7B" in c for c in calls)
    assert any("/workspace/models/Qwen2-7B" in c for c in calls)

def test_port_cleanup_refuses_to_kill_sshd():
    ship = make_shipit()
    ship.remote.run.side_effect = [
        MagicMock(stdout="1234\n"),       # lsof pid
        MagicMock(stdout="sshd\n"),       # ps proc name
    ]
    with pytest.raises(RuntimeError, match="Refusing"):
        ship._clear_port(8000)

def test_self_heal_retries_up_to_3_times():
    ship = make_shipit()
    brain = MagicMock()
    brain.diagnose.return_value = {
        "fix_commands": ["pip install vllm-ascend --upgrade"],
        "diagnosis": "missing package",
        "adjusted_param": {},
    }
    # actor.start_service always fails (simulates persistent crash)
    ship.actor.start_service.side_effect = RuntimeError("OOM")
    ship.remote.read_log_tail = MagicMock(return_value="OOM in log")
    with pytest.raises(RuntimeError, match="self-heal"):
        ship.self_heal(error="OOM", attempt_history=[], brain=brain,
                       default_infra_config={})
    assert brain.diagnose.call_count == 3
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_shipit.py -v
```

- [ ] **Step 3: Implement shipit.py**

```python
# vllm_tuner/shipit.py
import re
import time


class ShipIt:
    """Phase 0: Automated deployment agent. Can be used standalone."""

    def __init__(self, remote, actor, framework, config):
        self.remote = remote
        self.actor = actor
        self.framework = framework
        self.cfg = config

    def run(self, default_infra_config: dict) -> bool:
        """Full deployment pipeline. Returns True on success."""
        print("[Ship-It] Checking remote environment...")
        env = self.check_remote_env()
        print(f"[✓] HBM: {env['hbm_total_mb']}MB, Health: {env['health']}")

        self.auto_fix_env(env)

        if not env["model_exists"]:
            print(f"[↓] Pulling model {self.cfg.model.hf_url}...")
            self.pull_model()
            print("[✓] Model pull complete")

        print("[...] Starting inference service...")
        self.actor.start_service(self.cfg.model.local_path, default_infra_config)

        if not self.actor.fast_fail_check():
            logs = self.remote.read_log_tail(
                f"{self.cfg.remote.working_dir}/vllm.log"
            )
            # Attempt self-healing (requires brain to be passed in or set externally)
            if hasattr(self, "brain") and self.brain is not None:
                self.self_heal(
                    error="fast-fail check failed",
                    attempt_history=[],
                    brain=self.brain,
                    default_infra_config=default_infra_config,
                )
            else:
                raise RuntimeError(f"Service fast-fail check failed. Logs:\n{logs}")

        print("[✓] Service healthy and responding")
        return True

    def self_heal(
        self,
        error: str,
        attempt_history: list,
        brain,
        default_infra_config: dict,
    ):
        """
        Retry deployment up to 3 times, asking Claude to diagnose and fix each time.
        Structurally deduplicates fix_commands vs attempt_history to avoid repeating
        the same commands (does not rely on LLM compliance alone).
        Raises RuntimeError after 3 failures.
        """
        for attempt in range(3):
            logs = self.remote.read_log_tail(
                f"{self.cfg.remote.working_dir}/vllm.log"
            )
            diagnosis = brain.diagnose(
                logs=logs, error=error, attempt_history=attempt_history
            )
            # Structural dedup: only run commands not already tried
            already_tried = {cmd for h in attempt_history for cmd in h.get("cmd", [])}
            new_cmds = [
                cmd for cmd in diagnosis.get("fix_commands", [])
                if cmd not in already_tried
            ]
            for cmd in new_cmds:
                self.remote.run(cmd, timeout=120)
            attempt_history.append({"attempt": attempt, "error": error, "cmd": new_cmds})

            try:
                self.actor.start_service(
                    self.cfg.model.local_path, default_infra_config
                )
                if self.actor.fast_fail_check():
                    return  # Success
                error = "fast-fail check failed after fix attempt"
            except Exception as e:
                error = str(e)

        raise RuntimeError(
            f"self-heal exhausted 3 attempts. "
            f"Check {self.cfg.remote.working_dir}/vllm.log on the remote machine."
        )

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

        # vLLM installed
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

    def auto_fix_env(self, env: dict):
        if not env["vllm_installed"]:
            print("[Fix] Installing vllm-ascend...")
            self.remote.run("pip install vllm-ascend", timeout=300)

        if env["port_occupied"]:
            print(f"[!] Port {self.cfg.remote.vllm_port} occupied, clearing...")
            self._clear_port(self.cfg.remote.vllm_port)

        # Activate CANN if ascend
        if self.cfg.hardware.type == "ascend":
            self.remote.run(
                "source /usr/local/Ascend/ascend-toolkit/set_env.sh || true"
            )

    def pull_model(self):
        url = self.cfg.model.hf_url
        repo_id = "/".join(url.rstrip("/").split("/")[-2:])
        local_path = self.cfg.model.local_path

        # Prefer remote env var for token
        token_part = ""
        token_check = self.remote.run("echo $HF_TOKEN").stdout.strip()
        if not token_check and self.cfg.model.hf_token:
            token_part = f" --token {self.cfg.model.hf_token}"

        cmd = (
            f"huggingface-cli download {repo_id}"
            f" --local-dir {local_path}{token_part}"
        )
        self.remote.run(cmd, timeout=3600)

    def _clear_port(self, port: int):
        pid_result = self.remote.run(f"lsof -ti:{port} -sTCP:LISTEN")
        pid = pid_result.stdout.strip()
        if not pid:
            return
        name_result = self.remote.run(f"ps -p {pid} -o comm=")
        proc_name = name_result.stdout.strip().lower()
        safe = ("python", "vllm", "lmdeploy", "sglang")
        if not any(s in proc_name for s in safe):
            raise RuntimeError(
                f"Refusing to kill PID {pid} (process: {proc_name!r}) on port {port}."
            )
        self.remote.run(f"kill -15 {pid}")
        time.sleep(3)
        check = self.remote.run(f"ps -p {pid} -o pid=")
        if check.stdout.strip():
            self.remote.run(f"kill -9 {pid}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_shipit.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/shipit.py tests/test_shipit.py
git commit -m "feat: ShipIt Phase 0 - env check, model pull, safe port cleanup, deploy, self_heal"
```

---

## Task 9: Brain (Claude Decision Engine)

**Files:**
- Create: `vllm_tuner/brain.py`
- Create: `tests/test_brain.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_brain.py
import json
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.brain import Brain

def make_brain():
    return Brain(api_key="test-key", model="claude-sonnet-4-6")

def mock_claude_response(content: dict):
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(content))]
    return msg

def test_decide_next_config_returns_dict():
    brain = make_brain()
    with patch("vllm_tuner.brain.anthropic.Anthropic") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        client.messages.create.return_value = mock_claude_response({
            "optimization_focus": "speed",
            "next_config": {"block_size": 8, "gpu_memory_utilization": 0.85},
            "reasoning": "KV碎片高",
            "confidence": 0.8,
            "skip_reason": None
        })
        brain = Brain(api_key="test-key", model="claude-sonnet-4-6")
        result = brain.decide_next_config(
            sweep_result={}, history=[], hw_stats=None,
            phase="2a", param_space={"block_size": [8, 16]},
            seen_hashes=set()
        )
    assert "next_config" in result
    assert "reasoning" in result

def test_diagnose_returns_fix_commands():
    brain = make_brain()
    with patch("vllm_tuner.brain.anthropic.Anthropic") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        client.messages.create.return_value = mock_claude_response({
            "diagnosis": "OOM due to large block_size",
            "fix_commands": ["pip install vllm-ascend --upgrade"],
            "adjusted_param": {"gpu_memory_utilization": 0.80}
        })
        brain = Brain(api_key="test-key", model="claude-sonnet-4-6")
        result = brain.diagnose(
            logs="CANN OOM error...", error="OOM", attempt_history=[]
        )
    assert "fix_commands" in result
    assert isinstance(result["fix_commands"], list)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_brain.py -v
```

- [ ] **Step 3: Implement brain.py**

```python
# vllm_tuner/brain.py
import json
import anthropic


DECIDE_SYSTEM = """You are a performance engineer specializing in LLM inference on Ascend NPU.
Given hardware metrics and optimization history, decide the next parameter configuration to test.

Hardware signal interpretation:
- HBM usage > 90% + AICore < 50%: block_size too large, KV cache fragmentation
- AICore < 30% + low throughput: max_num_seqs too low, insufficient concurrency
- Temp > 85C: skip this round, wait for cooling
- Health = Warning: reduce confidence, pick conservative parameters

Always output valid JSON matching this schema:
{
  "optimization_focus": "speed" | "accuracy",
  "next_config": {<param>: <value>, ...},
  "reasoning": "<causal explanation>",
  "confidence": 0.0-1.0,
  "skip_reason": null | "<reason to skip this round>"
}

Do NOT suggest configs whose hash already appears in seen_hashes."""

DIAGNOSE_SYSTEM = """You are a systems debugger for vLLM on Ascend NPU.
Given error logs and previous failed fix attempts, diagnose the root cause and suggest new fix commands.
Do NOT repeat commands from attempt_history.

Output valid JSON:
{
  "diagnosis": "<root cause>",
  "fix_commands": ["<shell command>", ...],
  "adjusted_param": {<param>: <value>}
}"""


class Brain:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def decide_next_config(
        self,
        sweep_result: dict,
        history: list[dict],
        hw_stats,
        phase: str,
        param_space: dict,
        seen_hashes: set,
    ) -> dict:
        hw_str = str(hw_stats) if hw_stats else "unavailable"
        history_str = json.dumps(history[-10:], ensure_ascii=False)  # last 10 rows

        user_msg = (
            f"Phase: {phase}\n"
            f"Parameter space: {json.dumps(param_space)}\n"
            f"Seen config hashes (skip these): {list(seen_hashes)}\n"
            f"Sweep baseline: {json.dumps(sweep_result)}\n"
            f"Recent history (last 10 rounds): {history_str}\n"
            f"Current hardware snapshot: {hw_str}\n\n"
            "Decide the next config to test."
        )

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=DECIDE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return json.loads(msg.content[0].text)

    def diagnose(self, logs: str, error: str, attempt_history: list) -> dict:
        history_str = json.dumps(attempt_history, ensure_ascii=False)
        user_msg = (
            f"Error: {error}\n"
            f"Log tail:\n{logs}\n"
            f"Previous failed attempts (do NOT repeat): {history_str}\n\n"
            "Diagnose and suggest new fix commands."
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=DIAGNOSE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return json.loads(msg.content[0].text)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_brain.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/brain.py tests/test_brain.py
git commit -m "feat: Brain Claude decision engine with decide_next_config and diagnose"
```

---

## Task 10: Sweep (Phase 1)

**Files:**
- Create: `vllm_tuner/sweep.py`
- Create: `tests/test_sweep.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.sweep import BenchmarkSweep

def test_sweep_builds_matrix():
    with patch("vllm_tuner.sweep.requests.post") as mock_post, \
         patch("vllm_tuner.sweep.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "hi"}}],
                          "usage": {"completion_tokens": 10}}
        )
        sweep = BenchmarkSweep(
            server_url="http://1.2.3.4:8000/v1",
            health_url="http://1.2.3.4:8000/health",
            concurrency_levels=[1, 2],
            input_lengths=[128, 256],
            requests_per_cell=2,
            request_timeout=10,
        )
        result = sweep.run()
    assert "matrix" in result
    assert "1" in result["matrix"]
    assert "128" in result["matrix"]["1"]
    assert result["matrix"]["1"]["128"]["status"] == "OK"

def test_sweep_marks_oom_on_health_failure():
    call_count = [0]

    def post_side_effect(url, **kwargs):
        call_count[0] += 1
        if call_count[0] > 2:
            raise Exception("timeout")
        return MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "x"}}],
                          "usage": {"completion_tokens": 5}}
        )

    with patch("vllm_tuner.sweep.requests.post", side_effect=post_side_effect), \
         patch("vllm_tuner.sweep.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(status_code=200),  # healthy at start
            Exception("down"),           # health check after timeout → OOM
        ]
        sweep = BenchmarkSweep(
            server_url="http://1.2.3.4:8000/v1",
            health_url="http://1.2.3.4:8000/health",
            concurrency_levels=[1, 4],
            input_lengths=[128, 512],
            requests_per_cell=2,
            request_timeout=5,
        )
        result = sweep.run()
    # At least one OOM cell exists
    has_oom = any(
        cell.get("status") == "OOM"
        for conc_data in result["matrix"].values()
        for cell in conc_data.values()
    )
    assert has_oom
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_sweep.py -v
```

- [ ] **Step 3: Implement sweep.py**

```python
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
        concurrency_levels: list[int],
        input_lengths: list[int],
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

    def _single_request(self, input_len: int) -> tuple[float, float]:
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

    def _recommend(self, matrix: dict, oom_boundary: dict | None) -> dict:
        safe_concurrencies = [
            int(c) for c, cells in matrix.items()
            if any(v.get("status") == "OK" for v in cells.values())
        ]
        # block_size recommendation: prefer 8 or 16 for better KV cache efficiency;
        # always include full candidate list so Brain can choose based on hardware signals.
        return {
            "max_num_seqs": safe_concurrencies[:3] if safe_concurrencies else [1],
            "block_size": [8, 16, 32],
            "note": f"OOM boundary: {oom_boundary}" if oom_boundary else "No OOM observed",
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sweep.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/sweep.py tests/test_sweep.py
git commit -m "feat: BenchmarkSweep Phase 1 - concurrency x input_len matrix with OOM detection"
```

---

## Task 11: Orchestrator (Phase 2a + 2b)

**Files:**
- Create: `vllm_tuner/orchestrator.py`
- Create: `vllm_tuner/runner.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orchestrator.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.orchestrator import Orchestrator

def make_orchestrator():
    cfg = MagicMock()
    cfg.model.local_path = "/models/Qwen2-7B"
    cfg.remote.host = "1.2.3.4"
    cfg.remote.vllm_port = 8000
    cfg.evaluation.metric_direction = "maximize"
    cfg.optimization.phase_2a.max_rounds = 3
    cfg.optimization.phase_2a.patience = 2
    cfg.optimization.phase_2a.parameters = {"block_size": [8, 16]}
    cfg.optimization.phase_2b.max_rounds = 3
    cfg.optimization.phase_2b.patience = 2
    cfg.optimization.phase_2b.baseline = {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0
    }
    cfg.optimization.phase_2b.parameters = {"temperature": [0.7, 0.8]}
    cfg.evaluation.skills = ["throughput", "latency"]

    actor = MagicMock()
    actor.fast_fail_check.return_value = True

    brain = MagicMock()
    brain.decide_next_config.return_value = {
        "next_config": {"block_size": 8},
        "reasoning": "test",
        "confidence": 0.9,
        "skip_reason": None,
        "optimization_focus": "speed",
    }

    runner = MagicMock()
    runner.run_all.return_value = {
        "throughput": 900.0, "latency_p99": 130.0,
        "task_metric": None, "memory_pct": 0.8
    }

    reporter = MagicMock()
    reporter.append_row.return_value = True
    reporter.load_all.return_value = []

    return Orchestrator(
        config=cfg, actor=actor, brain=brain,
        runner=runner, reporter=reporter,
        sweep_result={}, observer=MagicMock()
    )

def test_phase_2a_runs_at_most_max_rounds():
    orch = make_orchestrator()
    best = orch.run_phase_2a()
    assert orch.actor.restart_service.call_count <= 3

def test_phase_2a_returns_best_infra_config():
    orch = make_orchestrator()
    best = orch.run_phase_2a()
    assert "block_size" in best or best == {}
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_orchestrator.py -v
```

- [ ] **Step 3: Implement runner.py and orchestrator.py**

```python
# vllm_tuner/runner.py
class Runner:
    """Runs EvalSkills sequentially for a single round."""

    def __init__(self, skills: list, server_url: str):
        self.skills = skills
        self.server_url = server_url

    def run_all(self, gen_params: dict) -> dict:
        results = {}
        for skill in self.skills:
            try:
                val = skill.measure(self.server_url, gen_params)
                if isinstance(val, dict):
                    results.update(val)
                else:
                    results[skill.name] = val
            except Exception as e:
                results[skill.name] = None
                print(f"[!] Skill {skill.name} failed: {e}")

        return {
            "throughput": results.get("throughput"),
            "latency_p99": results.get("p99_ms") or results.get("latency"),
            "task_metric": results.get("task_metric"),
            "memory_pct": results.get("memory"),
        }
```

```python
# vllm_tuner/orchestrator.py
from vllm_tuner.reporter import compute_config_hash


class Orchestrator:
    def __init__(self, config, actor, brain, runner, reporter, sweep_result, observer):
        self.cfg = config
        self.actor = actor
        self.brain = brain
        self.runner = runner
        self.reporter = reporter
        self.sweep_result = sweep_result
        self.observer = observer
        self._round = 0

    def run_phase_2a(self) -> dict:
        """Infra tuning: restarts service each round. Returns best_infra_config."""
        cfg2a = self.cfg.optimization.phase_2a
        best_config = {}
        best_throughput = 0.0
        best_lat_ms: float | None = None
        no_improve = 0

        for _ in range(cfg2a.max_rounds):
            self._round += 1
            history = self.reporter.load_all()
            seen = {r["config_hash"] for r in history}
            hw_stats = self.observer.get_stats() if self.observer else None

            decision = self.brain.decide_next_config(
                sweep_result=self.sweep_result,
                history=history,
                hw_stats=hw_stats,
                phase="2a",
                param_space=cfg2a.parameters,
                seen_hashes=seen,
            )

            if decision.get("skip_reason"):
                print(f"[Round {self._round}] Skipped: {decision['skip_reason']}")
                continue

            next_cfg = decision["next_config"]
            cfg_hash = compute_config_hash(next_cfg)
            if cfg_hash in seen:
                no_improve += 1
                if no_improve >= cfg2a.patience:
                    break
                continue

            # Restart with new infra config; self-heal on failure (max 3 retries)
            attempt_history = []
            deployed = False
            for attempt in range(3):
                try:
                    self.actor.restart_service(self.cfg.model.local_path, next_cfg)
                    if not self.actor.fast_fail_check():
                        raise RuntimeError("fast-fail check failed after restart")
                    deployed = True
                    break
                except Exception as e:
                    logs = self.actor.remote.read_log_tail(
                        f"{self.cfg.remote.working_dir}/vllm.log"
                    )
                    diagnosis = self.brain.diagnose(
                        logs=logs, error=str(e), attempt_history=attempt_history
                    )
                    # Structural dedup: flatten all previously tried commands into a set,
                    # then skip any commands that appear in that set.
                    already_tried = {cmd for h in attempt_history for cmd in h.get("cmd", [])}
                    new_cmds = [
                        cmd for cmd in diagnosis.get("fix_commands", [])
                        if cmd not in already_tried
                    ]
                    for cmd in new_cmds:
                        self.actor.remote.run(cmd, timeout=120)
                    attempt_history.append({"attempt": attempt, "error": str(e),
                                            "cmd": new_cmds})

            if not deployed:
                self.reporter.append_row(
                    self._round, "2a", next_cfg, None, None, None, None,
                    "CRASH", f"Failed after 3 self-heal attempts"
                )
                print(f"[Round {self._round}] CRASH: exhausted self-heal retries")
                continue

            metrics = self.runner.run_all(gen_params={})
            tps = metrics.get("throughput") or 0.0
            lat = metrics.get("latency_p99") or 0.0

            # KEEP if: at least one of (throughput, latency) improves,
            # AND the other does not regress. Spec: dual-objective KEEP condition.
            prev_lat = best_lat_ms if best_lat_ms is not None else float("inf")
            tps_improved = tps > best_throughput
            lat_improved = lat < prev_lat
            tps_ok = tps >= best_throughput * 0.99   # allow 1% tolerance
            lat_ok = lat <= prev_lat * 1.01
            improved = (tps_improved and lat_ok) or (lat_improved and tps_ok)
            status = "KEEP" if improved else "DISCARD"
            self.reporter.append_row(
                self._round, "2a", next_cfg,
                tps, lat, None, metrics.get("memory_pct"),
                status, decision.get("reasoning", "")
            )
            print(f"[Round {self._round}/2a] {next_cfg} → {tps:.0f}tok/s, P99={lat:.0f}ms  {status}")

            if improved:
                best_throughput = tps
                best_lat_ms = lat
                best_config = next_cfg
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg2a.patience:
                print(f"[Converged] {no_improve} rounds without improvement")
                break

        return best_config

    def run_phase_2b(self, infra_config: dict) -> dict:
        """Generation param tuning: no restart. Returns best_gen_config."""
        cfg2b = self.cfg.optimization.phase_2b
        direction = self.cfg.evaluation.metric_direction  # maximize or minimize
        best_metric = float("-inf") if direction == "maximize" else float("inf")
        best_config = dict(cfg2b.baseline)
        no_improve = 0

        for _ in range(cfg2b.max_rounds):
            self._round += 1
            history = self.reporter.load_all()
            seen = {r["config_hash"] for r in history}

            decision = self.brain.decide_next_config(
                sweep_result=self.sweep_result,
                history=history,
                hw_stats=None,
                phase="2b",
                param_space=cfg2b.parameters,
                seen_hashes=seen,
            )

            if decision.get("skip_reason"):
                continue

            next_gen = decision["next_config"]
            cfg_hash = compute_config_hash(next_gen)
            if cfg_hash in seen:
                no_improve += 1
                if no_improve >= cfg2b.patience:
                    break
                continue

            # No restart — pass gen params via API
            metrics = self.runner.run_all(gen_params=next_gen)
            task_val = metrics.get("task_metric")

            if task_val is None:
                status = "CRASH"
            else:
                if direction == "maximize":
                    improved = task_val > best_metric
                else:
                    improved = task_val < best_metric
                status = "KEEP" if improved else "DISCARD"

            self.reporter.append_row(
                self._round, "2b", next_gen,
                metrics.get("throughput"), metrics.get("latency_p99"),
                task_val, metrics.get("memory_pct"),
                status, decision.get("reasoning", "")
            )
            print(f"[Round {self._round}/2b] {next_gen} → metric={task_val}  {status}")

            if status == "KEEP":
                best_metric = task_val
                best_config = next_gen
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg2b.patience:
                print(f"[Converged] {no_improve} rounds without improvement")
                break

        return best_config
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_orchestrator.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_tuner/orchestrator.py vllm_tuner/runner.py tests/test_orchestrator.py
git commit -m "feat: Orchestrator Phase 2a/2b loop with patience convergence and Runner"
```

---

## Task 12: Analysis (Pareto) + Reporter (Final Report)

**Files:**
- Create: `vllm_tuner/analysis.py`
- Modify: `vllm_tuner/reporter.py` (add `generate_best_report`)
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analysis.py
import pytest
from vllm_tuner.analysis import is_pareto_dominated, get_pareto_frontier, select_bests

def make_rows(data):
    return [
        {"phase": r[0], "throughput": str(r[1]), "task_metric": str(r[2]),
         "latency_p99": str(r[3]), "status": "KEEP",
         "config_hash": f"h{i}", "round": str(i)}
        for i, r in enumerate(data)
    ]

def test_pareto_dominated_maximize():
    candidate = {"throughput": 800.0, "task_metric": 0.60}
    others = [{"throughput": 900.0, "task_metric": 0.65}]
    assert is_pareto_dominated(candidate, others, "maximize") is True

def test_pareto_not_dominated_tradeoff():
    candidate = {"throughput": 1000.0, "task_metric": 0.60}
    others = [{"throughput": 800.0, "task_metric": 0.70}]
    # neither dominates the other (tradeoff)
    assert is_pareto_dominated(candidate, others, "maximize") is False

def test_pareto_dominated_minimize():
    # lower task_metric is better for minimize
    candidate = {"throughput": 800.0, "task_metric": 0.50}
    others = [{"throughput": 900.0, "task_metric": 0.40}]
    assert is_pareto_dominated(candidate, others, "minimize") is True

def test_get_pareto_frontier_returns_non_dominated():
    rows = make_rows([
        ("2b", 800, 0.60, 150),  # dominated
        ("2b", 1000, 0.60, 130), # on frontier (best tps)
        ("2b", 800, 0.70, 150),  # on frontier (best accuracy)
    ])
    frontier = get_pareto_frontier(rows, "maximize")
    assert len(frontier) == 2

def test_select_bests_from_phase_2a():
    rows = make_rows([
        ("2a", 820, "-", 180),
        ("2a", 1050, "-", 145),
        ("2a", 950, "-", 200),
    ])
    # Fix task_metric for 2a rows to "-"
    for r in rows:
        r["task_metric"] = "-"
    bests = select_bests(rows)
    assert bests["best_latency"]["latency_p99"] == "145"
    assert bests["best_throughput"]["throughput"] == "1050"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_analysis.py -v
```

- [ ] **Step 3: Implement analysis.py**

```python
# vllm_tuner/analysis.py
from __future__ import annotations


def is_pareto_dominated(candidate: dict, others: list[dict], metric_direction: str) -> bool:
    """
    Returns True if candidate is dominated by any row in others.
    For maximize: higher throughput and task_metric is better.
    For minimize: higher throughput, LOWER task_metric is better.
    """
    def not_worse_metric(other_val, cand_val):
        if metric_direction == "maximize":
            return other_val >= cand_val
        else:  # minimize
            return other_val <= cand_val

    def strictly_better_metric(other_val, cand_val):
        if metric_direction == "maximize":
            return other_val > cand_val
        else:
            return other_val < cand_val

    cand_tps = candidate["throughput"]
    cand_metric = candidate["task_metric"]

    for other in others:
        if other is candidate:
            continue
        o_tps = other["throughput"]
        o_metric = other["task_metric"]

        tps_not_worse = o_tps >= cand_tps
        metric_not_worse = not_worse_metric(o_metric, cand_metric)
        tps_better = o_tps > cand_tps
        metric_better = strictly_better_metric(o_metric, cand_metric)

        if tps_not_worse and metric_not_worse and (tps_better or metric_better):
            return True
    return False


def get_pareto_frontier(phase2b_rows: list[dict], metric_direction: str) -> list[dict]:
    """Only Phase 2b rows with numeric task_metric. Returns non-dominated configs."""
    valid = []
    for r in phase2b_rows:
        if r.get("phase") != "2b" or r.get("task_metric") == "-":
            continue
        if r.get("status") not in ("KEEP", "DISCARD"):
            continue
        try:
            valid.append({
                **r,
                "throughput": float(r["throughput"]),
                "task_metric": float(r["task_metric"]),
            })
        except (ValueError, TypeError):
            continue

    return [r for r in valid if not is_pareto_dominated(r, valid, metric_direction)]


def select_bests(all_rows: list[dict]) -> dict:
    """Select best_latency and best_throughput from Phase 2a KEEP rows."""
    phase2a = [
        r for r in all_rows
        if r.get("phase") == "2a" and r.get("status") == "KEEP"
        and r.get("throughput") not in (None, "-")
        and r.get("latency_p99") not in (None, "-")
    ]
    if not phase2a:
        return {"best_latency": None, "best_throughput": None}

    best_lat = min(phase2a, key=lambda r: float(r["latency_p99"]))
    best_tps = max(phase2a, key=lambda r: float(r["throughput"]))
    return {"best_latency": best_lat, "best_throughput": best_tps}
```

- [ ] **Step 4: Add `generate_best_report` as a method inside the `Reporter` class in reporter.py**

**IMPORTANT:** This is a method of `Reporter`, not a top-level function. Insert it inside the class body, indented at the same level as `append_row` and `load_all`.

```python
# vllm_tuner/reporter.py  — add inside class Reporter, after load_all():

    def generate_best_report(
        self,
        model_name: str,
        hardware_name: str,
        metric_direction: str,
    ) -> str:
        """Generate best_config.md. Returns path to written file."""
        from vllm_tuner.analysis import get_pareto_frontier, select_bests
        rows = self.load_all()
        bests = select_bests(rows)
        frontier = get_pareto_frontier(rows, metric_direction)

        lines = [
            "# vLLM-Tuner Optimization Report",
            f"## Model: {model_name} | Hardware: {hardware_name}",
            "",
        ]

        b_lat = bests.get("best_latency")
        b_tps = bests.get("best_throughput")
        if b_lat:
            lines += [f"### ① Best Latency (Round {b_lat['round']})",
                      f"Config hash: {b_lat['config_hash']}",
                      f"P99 latency: {b_lat['latency_p99']}ms", ""]
        if b_tps:
            lines += [f"### ② Best Throughput (Round {b_tps['round']})",
                      f"Config hash: {b_tps['config_hash']}",
                      f"Throughput: {b_tps['throughput']} tok/s", ""]

        phase2b = [r for r in rows if r.get("phase") == "2b" and r.get("status") == "KEEP"
                   and r.get("task_metric") not in (None, "-")]
        if phase2b:
            best_acc = (max if metric_direction == "maximize" else min)(
                phase2b, key=lambda r: float(r["task_metric"])
            )
            lines += [f"### ③ Best Accuracy (Round {best_acc['round']})",
                      f"Config hash: {best_acc['config_hash']}",
                      f"Task metric: {best_acc['task_metric']}", ""]

        if frontier:
            rec = frontier[0]
            lines += ["### ⭐ Pareto Recommendation",
                      f"Round {rec['round']}: throughput={rec['throughput']:.0f}, "
                      f"metric={rec['task_metric']:.4f}", ""]

        report_path = self.save_dir / "best_config.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_analysis.py tests/test_reporter.py -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add vllm_tuner/analysis.py vllm_tuner/reporter.py tests/test_analysis.py
git commit -m "feat: Pareto analysis with metric_direction support + best report generation"
```

---

## Task 13: main.py CLI Entry Point

**Files:**
- Create: `vllm_tuner/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_main.py
import sys
from unittest.mock import patch, MagicMock
import pytest
from vllm_tuner.main import build_parser

def test_parser_run_requires_config():
    parser = build_parser()
    args = parser.parse_args(["run", "--config", "tuner_config.yaml"])
    assert args.config == "tuner_config.yaml"
    assert args.command == "run"

def test_parser_deploy_subcommand():
    parser = build_parser()
    args = parser.parse_args(["deploy", "--config", "tuner_config.yaml"])
    assert args.command == "deploy"

def test_parser_deploy_with_best_config():
    parser = build_parser()
    args = parser.parse_args([
        "deploy", "--config", "cfg.yaml",
        "--use-best-config", "results/best_config.md"
    ])
    assert args.use_best_config == "results/best_config.md"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_main.py -v
```

- [ ] **Step 3: Implement main.py**

```python
# vllm_tuner/main.py
import argparse
import os
import sys

from vllm_tuner.config import load_config
from vllm_tuner.remote_env import RemoteEnv
from vllm_tuner.frameworks import get_framework
from vllm_tuner.hardware import get_observer
from vllm_tuner.actor import Actor
from vllm_tuner.shipit import ShipIt
from vllm_tuner.sweep import BenchmarkSweep
from vllm_tuner.brain import Brain
from vllm_tuner.runner import Runner
from vllm_tuner.reporter import Reporter
from vllm_tuner.orchestrator import Orchestrator
from vllm_tuner.skills import ThroughputSkill, LatencySkill, TaskMetricSkill, MemorySkill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-tuner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Full optimization pipeline")
    run_p.add_argument("--config", required=True, help="Path to tuner_config.yaml")

    deploy_p = sub.add_parser("deploy", help="Deploy only (Ship-It)")
    deploy_p.add_argument("--config", required=True)
    deploy_p.add_argument("--use-best-config", default=None,
                          help="Read best infra+gen params from this report file")
    return parser


def _build_skills(cfg, remote, observer):
    skill_map = {
        "throughput": ThroughputSkill(concurrency=4, num_requests=20, input_len=256),
        "latency": LatencySkill(num_requests=10, input_len=256),
        "task_metric": TaskMetricSkill(
            remote=remote,
            script=cfg.evaluation.script,
            data_dir=cfg.evaluation.data,
            sample_size=cfg.evaluation.sample_size,
            timeout_seconds=cfg.evaluation.timeout_seconds,
        ),
        "memory": MemorySkill(observer=observer),
    }
    return [skill_map[s] for s in cfg.evaluation.skills if s in skill_map]


def cmd_run(args):
    cfg = load_config(args.config)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    remote = RemoteEnv(cfg.remote)
    framework = get_framework(cfg.framework, host=cfg.remote.host, port=cfg.remote.vllm_port)
    observer = get_observer(cfg.hardware.type, remote,
                            npu_id=cfg.hardware.npu_id, chip_id=cfg.hardware.chip_id)
    actor = Actor(remote=remote, framework=framework,
                  work_dir=cfg.remote.working_dir,
                  host=cfg.remote.host, port=cfg.remote.vllm_port)
    brain = Brain(api_key=api_key)
    reporter = Reporter(save_dir=cfg.save_dir)
    skills = _build_skills(cfg, remote, observer)
    server_url = framework.get_api_base()
    runner = Runner(skills=skills, server_url=server_url)

    # Phase 0: Ship-It
    print("\n─── Phase 0: Ship-It ───")
    ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg)
    default_infra = {
        "block_size": cfg.optimization.phase_2a.parameters["block_size"][0],
        "gpu_memory_utilization": cfg.optimization.phase_2a.parameters.get(
            "gpu_memory_utilization", [0.85])[0],
    }
    ship.run(default_infra)

    # Phase 1: Sweep
    print("\n─── Phase 1: Sweep ───")
    sweep = BenchmarkSweep(
        server_url=server_url,
        health_url=framework.get_health_endpoint(),
        concurrency_levels=cfg.sweep.concurrency_levels,
        input_lengths=cfg.sweep.input_lengths,
        requests_per_cell=cfg.sweep.requests_per_cell,
        request_timeout=cfg.sweep.request_timeout_seconds,
    )
    sweep_result = sweep.run()
    print(f"[Sweep] Best: {sweep_result.get('best_throughput')}")

    # Phase 2a + 2b
    orch = Orchestrator(
        config=cfg, actor=actor, brain=brain,
        runner=runner, reporter=reporter,
        sweep_result=sweep_result, observer=observer,
    )
    print("\n─── Phase 2a: Infra Tuning ───")
    best_infra = orch.run_phase_2a()

    print("\n─── Phase 2b: Accuracy Tuning ───")
    orch.run_phase_2b(infra_config=best_infra)

    # Phase 3: Report
    print("\n─── Phase 3: Report ───")
    model_name = cfg.model.hf_url.split("/")[-1]
    report_path = reporter.generate_best_report(
        model_name=model_name,
        hardware_name=cfg.hardware.type.upper(),
        metric_direction=cfg.evaluation.metric_direction,
    )
    print(f"[✓] Report saved: {report_path}")
    print(f"[✓] Service running at http://{cfg.remote.host}:{cfg.remote.vllm_port}")
    remote.close()


def _load_best_infra_from_report(report_path: str, save_dir: str) -> dict | None:
    """
    Parse best_config.md to find the 'Best Throughput' config hash,
    then look it up in results.tsv (config_json column) to get the full param dict.
    Returns the infra param dict, or None if not found (caller uses config defaults).
    """
    import re, csv, json
    from pathlib import Path

    report = Path(report_path).read_text(encoding="utf-8")
    # Extract config hash from "Config hash: <hash>" under Best Throughput section
    m = re.search(r"### ② Best Throughput.*?Config hash: (\w+)", report, re.DOTALL)
    if not m:
        return None
    target_hash = m.group(1)

    tsv_path = Path(save_dir) / "results.tsv"
    if not tsv_path.exists():
        return None

    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("config_hash") == target_hash and row.get("phase") == "2a":
                config_json = row.get("config_json", "")
                if config_json and config_json != "-":
                    try:
                        return json.loads(config_json)
                    except json.JSONDecodeError:
                        pass
    return None


def cmd_deploy(args):
    cfg = load_config(args.config)
    remote = RemoteEnv(cfg.remote)
    framework = get_framework(cfg.framework, host=cfg.remote.host, port=cfg.remote.vllm_port)
    actor = Actor(remote=remote, framework=framework,
                  work_dir=cfg.remote.working_dir,
                  host=cfg.remote.host, port=cfg.remote.vllm_port)
    ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg)

    infra = {
        "block_size": cfg.optimization.phase_2a.parameters["block_size"][0],
        "gpu_memory_utilization": cfg.optimization.phase_2a.parameters.get(
            "gpu_memory_utilization", [0.85])[0],
    }

    if args.use_best_config:
        print(f"[Deploy] Loading best config from {args.use_best_config}")
        best = _load_best_infra_from_report(args.use_best_config, cfg.save_dir)
        if best:
            infra.update(best)
        else:
            print("[Deploy] Could not parse best infra params; using config defaults.")

    ship.run(infra)
    remote.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "deploy":
        cmd_deploy(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_main.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASSED

- [ ] **Step 6: Commit**

```bash
git add vllm_tuner/main.py tests/test_main.py
git commit -m "feat: main.py CLI with run and deploy subcommands - full pipeline wired"
```

---

## Task 14: Demo Fixture — nanoGPT / vLLM Target

**Files:**
- Create: `demo/eval_stub.py` — example eval_script that satisfies the contract

- [ ] **Step 1: Create eval_stub.py (satisfies eval_script contract)**

```python
# demo/eval_stub.py
"""
Example eval script satisfying vLLM-Tuner's eval_script contract.
Usage: python eval_stub.py --server-url URL --data-dir DIR --sample-size N
Stdout: {"metric": <float>}
"""
import argparse
import json
import random
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    correct = 0
    for i in range(args.sample_size):
        try:
            resp = requests.post(
                f"{args.server_url}/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": f"Question {i}: What is 2+2?"}],
                    "max_tokens": 10,
                },
                timeout=30,
            )
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            if "4" in answer:
                correct += 1
        except Exception:
            pass

    accuracy = correct / args.sample_size
    print(json.dumps({"metric": accuracy}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify eval_stub works standalone**

```bash
# (requires a running vLLM server — use during actual demo)
python demo/eval_stub.py --server-url http://localhost:8000/v1 --data-dir ./demo --sample-size 5
```

Expected output: `{"metric": 0.8}` (or similar)

- [ ] **Step 3: Commit**

```bash
git add demo/
git commit -m "demo: eval_stub.py satisfying task_metric contract, ready for hackathon"
```

---

## Task 15: Final Polish — Requirements + Smoke Test

**Files:**
- Create: `requirements.txt`
- Create: `README.md` (brief, for hackathon judges)

- [ ] **Step 1: Write requirements.txt**

```
fabric>=3.0
anthropic>=0.40
requests>=2.31
pyyaml>=6.0
pandas>=2.0
matplotlib>=3.7
pytest>=7.0
```

- [ ] **Step 2: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS, zero failures

- [ ] **Step 3: Verify CLI help works**

```bash
python -m vllm_tuner.main --help
python -m vllm_tuner.main run --help
python -m vllm_tuner.main deploy --help
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt README.md
git commit -m "chore: requirements.txt and final smoke test pass"
```

---

## Implementation Order Summary

| Task | Component | Tests | Depends on |
|------|-----------|-------|------------|
| 1 | Config loading | 2 | — |
| 2 | RemoteEnv SSH | 3 | 1 |
| 3 | Frameworks | 8 | 1 |
| 4 | Hardware observer | 4 | 2 |
| 5 | EvalSkills | 5 | 4 |
| 6 | Reporter + hash | 6 | 1 |
| 7 | Actor lifecycle | 3 | 2, 3 |
| 8 | Ship-It Phase 0 | 3 | 7 |
| 9 | Brain (Claude) | 2 | 1 |
| 10 | Sweep Phase 1 | 2 | 5, 7 |
| 11 | Orchestrator + Runner | 2 | 6, 7, 9 |
| 12 | Analysis + Report | 6 | 6, 11 |
| 13 | main.py CLI | 3 | all |
| 14 | Demo fixtures | 0 | 13 |
| 15 | Polish | 0 | all |
