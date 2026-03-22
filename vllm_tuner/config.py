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
    type: str
    npu_id: int = 0
    chip_id: int = 0


@dataclass
class ModelConfig:
    hf_url: str
    hf_token: str
    local_path: str = ""

    def _set_working_dir(self, working_dir: str):
        """Derive local_path from hf_url and working_dir if not set."""
        raise NotImplementedError


@dataclass
class EvaluationConfig:
    script: str
    data: str
    metric: str
    metric_direction: str
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
    """Load and validate tuner_config.yaml into TunerConfig dataclasses."""
    raise NotImplementedError
