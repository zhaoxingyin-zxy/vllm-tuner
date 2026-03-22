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
        """Derive local_path from last segment of hf_url under working_dir/models/."""
        if not self.local_path:
            model_name = self.hf_url.rstrip("/").split("/")[-1]
            self.local_path = f"{working_dir}/models/{model_name}"


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
