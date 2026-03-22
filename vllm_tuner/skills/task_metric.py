import json
from vllm_tuner.skills.base import EvalSkill


class TaskMetricSkill(EvalSkill):
    """Execute remote eval_script via SSH, parse stdout JSON {metric: float}."""

    name = "task_metric"

    def __init__(self, remote, script: str, data_dir: str,
                 sample_size: int, timeout_seconds: int):
        pass

    def measure(self, server_url: str, gen_params: dict) -> float:
        """SSH-execute eval_script and parse the metric value."""
        raise NotImplementedError
