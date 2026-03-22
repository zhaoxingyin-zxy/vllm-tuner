import json
from vllm_tuner.skills.base import EvalSkill


class TaskMetricSkill(EvalSkill):
    """Execute remote eval_script via SSH, parse stdout JSON {metric: float}."""

    name = "task_metric"

    def __init__(self, remote, script: str, data_dir: str,
                 sample_size: int, timeout_seconds: int):
        self.remote = remote
        self.script = script
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.timeout_seconds = timeout_seconds

    def measure(self, server_url: str, gen_params: dict) -> float:
        """SSH-execute eval_script and parse the metric value."""
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
