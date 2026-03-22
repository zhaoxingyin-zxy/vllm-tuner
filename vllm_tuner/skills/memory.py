from vllm_tuner.skills.base import EvalSkill


class MemorySkill(EvalSkill):
    """Snapshot HBM/GPU memory utilization percentage."""

    name = "memory"

    def __init__(self, observer):
        self.observer = observer

    def measure(self, server_url: str, gen_params: dict) -> float:
        """Return HBM utilization percent."""
        return self.observer.get_stats().hbm_util_pct
