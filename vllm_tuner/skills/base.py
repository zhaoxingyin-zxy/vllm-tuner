from abc import ABC, abstractmethod


class EvalSkill(ABC):
    """Abstract evaluation skill plugin."""

    name: str

    @abstractmethod
    def measure(self, server_url: str, gen_params: dict) -> float | dict:
        """Return scalar metric or dict of metrics."""
        ...
