import json
import anthropic


class Brain:
    """Claude API decision engine: causal reasoning for next config and failure diagnosis."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        pass

    def decide_next_config(
        self,
        sweep_result: dict,
        history: list[dict],
        hw_stats,
        phase: str,
        param_space: dict,
        seen_hashes: set,
    ) -> dict:
        """Ask Claude to pick next parameter config based on hardware signals and history."""
        raise NotImplementedError

    def diagnose(self, logs: str, error: str, attempt_history: list) -> dict:
        """Ask Claude to diagnose a deployment failure and suggest fix commands."""
        raise NotImplementedError
