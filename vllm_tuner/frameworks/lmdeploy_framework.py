from vllm_tuner.frameworks.base import InferenceFramework


class LMDeployFramework(InferenceFramework):
    """LMDeploy — Ascend ecosystem alternative."""

    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build lmdeploy serve api_server launch command."""
        raise NotImplementedError

    def get_health_endpoint(self) -> str:
        """Return http://host:port/v1/models."""
        raise NotImplementedError
