from vllm_tuner.frameworks.base import InferenceFramework


class SGLangFramework(InferenceFramework):
    """SGLang — CUDA scene."""

    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build sglang.launch_server launch command."""
        raise NotImplementedError

    def get_health_endpoint(self) -> str:
        """Return http://host:port/health."""
        raise NotImplementedError
