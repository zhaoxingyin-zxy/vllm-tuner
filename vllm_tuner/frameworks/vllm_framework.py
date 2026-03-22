from vllm_tuner.frameworks.base import InferenceFramework


class VLLMFramework(InferenceFramework):
    """vLLM on Ascend NPU via vllm-ascend plugin."""

    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build vllm.entrypoints.openai.api_server launch command."""
        raise NotImplementedError

    def get_health_endpoint(self) -> str:
        """Return http://host:port/health."""
        raise NotImplementedError

    @property
    def tunable_infra_params(self) -> dict:
        """vLLM-specific Phase 2a tunable ranges."""
        raise NotImplementedError
