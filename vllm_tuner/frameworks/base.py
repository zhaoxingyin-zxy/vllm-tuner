from abc import ABC, abstractmethod


class InferenceFramework(ABC):
    """Abstract base for inference server frameworks (vLLM, LMDeploy, SGLang)."""

    def __init__(self, host: str, port: int):
        pass

    @abstractmethod
    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build shell command to launch the inference server."""
        ...

    def get_api_base(self) -> str:
        """Return OpenAI-compatible API base URL."""
        raise NotImplementedError

    @abstractmethod
    def get_health_endpoint(self) -> str:
        """Return health check URL."""
        ...

    @property
    def tunable_infra_params(self) -> dict:
        """Phase 2a tunable parameter ranges (block_size, gpu_memory_utilization, max_num_seqs)."""
        raise NotImplementedError
