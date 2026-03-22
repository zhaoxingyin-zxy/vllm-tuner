from abc import ABC, abstractmethod


class InferenceFramework(ABC):
    """Abstract base for inference server frameworks (vLLM, LMDeploy, SGLang)."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @abstractmethod
    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build shell command to launch the inference server."""
        ...

    def get_api_base(self) -> str:
        """Return OpenAI-compatible API base URL."""
        return f"http://{self.host}:{self.port}/v1"

    @abstractmethod
    def get_health_endpoint(self) -> str:
        """Return health check URL."""
        ...

    @property
    def tunable_infra_params(self) -> dict:
        """Phase 2a tunable parameter ranges."""
        return {
            "block_size": [8, 16, 32],
            "gpu_memory_utilization": [0.75, 0.80, 0.85, 0.90],
            "max_num_seqs": [64, 128, 256],
        }
