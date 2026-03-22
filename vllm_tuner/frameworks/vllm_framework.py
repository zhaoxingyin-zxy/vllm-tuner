from vllm_tuner.frameworks.base import InferenceFramework


class VLLMFramework(InferenceFramework):
    """vLLM on Ascend NPU via vllm-ascend plugin."""

    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build vllm.entrypoints.openai.api_server launch command."""
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server"
            f" --model {model_weights}"
            f" --port {self.port}"
        )
        if "block_size" in config:
            cmd += f" --block-size {config['block_size']}"
        if "gpu_memory_utilization" in config:
            cmd += f" --gpu-memory-utilization {config['gpu_memory_utilization']}"
        if "max_num_seqs" in config:
            cmd += f" --max-num-seqs {config['max_num_seqs']}"
        return cmd

    def get_health_endpoint(self) -> str:
        """Return http://host:port/health."""
        return f"http://{self.host}:{self.port}/health"

    @property
    def tunable_infra_params(self) -> dict:
        """vLLM-specific Phase 2a tunable ranges."""
        return {
            "block_size": [8, 16, 32],
            "gpu_memory_utilization": [0.75, 0.80, 0.85, 0.90],
            "max_num_seqs": [64, 128, 256],
        }
