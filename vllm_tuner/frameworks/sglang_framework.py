from vllm_tuner.frameworks.base import InferenceFramework


class SGLangFramework(InferenceFramework):
    """SGLang — CUDA scene."""

    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build sglang.launch_server launch command."""
        cmd = f"python -m sglang.launch_server --model-path {model_weights} --port {self.port}"
        if "mem_fraction_static" in config:
            cmd += f" --mem-fraction-static {config['mem_fraction_static']}"
        return cmd

    def get_health_endpoint(self) -> str:
        """Return http://host:port/health."""
        return f"http://{self.host}:{self.port}/health"
