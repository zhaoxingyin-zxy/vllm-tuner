from vllm_tuner.frameworks.base import InferenceFramework


class LMDeployFramework(InferenceFramework):
    """LMDeploy — Ascend ecosystem alternative."""

    def build_start_cmd(self, model_weights: str, config: dict) -> str:
        """Build lmdeploy serve api_server launch command."""
        cmd = f"lmdeploy serve api_server {model_weights} --server-port {self.port}"
        if "cache_max_entry_count" in config:
            cmd += f" --cache-max-entry-count {config['cache_max_entry_count']}"
        return cmd

    def get_health_endpoint(self) -> str:
        """Return http://host:port/v1/models."""
        return f"http://{self.host}:{self.port}/v1/models"
