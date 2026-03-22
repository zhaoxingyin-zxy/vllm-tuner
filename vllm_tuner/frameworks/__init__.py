from vllm_tuner.frameworks.vllm_framework import VLLMFramework
from vllm_tuner.frameworks.lmdeploy_framework import LMDeployFramework
from vllm_tuner.frameworks.sglang_framework import SGLangFramework


def get_framework(name: str, host: str, port: int):
    """Factory: return InferenceFramework instance by framework name."""
    raise NotImplementedError
