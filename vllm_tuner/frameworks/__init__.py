from vllm_tuner.frameworks.vllm_framework import VLLMFramework
from vllm_tuner.frameworks.lmdeploy_framework import LMDeployFramework
from vllm_tuner.frameworks.sglang_framework import SGLangFramework

_REGISTRY = {
    "vllm": VLLMFramework,
    "lmdeploy": LMDeployFramework,
    "sglang": SGLangFramework,
}


def get_framework(name: str, host: str, port: int):
    """Factory: return InferenceFramework instance by framework name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown framework: {name}. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](host=host, port=port)
