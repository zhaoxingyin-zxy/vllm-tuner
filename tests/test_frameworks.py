import pytest
from vllm_tuner.frameworks.vllm_framework import VLLMFramework
from vllm_tuner.frameworks.lmdeploy_framework import LMDeployFramework
from vllm_tuner.frameworks.sglang_framework import SGLangFramework
from vllm_tuner.frameworks import get_framework


def test_vllm_start_cmd_contains_model_and_params():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    config = {"block_size": 16, "gpu_memory_utilization": 0.85, "max_num_seqs": 128}
    cmd = fw.build_start_cmd("/models/Qwen2-7B", config)
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model /models/Qwen2-7B" in cmd
    assert "--block-size 16" in cmd
    assert "--gpu-memory-utilization 0.85" in cmd
    assert "--max-num-seqs 128" in cmd


def test_vllm_health_endpoint():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    assert fw.get_health_endpoint() == "http://1.2.3.4:8000/health"


def test_vllm_api_base():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    assert fw.get_api_base() == "http://1.2.3.4:8000/v1"


def test_lmdeploy_start_cmd():
    fw = LMDeployFramework(host="1.2.3.4", port=8000)
    cmd = fw.build_start_cmd("/models/Qwen2-7B", {"cache_max_entry_count": 0.8})
    assert "lmdeploy serve api_server" in cmd
    assert "/models/Qwen2-7B" in cmd


def test_lmdeploy_health_endpoint():
    fw = LMDeployFramework(host="1.2.3.4", port=8000)
    assert fw.get_health_endpoint() == "http://1.2.3.4:8000/v1/models"


def test_sglang_start_cmd():
    fw = SGLangFramework(host="1.2.3.4", port=8000)
    cmd = fw.build_start_cmd("/models/Qwen2-7B", {"mem_fraction_static": 0.85})
    assert "sglang.launch_server" in cmd
    assert "--model-path /models/Qwen2-7B" in cmd


def test_framework_factory():
    fw = get_framework("vllm", host="1.2.3.4", port=8000)
    assert isinstance(fw, VLLMFramework)
    fw2 = get_framework("lmdeploy", host="1.2.3.4", port=8000)
    assert isinstance(fw2, LMDeployFramework)


def test_vllm_tunable_infra_params():
    fw = VLLMFramework(host="1.2.3.4", port=8000)
    params = fw.tunable_infra_params
    assert "block_size" in params
    assert "gpu_memory_utilization" in params
    assert "max_num_seqs" in params
    assert 8 in params["block_size"]
