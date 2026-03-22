import pytest
from vllm_tuner.config import TunerConfig, load_config


MINIMAL_YAML = """
remote:
  host: 192.168.1.100
  port: 22
  user: ubuntu
  key_file: ~/.ssh/id_rsa
  working_dir: /home/ubuntu/workspace
  vllm_port: 8000
hardware:
  type: ascend
  npu_id: 4
  chip_id: 0
framework: vllm
model:
  hf_url: https://huggingface.co/Qwen/Qwen2-7B
  hf_token: ""
evaluation:
  script: /evals/run_mmlu.py
  data: /datasets/mmlu_sample/
  metric: accuracy
  metric_direction: maximize
  sample_size: 50
  timeout_seconds: 600
  skills: [throughput, latency, task_metric, memory]
sweep:
  concurrency_levels: [1, 2, 4, 8]
  input_lengths: [128, 256, 512]
  requests_per_cell: 10
  request_timeout_seconds: 30
optimization:
  phase_2a:
    max_rounds: 15
    patience: 3
    parameters:
      block_size: [8, 16, 32]
      gpu_memory_utilization: [0.75, 0.80, 0.85, 0.90]
      max_num_seqs: [64, 128, 256]
  phase_2b:
    max_rounds: 15
    patience: 3
    baseline:
      temperature: 1.0
      top_p: 1.0
      top_k: -1
      repetition_penalty: 1.0
    parameters:
      temperature: [0.6, 0.7, 0.8, 1.0]
      top_p: [0.85, 0.90, 0.95, 1.0]
      top_k: [-1, 20, 50, 100]
      repetition_penalty: [1.0, 1.05, 1.1]
save_dir: ./results/
"""


def test_load_config_remote(tmp_path):
    cfg_file = tmp_path / "tuner_config.yaml"
    cfg_file.write_text(MINIMAL_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.remote.host == "192.168.1.100"
    assert cfg.remote.vllm_port == 8000
    assert cfg.hardware.type == "ascend"
    assert cfg.hardware.npu_id == 4
    assert cfg.framework == "vllm"
    assert cfg.model.hf_url == "https://huggingface.co/Qwen/Qwen2-7B"
    assert cfg.model.local_path == "/home/ubuntu/workspace/models/Qwen2-7B"
    assert cfg.evaluation.metric_direction == "maximize"
    assert cfg.evaluation.sample_size == 50
    assert cfg.optimization.phase_2a.patience == 3
    assert cfg.optimization.phase_2b.baseline["temperature"] == 1.0
    assert 8 in cfg.optimization.phase_2a.parameters["block_size"]


def test_local_path_derivation(tmp_path):
    yaml_content = MINIMAL_YAML.replace(
        "hf_url: https://huggingface.co/Qwen/Qwen2-7B",
        "hf_url: https://huggingface.co/Qwen/Qwen2-7B-Instruct",
    ).replace(
        "working_dir: /home/ubuntu/workspace",
        "working_dir: /workspace",
    )
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.model.local_path == "/workspace/models/Qwen2-7B-Instruct"


def test_phase_2b_baseline_loaded(tmp_path):
    cfg_file = tmp_path / "tuner_config.yaml"
    cfg_file.write_text(MINIMAL_YAML)
    cfg = load_config(str(cfg_file))
    baseline = cfg.optimization.phase_2b.baseline
    assert baseline["temperature"] == 1.0
    assert baseline["top_k"] == -1
    assert baseline["repetition_penalty"] == 1.0


def test_evaluation_skills_list(tmp_path):
    cfg_file = tmp_path / "tuner_config.yaml"
    cfg_file.write_text(MINIMAL_YAML)
    cfg = load_config(str(cfg_file))
    assert "throughput" in cfg.evaluation.skills
    assert "task_metric" in cfg.evaluation.skills
