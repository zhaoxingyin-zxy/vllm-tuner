# tests/test_config_docker.py
from vllm_tuner.config import DockerConfig, TunerConfig
import pytest

def test_docker_config_defaults():
    dc = DockerConfig(image="vllm:latest", container_name="vllm_tuner")
    assert dc.shm_size == "8g"
    assert dc.registry == ""
    assert dc.extra_flags == ""
    assert dc.device_index == -1

def test_tuner_config_docker_is_optional():
    # TunerConfig.docker defaults to None — no docker block needed
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(TunerConfig)}
    assert "docker" in fields
    assert fields["docker"].default is None
