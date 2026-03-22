import pytest
from unittest.mock import MagicMock, patch
from vllm_tuner.skills.throughput import ThroughputSkill
from vllm_tuner.skills.latency import LatencySkill
from vllm_tuner.skills.task_metric import TaskMetricSkill
from vllm_tuner.skills.memory import MemorySkill


def test_throughput_returns_tokens_per_second():
    pass


def test_latency_returns_p99():
    pass


def test_task_metric_parses_json_stdout():
    pass


def test_task_metric_raises_on_bad_stdout():
    pass


def test_memory_skill_returns_hbm_pct():
    pass
