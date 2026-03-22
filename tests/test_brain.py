# tests/test_brain.py
import json
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.brain import Brain

def make_brain():
    return Brain(api_key="test-key", model="claude-sonnet-4-6")

def mock_claude_response(content: dict):
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(content))]
    return msg

def test_decide_next_config_returns_dict():
    brain = make_brain()
    with patch("vllm_tuner.brain.anthropic.Anthropic") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        client.messages.create.return_value = mock_claude_response({
            "optimization_focus": "speed",
            "next_config": {"block_size": 8, "gpu_memory_utilization": 0.85},
            "reasoning": "KV碎片高",
            "confidence": 0.8,
            "skip_reason": None
        })
        brain = Brain(api_key="test-key", model="claude-sonnet-4-6")
        result = brain.decide_next_config(
            sweep_result={}, history=[], hw_stats=None,
            phase="2a", param_space={"block_size": [8, 16]},
            seen_hashes=set()
        )
    assert "next_config" in result
    assert "reasoning" in result

def test_diagnose_returns_fix_commands():
    brain = make_brain()
    with patch("vllm_tuner.brain.anthropic.Anthropic") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        client.messages.create.return_value = mock_claude_response({
            "diagnosis": "OOM due to large block_size",
            "fix_commands": ["pip install vllm-ascend --upgrade"],
            "adjusted_param": {"gpu_memory_utilization": 0.80}
        })
        brain = Brain(api_key="test-key", model="claude-sonnet-4-6")
        result = brain.diagnose(
            logs="CANN OOM error...", error="OOM", attempt_history=[]
        )
    assert "fix_commands" in result
    assert isinstance(result["fix_commands"], list)
