from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.remote_env import RemoteEnv
from vllm_tuner.config import RemoteConfig


def make_config():
    return RemoteConfig(
        host="192.168.1.100", port=22, user="ubuntu",
        key_file="~/.ssh/id_rsa", working_dir="/workspace", vllm_port=8000
    )


@patch("vllm_tuner.remote_env.fabric")
def test_run_returns_stdout(mock_fabric):
    mock_conn = MagicMock()
    mock_conn.run.return_value = MagicMock(stdout="hello\n", stderr="", returncode=0)
    mock_fabric.Connection.return_value = mock_conn

    env = RemoteEnv(make_config())
    result = env.run("echo hello")
    assert result.stdout == "hello\n"


@patch("vllm_tuner.remote_env.fabric")
def test_run_background_uses_nohup(mock_fabric):
    mock_conn = MagicMock()
    mock_fabric.Connection.return_value = mock_conn

    env = RemoteEnv(make_config())
    env.run_background("python server.py", "/workspace/server.log")
    call_args = mock_conn.run.call_args[0][0]
    assert "nohup" in call_args
    assert "server.log" in call_args


@patch("vllm_tuner.remote_env.fabric")
def test_read_log_tail(mock_fabric):
    mock_conn = MagicMock()
    mock_conn.run.return_value = MagicMock(stdout="last line\n", stderr="")
    mock_fabric.Connection.return_value = mock_conn

    env = RemoteEnv(make_config())
    tail = env.read_log_tail("/workspace/vllm.log", lines=10)
    assert tail == "last line\n"
    cmd = mock_conn.run.call_args[0][0]
    assert "tail -10" in cmd
