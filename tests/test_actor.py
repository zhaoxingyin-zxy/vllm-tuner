# tests/test_actor.py
from unittest.mock import MagicMock, patch
import pytest
from vllm_tuner.actor import Actor
from vllm_tuner.docker_manager import ExecResult

def make_actor():
    remote = MagicMock()
    framework = MagicMock()
    framework.build_start_cmd.return_value = "python -m vllm.entrypoints.openai.api_server"
    framework.get_health_endpoint.return_value = "http://1.2.3.4:8000/health"
    return Actor(remote=remote, framework=framework, work_dir="/workspace",
                 host="1.2.3.4", port=8000)

def test_fast_fail_passes_when_healthy():
    actor = make_actor()
    with patch("vllm_tuner.actor.requests.post") as mock_post, \
         patch("vllm_tuner.actor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "hi"}}]}
        )
        assert actor.fast_fail_check() is True

def test_fast_fail_fails_when_health_down():
    actor = make_actor()
    with patch("vllm_tuner.actor.requests.get") as mock_get:
        mock_get.side_effect = Exception("connection refused")
        assert actor.fast_fail_check() is False

def test_stop_service_kills_named_process():
    actor = make_actor()
    # Four separate calls: lsof (PID), ps (process name), kill -15, ps (check after kill)
    actor.remote.run.side_effect = [
        MagicMock(stdout="12345\n", returncode=0),   # lsof → PID only
        MagicMock(stdout="vllm\n", returncode=0),    # ps → process name
        MagicMock(stdout="", returncode=0),          # kill -15
        MagicMock(stdout="", returncode=0),          # ps check after kill (dead)
    ]
    actor.stop_service()
    calls = [str(c) for c in actor.remote.run.call_args_list]
    assert any("kill" in c for c in calls)


def make_docker_actor():
    from vllm_tuner.docker_manager import DockerManager
    remote = MagicMock()
    framework = MagicMock()
    framework.build_start_cmd.return_value = "python -m vllm.entrypoints.openai.api_server"
    framework.get_health_endpoint.return_value = "http://1.2.3.4:8000/health"
    docker_mgr = MagicMock(spec=DockerManager)
    docker_mgr.exec_run.return_value = ExecResult(stdout="", stderr="", returncode=0)
    return Actor(remote=remote, framework=framework, work_dir="/workspace",
                 host="1.2.3.4", port=8000, docker_manager=docker_mgr), docker_mgr


def test_stop_service_docker_uses_pkill():
    actor, docker_mgr = make_docker_actor()
    # First exec_run: pkill (process gone)
    # Second exec_run: pgrep (empty = process dead)
    docker_mgr.exec_run.side_effect = [
        ExecResult(stdout="", stderr="", returncode=0),   # pkill -f
        ExecResult(stdout="", stderr="", returncode=1),   # pgrep (not found = dead)
    ]
    actor.stop_service()
    cmds = [str(c) for c in docker_mgr.exec_run.call_args_list]
    assert any("pkill" in c for c in cmds)
    # remote.run must NOT be called (no lsof on host)
    actor.remote.run.assert_not_called()


def test_start_service_docker_uses_exec_background():
    actor, docker_mgr = make_docker_actor()
    with patch("vllm_tuner.actor.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        actor.start_service("/models/Qwen", {})
    docker_mgr.exec_background.assert_called_once()
    actor.remote.run_background.assert_not_called()


def test_backward_compat_no_docker_manager():
    # Without docker_manager, must still work as before (bare lsof+kill)
    actor = make_actor()  # existing helper — no docker_manager
    actor.remote.run.side_effect = [
        MagicMock(stdout="12345\n", returncode=0),   # lsof → PID
        MagicMock(stdout="vllm\n", returncode=0),    # ps → process name
        MagicMock(stdout="", returncode=0),           # kill -15
        MagicMock(stdout="", returncode=0),           # ps check after kill
    ]
    actor.stop_service()
    calls = [str(c) for c in actor.remote.run.call_args_list]
    assert any("lsof" in c for c in calls)
