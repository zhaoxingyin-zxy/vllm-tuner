# tests/test_docker_manager.py
from unittest.mock import MagicMock, call
import pytest
from vllm_tuner.docker_manager import DockerManager, ExecResult
from vllm_tuner.config import DockerConfig, HardwareConfig


def make_mgr(hw_type="ascend", npu_id=0, chip_id=0, device_index=-1, registry=""):
    remote = MagicMock()
    remote.run.return_value = MagicMock(stdout="", stderr="", returncode=0)
    docker_cfg = DockerConfig(
        image="vllm-ascend:latest",
        container_name="vllm_tuner",
        shm_size="8g",
        registry=registry,
        extra_flags="",
        device_index=device_index,
    )
    hw_cfg = HardwareConfig(type=hw_type, npu_id=npu_id, chip_id=chip_id)
    return DockerManager(remote=remote, docker_cfg=docker_cfg, hw_cfg=hw_cfg), remote


def test_is_image_present_true():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="abc123\n", returncode=0)
    assert mgr.is_image_present() is True


def test_is_image_present_false():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="", returncode=0)
    assert mgr.is_image_present() is False


def test_is_container_running_true():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="deadbeef\n", returncode=0)
    assert mgr.is_container_running() is True


def test_is_container_running_false():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="", returncode=0)
    assert mgr.is_container_running() is False


def test_pull_without_registry():
    mgr, remote = make_mgr()
    mgr.pull()
    cmds = [str(c) for c in remote.run.call_args_list]
    assert not any("docker login" in c for c in cmds)
    assert any("docker pull vllm-ascend:latest" in c for c in cmds)


def test_pull_with_registry_calls_login_first():
    mgr, remote = make_mgr(registry="registry.example.com")
    mgr.pull()
    cmds = [str(c) for c in remote.run.call_args_list]
    assert any("docker login registry.example.com" in c for c in cmds)
    assert any("docker pull" in c for c in cmds)
    # login must come before pull
    login_idx = next(i for i, c in enumerate(cmds) if "docker login" in c)
    pull_idx = next(i for i, c in enumerate(cmds) if "docker pull" in c)
    assert login_idx < pull_idx


def test_run_container_removes_stale_then_runs(monkeypatch):
    mgr, remote = make_mgr()
    # Simulate stale container exists: docker ps -a returns a line
    remote.run.side_effect = [
        MagicMock(stdout="old_id\n", returncode=0),  # docker ps -a (stale check)
        MagicMock(stdout="", returncode=0),           # docker rm -f
        MagicMock(stdout="new_id\n", returncode=0),  # docker run
    ]
    mgr.run_container("/models", 8000)
    cmds = [str(c) for c in remote.run.call_args_list]
    assert any("docker rm -f" in c for c in cmds)
    assert any("docker run" in c for c in cmds)
    assert any("sleep infinity" in c for c in cmds)
    assert any("-p 8000:8000" in c for c in cmds)
    assert any("-v /models:/models:ro" in c for c in cmds)


def test_run_container_no_stale():
    mgr, remote = make_mgr()
    remote.run.side_effect = [
        MagicMock(stdout="", returncode=0),          # docker ps -a (no stale)
        MagicMock(stdout="new_id\n", returncode=0),  # docker run
    ]
    mgr.run_container("/models", 8000)
    cmds = [str(c) for c in remote.run.call_args_list]
    assert not any("docker rm -f" in c for c in cmds)
    assert any("docker run" in c for c in cmds)


def test_device_flags_ascend_auto():
    mgr, _ = make_mgr(hw_type="ascend", npu_id=1, chip_id=0, device_index=-1)
    flags = mgr._device_flags()
    # device_index = 1*2+0 = 2
    assert "--device /dev/davinci2" in flags
    assert "--device /dev/davinci_manager" in flags
    assert "--device /dev/hisi_hdc" in flags


def test_device_flags_ascend_explicit():
    mgr, _ = make_mgr(hw_type="ascend", npu_id=1, chip_id=0, device_index=0)
    flags = mgr._device_flags()
    assert "--device /dev/davinci0" in flags


def test_device_flags_cuda():
    mgr, _ = make_mgr(hw_type="cuda", npu_id=2)
    flags = mgr._device_flags()
    assert '--gpus "device=2"' in flags


def test_exec_run_returns_exec_result():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="hello\n", stderr="", returncode=0)
    result = mgr.exec_run("echo hello")
    assert isinstance(result, ExecResult)
    assert result.stdout == "hello\n"
    assert result.returncode == 0
    cmd_str = str(remote.run.call_args)
    assert "docker exec" in cmd_str
    assert "vllm_tuner" in cmd_str


def test_exec_background_calls_docker_exec():
    mgr, remote = make_mgr()
    mgr.exec_background("python -m vllm.entrypoints.openai.api_server", "/workspace/vllm.log")
    cmd_str = str(remote.run.call_args)
    assert "docker exec" in cmd_str
    assert "nohup" in cmd_str
    assert "/workspace/vllm.log" in cmd_str


def test_stop_container_suppresses_errors():
    mgr, remote = make_mgr()
    remote.run.return_value = MagicMock(stdout="", returncode=1)  # container not found
    mgr.stop_container()  # must not raise
    cmds = [str(c) for c in remote.run.call_args_list]
    assert any("docker stop" in c for c in cmds)
    assert any("docker rm" in c for c in cmds)
