# tests/test_shipit.py
from unittest.mock import MagicMock, patch, call, patch as mock_patch
import pytest
from vllm_tuner.shipit import ShipIt
from vllm_tuner.docker_manager import DockerManager, ExecResult
from pathlib import Path

def make_shipit():
    remote = MagicMock()
    actor = MagicMock()
    actor.fast_fail_check.return_value = True
    framework = MagicMock()
    framework.get_health_endpoint.return_value = "http://1.2.3.4:8000/health"
    cfg = MagicMock()
    cfg.remote.working_dir = "/workspace"
    cfg.remote.vllm_port = 8000
    cfg.remote.host = "1.2.3.4"
    cfg.hardware.npu_id = 4
    cfg.hardware.chip_id = 0
    cfg.model.hf_url = "https://huggingface.co/Qwen/Qwen2-7B"
    cfg.model.local_path = "/workspace/models/Qwen2-7B"
    cfg.model.hf_token = ""
    return ShipIt(remote=remote, actor=actor, framework=framework, config=cfg)

def test_check_env_detects_vllm_installed():
    ship = make_shipit()
    # Exactly 6 calls matching check_remote_env() order:
    # 1. npu-smi usages, 2. npu-smi health, 3. python -c 'import vllm',
    # 4. df, 5. lsof (port), 6. ls (model dir)
    ship.remote.run.side_effect = [
        MagicMock(stdout="HBM-Usage(MB): 1000 / 65536\n"),  # npu-smi usages
        MagicMock(stdout="OK\n"),                            # npu-smi health
        MagicMock(stdout="", returncode=0),                  # python -c 'import vllm'
        MagicMock(stdout="/dev/sda  400G  50G  350G\n"),     # df
        MagicMock(stdout=""),                                # lsof (port not occupied)
        MagicMock(stdout="exists\n"),                        # ls model dir
    ]
    env = ship.check_remote_env()
    assert env["hbm_total_mb"] == 65536
    assert env["health"] == "OK"
    assert env["model_exists"] is True

def test_pull_model_constructs_correct_command():
    ship = make_shipit()
    ship.remote.run.return_value = MagicMock(stdout="", returncode=0)
    ship.pull_model()
    calls = [str(c) for c in ship.remote.run.call_args_list]
    assert any("huggingface-cli download" in c for c in calls)
    assert any("Qwen/Qwen2-7B" in c for c in calls)
    assert any("/workspace/models/Qwen2-7B" in c for c in calls)

def test_port_cleanup_refuses_to_kill_sshd():
    ship = make_shipit()
    ship.remote.run.side_effect = [
        MagicMock(stdout="1234\n"),       # lsof pid
        MagicMock(stdout="sshd\n"),       # ps proc name
    ]
    with pytest.raises(RuntimeError, match="Refusing"):
        ship._clear_port(8000)

def test_self_heal_retries_up_to_3_times():
    ship = make_shipit()
    brain = MagicMock()
    brain.diagnose.return_value = {
        "fix_commands": ["pip install vllm-ascend --upgrade"],
        "diagnosis": "missing package",
        "adjusted_param": {},
    }
    # actor.start_service always fails (simulates persistent crash)
    ship.actor.start_service.side_effect = RuntimeError("OOM")
    ship.remote.read_log_tail = MagicMock(return_value="OOM in log")
    with pytest.raises(RuntimeError, match="self-heal"):
        ship.self_heal(error="OOM", attempt_history=[], brain=brain,
                       default_infra_config={})
    assert brain.diagnose.call_count == 3


def make_docker_shipit():
    remote = MagicMock()
    actor = MagicMock()
    actor.fast_fail_check.return_value = True
    framework = MagicMock()
    cfg = MagicMock()
    cfg.remote.working_dir = "/workspace"
    cfg.remote.vllm_port = 8000
    cfg.remote.host = "1.2.3.4"
    cfg.hardware.npu_id = 0
    cfg.hardware.chip_id = 0
    cfg.model.hf_url = "https://huggingface.co/Qwen/Qwen2-7B"
    cfg.model.local_path = "/workspace/models/Qwen2-7B"
    cfg.model.hf_token = ""
    docker_mgr = MagicMock(spec=DockerManager)
    docker_mgr.exec_run.return_value = ExecResult(stdout="", stderr="", returncode=0)
    ship = ShipIt(remote=remote, actor=actor, framework=framework, config=cfg,
                  docker_manager=docker_mgr)
    return ship, docker_mgr


def test_check_env_docker_uses_is_image_present():
    ship, docker_mgr = make_docker_shipit()
    docker_mgr.is_image_present.return_value = True
    # Docker branch: 5 remote.run calls (no "python -c 'import vllm'")
    ship.remote.run.side_effect = [
        MagicMock(stdout="HBM-Usage(MB): 1000 / 65536\n"),
        MagicMock(stdout="OK\n"),
        MagicMock(stdout="/dev/sda  400G  50G  350G\n"),
        MagicMock(stdout=""),           # lsof (port free)
        MagicMock(stdout="exists\n"),   # ls model
    ]
    env = ship.check_remote_env()
    assert env["vllm_installed"] is True
    docker_mgr.is_image_present.assert_called_once()
    # Must NOT call "python -c 'import vllm'" on host
    host_cmds = [str(c) for c in ship.remote.run.call_args_list]
    assert not any("import vllm" in c for c in host_cmds)


def test_auto_fix_env_docker_pulls_when_image_missing():
    ship, docker_mgr = make_docker_shipit()
    env = {"vllm_installed": False, "port_occupied": False}
    ship.auto_fix_env(env)
    docker_mgr.pull.assert_called_once()


def test_auto_fix_env_docker_port_uses_fuser():
    ship, docker_mgr = make_docker_shipit()
    env = {"vllm_installed": True, "port_occupied": True}
    ship.auto_fix_env(env)
    cmds = [str(c) for c in docker_mgr.exec_run.call_args_list]
    assert any("fuser" in c for c in cmds)


def test_run_docker_starts_container_when_not_running():
    ship, docker_mgr = make_docker_shipit()
    docker_mgr.is_container_running.return_value = False
    env = {
        "hbm_total_mb": 65536, "health": "OK",
        "vllm_installed": True, "disk_ok": True,
        "port_occupied": False, "port_pid": "", "model_exists": True,
    }
    with patch.object(ship, "check_remote_env", return_value=env), \
         patch.object(ship, "auto_fix_env"):
        ship.run({})
    docker_mgr.run_container.assert_called_once()


def test_run_docker_skips_container_start_when_already_running():
    ship, docker_mgr = make_docker_shipit()
    docker_mgr.is_container_running.return_value = True
    env = {
        "hbm_total_mb": 65536, "health": "OK",
        "vllm_installed": True, "disk_ok": True,
        "port_occupied": False, "port_pid": "", "model_exists": True,
    }
    with patch.object(ship, "check_remote_env", return_value=env), \
         patch.object(ship, "auto_fix_env"):
        ship.run({})
    docker_mgr.run_container.assert_not_called()


def test_backward_compat_shipit_no_docker():
    # Without docker_manager, check_remote_env still calls "python -c 'import vllm'"
    ship = make_shipit()  # existing helper — no docker_manager
    ship.remote.run.side_effect = [
        MagicMock(stdout="HBM-Usage(MB): 1000 / 65536\n"),
        MagicMock(stdout="OK\n"),
        MagicMock(stdout="", returncode=0),   # python -c 'import vllm'
        MagicMock(stdout="/dev/sda  400G\n"),
        MagicMock(stdout=""),
        MagicMock(stdout="exists\n"),
    ]
    env = ship.check_remote_env()
    host_cmds = [str(c) for c in ship.remote.run.call_args_list]
    assert any("import vllm" in c for c in host_cmds)
