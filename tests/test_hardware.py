# tests/test_hardware.py
from unittest.mock import MagicMock
from vllm_tuner.hardware.ascend import AscendObserver
from vllm_tuner.hardware import get_observer

def make_remote(outputs: dict):
    """Helper: RemoteEnv mock that returns preset stdout per command substring."""
    remote = MagicMock()
    def run_side_effect(cmd, **kwargs):
        for key, stdout in outputs.items():
            if key in cmd:
                return MagicMock(stdout=stdout, returncode=0)
        return MagicMock(stdout="", returncode=0)
    remote.run.side_effect = run_side_effect
    return remote

def test_ascend_parse_hbm():
    remote = make_remote({
        "usages": "HBM-Usage(MB): 3161 / 65536\n",
        "common": "AICore(%): 87\n",
        "power":  "Power(W): 93.6\n",
        "temp":   "Temp(C): 40\n",
        "health": "OK\n",
    })
    obs = AscendObserver(remote, npu_id=4, chip_id=0)
    stats = obs.get_stats()
    assert stats.hbm_used_mb == 3161
    assert stats.hbm_total_mb == 65536
    assert abs(stats.hbm_util_pct - 4.8) < 0.1
    assert stats.aicore_util_pct == 87.0
    assert stats.power_w == 93.6
    assert stats.temp_c == 40.0
    assert stats.health == "OK"

def test_ascend_is_healthy_ok():
    remote = make_remote({"health": "OK\n", "usages": "HBM-Usage(MB): 1000 / 65536\n",
                           "common": "AICore(%): 50\n", "power": "Power(W): 80\n",
                           "temp": "Temp(C): 60\n"})
    obs = AscendObserver(remote, npu_id=4, chip_id=0)
    assert obs.is_healthy() is True

def test_ascend_is_healthy_warning():
    remote = make_remote({"health": "Warning\n", "usages": "HBM-Usage(MB): 1000 / 65536\n",
                           "common": "AICore(%): 50\n", "power": "Power(W): 80\n",
                           "temp": "Temp(C): 60\n"})
    obs = AscendObserver(remote, npu_id=4, chip_id=0)
    assert obs.is_healthy() is False

def test_get_observer_factory():
    remote = MagicMock()
    obs = get_observer("ascend", remote, npu_id=0, chip_id=0)
    assert isinstance(obs, AscendObserver)
