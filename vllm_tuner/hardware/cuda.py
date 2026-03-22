import re
from vllm_tuner.hardware.base import HardwareObserver, HardwareStats

class CUDAObserver(HardwareObserver):
    def __init__(self, remote, device_id: int = 0):
        self.remote = remote
        self.device_id = device_id

    def get_stats(self) -> HardwareStats:
        out = self.remote.run(
            f"nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,"
            f"power.draw,temperature.gpu --format=csv,noheader,nounits "
            f"--id={self.device_id}"
        ).stdout
        parts = [p.strip() for p in out.split(",")]
        used, total, util, power, temp = (float(p) for p in parts)
        pct = round(used / total * 100, 1)
        return HardwareStats(
            hbm_used_mb=used, hbm_total_mb=total, hbm_util_pct=pct,
            aicore_util_pct=util, power_w=power, temp_c=temp, health="OK"
        )
