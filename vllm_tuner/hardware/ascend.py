import re
from vllm_tuner.hardware.base import HardwareObserver, HardwareStats

class AscendObserver(HardwareObserver):
    def __init__(self, remote, npu_id: int, chip_id: int):
        self.remote = remote
        self.npu_id = npu_id
        self.chip_id = chip_id

    def _run(self, flag: str) -> str:
        result = self.remote.run(
            f"npu-smi info -t {flag} -i {self.npu_id} -c {self.chip_id}"
        )
        return result.stdout

    def _run_health(self) -> str:
        result = self.remote.run(
            f"npu-smi info -t health -i {self.npu_id}"
        )
        return result.stdout

    def get_stats(self) -> HardwareStats:
        usages_out = self._run("usages")
        common_out = self._run("common")
        power_out  = self._run("power")
        temp_out   = self._run("temp")
        health_out = self._run_health()

        # Parse HBM: "HBM-Usage(MB): 3161 / 65536"
        m = re.search(r"HBM-Usage\(MB\):\s*(\d+)\s*/\s*(\d+)", usages_out)
        hbm_used = float(m.group(1)) if m else 0.0
        hbm_total = float(m.group(2)) if m else 1.0
        hbm_pct = round(hbm_used / hbm_total * 100, 1)

        # Parse AICore: "AICore(%): 87"
        m = re.search(r"AICore\(%\):\s*(\d+)", common_out)
        aicore = float(m.group(1)) if m else 0.0

        # Parse Power: "Power(W): 93.6"
        m = re.search(r"Power\(W\):\s*([\d.]+)", power_out)
        power = float(m.group(1)) if m else 0.0

        # Parse Temp: "Temp(C): 40"
        m = re.search(r"Temp\(C\):\s*([\d.]+)", temp_out)
        temp = float(m.group(1)) if m else 0.0

        health = health_out.strip().split()[0] if health_out.strip() else "UNKNOWN"

        return HardwareStats(
            hbm_used_mb=hbm_used,
            hbm_total_mb=hbm_total,
            hbm_util_pct=hbm_pct,
            aicore_util_pct=aicore,
            power_w=power,
            temp_c=temp,
            health=health,
        )
