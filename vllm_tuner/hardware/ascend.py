import re
from vllm_tuner.hardware.base import HardwareObserver, HardwareStats


class AscendObserver(HardwareObserver):
    """Ascend NPU observer using npu-smi single-snapshot commands (no watch)."""

    def __init__(self, remote, npu_id: int, chip_id: int):
        pass

    def get_stats(self) -> HardwareStats:
        """Collect HBM, AICore, power, temp, health via npu-smi snapshots."""
        raise NotImplementedError

    def _run(self, flag: str) -> str:
        """Run npu-smi info -t <flag> -i <npu_id> -c <chip_id>."""
        raise NotImplementedError

    def _run_health(self) -> str:
        """Run npu-smi info -t health -i <npu_id>."""
        raise NotImplementedError
