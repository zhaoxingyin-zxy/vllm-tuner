import re
from vllm_tuner.hardware.base import HardwareObserver, HardwareStats


class CUDAObserver(HardwareObserver):
    """CUDA GPU observer using nvidia-smi."""

    def __init__(self, remote, device_id: int = 0):
        pass

    def get_stats(self) -> HardwareStats:
        """Collect VRAM, utilization, power, temp via nvidia-smi."""
        raise NotImplementedError
