from vllm_tuner.hardware.ascend import AscendObserver
from vllm_tuner.hardware.cuda import CUDAObserver


def get_observer(hw_type: str, remote, npu_id: int = 0, chip_id: int = 0):
    """Factory: return HardwareObserver by hardware type (ascend/cuda)."""
    raise NotImplementedError
