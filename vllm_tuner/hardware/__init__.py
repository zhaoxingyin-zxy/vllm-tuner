from vllm_tuner.hardware.ascend import AscendObserver
from vllm_tuner.hardware.cuda import CUDAObserver

def get_observer(hw_type: str, remote, npu_id: int = 0, chip_id: int = 0):
    if hw_type == "ascend":
        return AscendObserver(remote, npu_id=npu_id, chip_id=chip_id)
    if hw_type == "cuda":
        return CUDAObserver(remote, device_id=npu_id)
    raise ValueError(f"Unknown hardware type: {hw_type}")
