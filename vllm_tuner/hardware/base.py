from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class HardwareStats:
    hbm_used_mb: float
    hbm_total_mb: float
    hbm_util_pct: float
    aicore_util_pct: float
    power_w: float
    temp_c: float
    health: str  # "OK" / "Warning" / "Alarm" / "Critical"


class HardwareObserver(ABC):
    """Abstract hardware observer for NPU/GPU stats."""

    @abstractmethod
    def get_stats(self) -> HardwareStats:
        """Snapshot current hardware metrics."""
        ...

    def is_healthy(self) -> bool:
        """Return True if health == OK."""
        raise NotImplementedError
