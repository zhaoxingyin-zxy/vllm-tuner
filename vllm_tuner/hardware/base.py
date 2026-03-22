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
    @abstractmethod
    def get_stats(self) -> HardwareStats: ...

    def is_healthy(self) -> bool:
        return self.get_stats().health == "OK"
