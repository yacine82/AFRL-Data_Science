from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Protocol, Dict

# Protocol so main.py can type-check any future SPO model with the same interface
class FuelModel(Protocol):
    def climb(self, w0_lb: float, dh_ft: float, dx_nm: float, tas_kt: float, isa_dev_c: float) -> Tuple[float, float]:
        """Return fuel_used_lb, time_s."""
    def cruise(self, w0_lb: float, dx_nm: float, tas_kt: float, isa_dev_c: float, alt_ft: float) -> Tuple[float, float]:
        """Return fuel_used_lb, time_s."""
    def descent(self, w0_lb: float, dh_ft: float, dx_nm: float, tas_kt: float, isa_dev_c: float) -> Tuple[float, float]:
        """Return fuel_used_lb, time_s."""

@dataclass
class MockFuelModel:
    # Simple tunables to simulate behavior
    climb_lb_per_kft: float = 220.0
    climb_lb_per_nm: float = 5.0
    cruise_lb_per_nm_at_sl: float = 2.8
    cruise_alt_factor_per_kft: float = -0.03  # higher alt slightly more efficient
    descent_lb_per_kft: float = 40.0
    descent_lb_per_nm: float = 1.0
    min_tas_kt: float = 220.0
    max_tas_kt: float = 460.0

    def _leg_time_s(self, dx_nm: float, tas_kt: float) -> float:
        tas = max(self.min_tas_kt, min(self.max_tas_kt, tas_kt))
        return 3600.0 * max(0.0, dx_nm) / max(tas, 1e-6)

    def climb(self, w0_lb: float, dh_ft: float, dx_nm: float, tas_kt: float, isa_dev_c: float) -> Tuple[float, float]:
        kft = max(0.0, dh_ft) / 1000.0
        fuel = self.climb_lb_per_kft * kft + self.climb_lb_per_nm * max(0.0, dx_nm)
        return fuel, self._leg_time_s(dx_nm, tas_kt)

    def cruise(self, w0_lb: float, dx_nm: float, tas_kt: float, isa_dev_c: float, alt_ft: float) -> Tuple[float, float]:
        kft = max(0.0, alt_ft) / 1000.0
        per_nm = max(0.8, self.cruise_lb_per_nm_at_sl * (1.0 + self.cruise_alt_factor_per_kft * kft))
        fuel = per_nm * max(0.0, dx_nm)
        return fuel, self._leg_time_s(dx_nm, tas_kt)

    def descent(self, w0_lb: float, dh_ft: float, dx_nm: float, tas_kt: float, isa_dev_c: float) -> Tuple[float, float]:
        kft = max(0.0, -dh_ft) / 1000.0  # negative dh for descent
        fuel = self.descent_lb_per_kft * kft + self.descent_lb_per_nm * max(0.0, dx_nm)
        return fuel, self._leg_time_s(dx_nm, tas_kt)

# Swappable mock catalog. Later, replace values or keys with SPO models.
MOCK_MODELS: Dict[str, FuelModel] = {
    "mock_v1": MockFuelModel(),
    "mock_econ": MockFuelModel(cruise_lb_per_nm_at_sl=2.4, cruise_alt_factor_per_kft=-0.035, max_tas_kt=440.0),
    "mock_fast": MockFuelModel(climb_lb_per_kft=260.0, cruise_lb_per_nm_at_sl=3.2, max_tas_kt=500.0),
}
