from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MaterialProperties:
    """Thermoplastic material properties for FDM/FFF 3-D printing simulation.

    All temperatures are in degrees Celsius unless otherwise noted.
    SI units are used throughout (kg, m, s, J, W, Pa).
    """

    # ── identity ────────────────────────────────────────────────────────
    name: str

    # ── bulk thermal properties ─────────────────────────────────────────
    density: float                  # kg/m³
    specific_heat: float            # J/(kg·K)
    thermal_conductivity: float     # W/(m·K)
    thermal_diffusivity: float      # m²/s

    # ── transition temperatures ─────────────────────────────────────────
    glass_transition_temp: float    # °C
    melt_temp: float                # °C
    crystallization_temp: Optional[float] = None  # °C (amorphous materials lack this)

    # ── recommended processing window ───────────────────────────────────
    nozzle_temp_range: Tuple[float, float] = (190.0, 220.0)  # (min, max) °C
    bed_temp_range: Tuple[float, float] = (50.0, 70.0)       # (min, max) °C

    # ── Cross-WLF viscosity model parameters ────────────────────────────
    #   η(T, γ̇) = η₀(T) / (1 + (η₀(T)·γ̇ / τ*)^(1-n))
    #   with  η₀(T) = D1 · exp(−A1·(T − T*) / (A2 + T − T*))
    d1: float = 1.0e12             # Pa·s  (reference viscosity prefactor)
    a1: float = 20.0               # –     (WLF coefficient 1)
    a2: float = 51.6               # °C    (WLF coefficient 2)
    t_star: float = 100.0          # °C    (reference temperature)
    tau_star: float = 25000.0      # Pa    (critical shear stress)
    n: float = 0.3                 # –     (power-law index)

    # ── shrinkage / warping ─────────────────────────────────────────────
    shrinkage_factor: float = 0.003        # linear shrinkage (dimensionless)
    warp_coefficient: float = 0.15         # propensity to warp (dimensionless)

    # ── adhesion ────────────────────────────────────────────────────────
    adhesion_coefficient: float = 1.0e6    # Pa (reference adhesion strength)
    adhesion_activation_energy: float = 50000.0  # J/mol

    # ── retraction / ooze ───────────────────────────────────────────────
    retraction_sensitivity: float = 0.8    # 0-1  (higher → more sensitive)
    ooze_coefficient: float = 0.3          # dimensionless ooze tendency

    # ── cooling ─────────────────────────────────────────────────────────
    fan_speed_default: float = 1.0         # 0-1  (fraction of max fan speed)

    # ── derived / computed properties ───────────────────────────────────
    @property
    def filament_cross_section_area(self) -> float:
        """Cross-sectional area of a standard 1.75 mm filament (m²)."""
        diameter = 1.75e-3  # 1.75 mm in metres
        return math.pi * (diameter / 2.0) ** 2
