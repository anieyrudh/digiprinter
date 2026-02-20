"""Weighted multi-component reward function for the RL environment."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from digiprinter.printer.state import PrinterState
from digiprinter.config import PrinterConfig, DEFAULT_CONFIG


@dataclass(frozen=True)
class RewardWeights:
    """Relative importance of each reward component.

    Weights should sum to ~1.0 (excluding the safety penalty which
    operates on a different scale).
    """
    adhesion: float = 0.25
    dimensional_accuracy: float = 0.20
    warping: float = 0.15
    stringing: float = 0.10
    speed: float = 0.10
    thermal_stability: float = 0.10
    energy_efficiency: float = 0.05
    safety: float = 0.05


class RewardCalculator:
    """Compute a shaped reward from the current printer state.

    Each component is scaled to roughly [-1, 1] (except the safety
    penalty which can be -10) and then combined with the configured
    :class:`RewardWeights`.
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        config: PrinterConfig | None = None,
    ) -> None:
        self.weights = weights or RewardWeights()
        self.config = config or DEFAULT_CONFIG
        self._prev_hotend_temp: float | None = None
        self._prev_bed_temp: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        state: PrinterState,
        action: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Return ``(total_reward, components_dict)``.

        Parameters
        ----------
        state:
            The current printer state after a simulation step.
        action:
            The action taken (reserved for future use; currently unused).
        """
        w = self.weights

        # --- Adhesion ---
        adhesion = state.adhesion_quality  # already 0-1

        # --- Dimensional accuracy ---
        dimensional_accuracy = max(0.0, 1.0 - state.dimensional_error * 5.0)

        # --- Warping ---
        warping = max(0.0, 1.0 - state.warping_amount * 2.0)

        # --- Stringing ---
        stringing = max(0.0, 1.0 - state.stringing_amount * 0.5)

        # --- Speed ---
        speed = min(
            state.current_speed / self.config.default_print_speed,
            1.5,
        )

        # --- Thermal stability ---
        if self._prev_hotend_temp is not None:
            hotend_osc = abs(state.hotend_temp - self._prev_hotend_temp)
            thermal_stability = max(0.0, 1.0 - hotend_osc * 0.5)
        else:
            thermal_stability = 1.0  # no history yet, assume stable
        self._prev_hotend_temp = state.hotend_temp
        self._prev_bed_temp = state.bed_temp

        # --- Energy efficiency ---
        max_power = self.config.hotend_heater_power + self.config.bed_heater_power
        energy_efficiency = max(
            0.0,
            1.0 - (state.hotend_power + state.bed_power) / max_power,
        )

        # --- Safety ---
        safety = -10.0 if state.fault else 0.0

        # --- Aggregate ---
        components = {
            "adhesion": adhesion,
            "dimensional_accuracy": dimensional_accuracy,
            "warping": warping,
            "stringing": stringing,
            "speed": speed,
            "thermal_stability": thermal_stability,
            "energy_efficiency": energy_efficiency,
            "safety": safety,
        }

        total = (
            w.adhesion * adhesion
            + w.dimensional_accuracy * dimensional_accuracy
            + w.warping * warping
            + w.stringing * stringing
            + w.speed * speed
            + w.thermal_stability * thermal_stability
            + w.energy_efficiency * energy_efficiency
            + w.safety * safety
        )

        return total, components

    def reset(self) -> None:
        """Clear internal state between episodes."""
        self._prev_hotend_temp = None
        self._prev_bed_temp = None
