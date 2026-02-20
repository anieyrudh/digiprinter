"""Adaptive time stepping for the simulation engine."""

from __future__ import annotations

import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG


class AdaptiveTimeStepper:
    """Dynamically adjusts the simulation time step based on the current
    thermal and kinematic state of the printer.

    When things change quickly (high temperature rates, fast motion, or
    dangerously high temperatures), the stepper shrinks ``dt`` for accuracy
    and safety.  During slow / steady-state intervals it grows ``dt`` to
    save computation.
    """

    def __init__(self, config: PrinterConfig = DEFAULT_CONFIG) -> None:
        self.default_dt: float = config.default_dt
        self.min_dt: float = config.min_dt
        self.max_dt: float = config.max_dt
        self.dt: float = self.default_dt

        # Last state-change rates (for external inspection / logging)
        self.last_temp_rate: float = 0.0
        self.last_speed: float = 0.0
        self.last_max_temp: float = 0.0

    # ------------------------------------------------------------------ #
    #  Core API                                                           #
    # ------------------------------------------------------------------ #

    def compute_dt(
        self,
        temp_rate: float,
        speed: float,
        max_temp: float,
    ) -> float:
        """Compute the next time step based on current simulation conditions.

        Parameters
        ----------
        temp_rate : float
            Absolute rate of hotend temperature change (deg C / s).
        speed : float
            Current print-head speed (mm/s).
        max_temp : float
            Current maximum temperature in the system (deg C).

        Returns
        -------
        float
            The adapted time step in seconds, clamped to
            [``min_dt``, ``max_dt``].
        """
        # Store for external inspection
        self.last_temp_rate = temp_rate
        self.last_speed = speed
        self.last_max_temp = max_temp

        # Safety override — very high temperatures need fine resolution
        if max_temp > 280.0:
            dt = self.min_dt
        # Fast dynamics — use minimum step
        elif temp_rate > 50.0 or speed > 200.0:
            dt = self.min_dt
        # Moderate dynamics — use default step
        elif temp_rate > 10.0 or speed > 50.0:
            dt = self.default_dt
        # Slow / steady state — use maximum step
        elif temp_rate < 1.0 and speed < 10.0:
            dt = self.max_dt
        # In-between — linearly interpolate from default toward max
        else:
            # Use the larger of the two normalised rates as the driver
            t_frac = max(
                (temp_rate - 1.0) / (10.0 - 1.0),
                (speed - 10.0) / (50.0 - 10.0),
            )
            t_frac = float(np.clip(t_frac, 0.0, 1.0))
            dt = self.max_dt + (self.default_dt - self.max_dt) * t_frac

        # Hard clamp
        self.dt = float(np.clip(dt, self.min_dt, self.max_dt))
        return self.dt

    def reset(self) -> None:
        """Reset the time step to the default value."""
        self.dt = self.default_dt
        self.last_temp_rate = 0.0
        self.last_speed = 0.0
        self.last_max_temp = 0.0
