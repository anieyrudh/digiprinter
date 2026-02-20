"""Fan model and convective/radiative cooling for the print surface."""

from __future__ import annotations

import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG


class CoolingModel:
    """Part-cooling fan with first-order lag and convective + radiative
    heat-loss calculations.

    All temperatures are in degrees Celsius, powers in watts, lengths in
    millimetres (unless otherwise noted), and time in seconds.
    """

    # Stefan-Boltzmann constant (W / (m^2 K^4))
    STEFAN_BOLTZMANN: float = 5.67e-8

    # ------------------------------------------------------------------ #
    #  Construction / reset                                               #
    # ------------------------------------------------------------------ #

    def __init__(self, config: PrinterConfig = DEFAULT_CONFIG) -> None:
        self.config: PrinterConfig = config
        self.fan_speed: float = 0.0   # current fan fraction [0, 1]
        self.fan_target: float = 0.0  # commanded fan fraction [0, 1]

    def reset(self) -> None:
        """Reset fan state to zero."""
        self.fan_speed = 0.0
        self.fan_target = 0.0

    # ------------------------------------------------------------------ #
    #  Fan dynamics                                                       #
    # ------------------------------------------------------------------ #

    def update_fan(self, dt: float, target: float) -> float:
        """Ramp the fan speed towards *target* using a first-order lag.

        The time constant is 0.5 s, which models the mechanical spin-up /
        spin-down of a typical 50 mm axial fan.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        target : float
            Desired fan fraction in [0, 1].

        Returns
        -------
        float
            Updated fan speed in [0, 1].
        """
        tau = 0.5  # time constant (s)
        self.fan_target = target
        self.fan_speed += (target - self.fan_speed) * (1.0 - np.exp(-dt / tau))
        self.fan_speed = float(np.clip(self.fan_speed, 0.0, 1.0))
        return self.fan_speed

    # ------------------------------------------------------------------ #
    #  Heat-loss calculations                                             #
    # ------------------------------------------------------------------ #

    def convective_cooling_rate(
        self,
        surface_temp: float,
        ambient_temp: float,
        fan_fraction: float,
    ) -> float:
        """Convective heat-loss rate from the print surface.

        A natural-convection base of 5 W/(m^2 K) is always present; the
        part-cooling fan adds forced convection proportional to
        ``config.fan_h_coefficient``.

        Parameters
        ----------
        surface_temp : float
            Surface temperature of the part (deg C).
        ambient_temp : float
            Surrounding air / chamber temperature (deg C).
        fan_fraction : float
            Fan duty in [0, 1].

        Returns
        -------
        float
            Heat-loss rate in watts (positive = heat leaving the part).
        """
        h_eff = self.config.fan_h_coefficient * fan_fraction + 5.0
        Q = h_eff * self.config.fan_print_area * (surface_temp - ambient_temp)
        return float(Q)

    def radiative_cooling_rate(
        self,
        surface_temp: float,
        ambient_temp: float,
        emissivity: float,
        area: float,
    ) -> float:
        """Radiative heat-loss rate from a surface.

        Parameters
        ----------
        surface_temp : float
            Surface temperature (deg C).
        ambient_temp : float
            Surrounding temperature (deg C).
        emissivity : float
            Surface emissivity in [0, 1].
        area : float
            Radiating surface area in m^2.

        Returns
        -------
        float
            Heat-loss rate in watts (positive = heat leaving the surface).
        """
        T_s = surface_temp + 273.15
        T_a = ambient_temp + 273.15
        Q = emissivity * self.STEFAN_BOLTZMANN * area * (T_s ** 4 - T_a ** 4)
        return float(Q)

    # ------------------------------------------------------------------ #
    #  Layer timing                                                       #
    # ------------------------------------------------------------------ #

    def layer_cooling_time(
        self,
        layer_area_mm2: float,
        print_speed: float,
        perimeter_mm: float,
    ) -> float:
        """Estimate the cooling time available for one layer.

        This is simply the travel time around the perimeter at the given
        print speed.

        Parameters
        ----------
        layer_area_mm2 : float
            Area of the layer in mm^2 (unused, kept for API symmetry).
        print_speed : float
            Print speed in mm/s.
        perimeter_mm : float
            Perimeter length of the layer in mm.

        Returns
        -------
        float
            Estimated layer time in seconds.
        """
        return perimeter_mm / print_speed

    # ------------------------------------------------------------------ #
    #  Combined cooling rate                                              #
    # ------------------------------------------------------------------ #

    def effective_cooling_rate(
        self,
        surface_temp: float,
        chamber_temp: float,
        fan_fraction: float,
        emissivity: float = 0.9,
    ) -> float:
        """Total (convective + radiative) cooling rate from the print surface.

        Both components use ``config.fan_print_area`` as the effective
        surface area.

        Parameters
        ----------
        surface_temp : float
            Part surface temperature (deg C).
        chamber_temp : float
            Chamber / ambient temperature (deg C).
        fan_fraction : float
            Fan duty in [0, 1].
        emissivity : float, optional
            Surface emissivity, by default 0.9.

        Returns
        -------
        float
            Total heat-loss rate in watts.
        """
        Q_conv = self.convective_cooling_rate(surface_temp, chamber_temp, fan_fraction)
        Q_rad = self.radiative_cooling_rate(
            surface_temp, chamber_temp, emissivity, self.config.fan_print_area
        )
        return Q_conv + Q_rad
