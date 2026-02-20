"""Lumped-parameter thermal model for hotend, bed, and chamber."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG

# Stefan-Boltzmann constant (W / (m^2 K^4))
_STEFAN_BOLTZMANN: float = 5.67e-8

# Approximate air density at room temperature (kg / m^3)
_AIR_DENSITY: float = 1.2

# Temperature clamp bounds (deg C)
_TEMP_MIN: float = -40.0
_TEMP_MAX: float = 400.0


class ThermalModel:
    """Lumped-parameter thermal model for a 3D-printer hotend, heated bed,
    and enclosed chamber.

    All temperatures are in degrees Celsius, powers in watts, and time in
    seconds.
    """

    # ------------------------------------------------------------------ #
    #  Construction / reset                                               #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        config: PrinterConfig = DEFAULT_CONFIG,
        ambient_temp: float = 22.0,
    ) -> None:
        self.config: PrinterConfig = config

        # --- State variables ---
        self.hotend_temp: float = ambient_temp
        self.bed_temp: float = ambient_temp
        self.chamber_temp: float = ambient_temp

        # --- Power tracking (W) ---
        self.hotend_power: float = 0.0
        self.bed_power: float = 0.0

        # --- Domain-randomization knobs ---
        self.heater_efficiency: float = 1.0
        self.thermistor_offset: float = 0.0  # deg C

    def reset(self, ambient_temp: float = 22.0) -> None:
        """Reset all temperatures to *ambient_temp* and zero the powers."""
        self.hotend_temp = ambient_temp
        self.bed_temp = ambient_temp
        self.chamber_temp = ambient_temp
        self.hotend_power = 0.0
        self.bed_power = 0.0

    # ------------------------------------------------------------------ #
    #  Measured-temperature properties (includes thermistor offset)       #
    # ------------------------------------------------------------------ #

    @property
    def hotend_temp_measured(self) -> float:
        """Hotend temperature as seen by the thermistor (true + offset)."""
        return self.hotend_temp + self.thermistor_offset

    @property
    def bed_temp_measured(self) -> float:
        """Bed temperature as seen by the thermistor (true + offset)."""
        return self.bed_temp + self.thermistor_offset

    # ------------------------------------------------------------------ #
    #  Forward-Euler time step                                            #
    # ------------------------------------------------------------------ #

    def step(
        self,
        dt: float,
        hotend_duty: float,
        bed_duty: float,
        fan_fraction: float,
        mass_flow_rate: float,
        filament_temp_in: float,
        ambient_temp: float,
    ) -> tuple[float, float, float]:
        """Advance the thermal state by *dt* seconds using forward Euler.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        hotend_duty : float
            Heater duty cycle for the hotend, clamped to [0, 1].
        bed_duty : float
            Heater duty cycle for the bed, clamped to [0, 1].
        fan_fraction : float
            Part-cooling fan fraction in [0, 1].
        mass_flow_rate : float
            Filament mass flow rate in kg/s entering the hotend.
        filament_temp_in : float
            Temperature of the incoming filament (deg C).
        ambient_temp : float
            External / room temperature (deg C).

        Returns
        -------
        tuple[float, float, float]
            Updated (hotend_temp, bed_temp, chamber_temp).
        """
        cfg = self.config

        # Clamp duty cycles to [0, 1]
        hotend_duty = float(np.clip(hotend_duty, 0.0, 1.0))
        bed_duty = float(np.clip(bed_duty, 0.0, 1.0))

        eff = self.heater_efficiency

        T_h = self.hotend_temp
        T_b = self.bed_temp
        T_c = self.chamber_temp

        # --- Hotend energy balance ---
        P_heater_h = cfg.hotend_heater_power * hotend_duty * eff
        Q_conv_h = cfg.hotend_h_conv * cfg.hotend_area * (T_h - T_c)
        Q_filament = mass_flow_rate * cfg.hotend_specific_heat * (T_h - filament_temp_in)
        # Note: we reuse hotend_specific_heat as a reasonable proxy for the
        # filament specific heat inside the melt zone.  A dedicated
        # c_filament can be injected via MaterialProperties if desired.
        dT_h = (P_heater_h - Q_conv_h - Q_filament) / (
            cfg.hotend_mass * cfg.hotend_specific_heat
        )

        # --- Bed energy balance ---
        P_heater_b = cfg.bed_heater_power * bed_duty * eff
        Q_conv_b = cfg.bed_h_conv * cfg.bed_area * (T_b - T_c)
        Q_rad_b = (
            cfg.bed_emissivity
            * _STEFAN_BOLTZMANN
            * cfg.bed_area
            * ((T_b + 273.15) ** 4 - (T_c + 273.15) ** 4)
        )
        dT_b = (P_heater_b - Q_conv_b - Q_rad_b) / (
            cfg.bed_mass * cfg.bed_specific_heat
        )

        # --- Chamber energy balance ---
        Q_hotend_to_c = cfg.hotend_h_conv * cfg.hotend_area * (T_h - T_c)
        Q_bed_to_c = cfg.bed_h_conv * cfg.bed_area * (T_b - T_c)
        Q_walls = cfg.chamber_wall_h * cfg.chamber_wall_area * (T_c - ambient_temp)
        Q_vent = (
            cfg.chamber_vent_flow
            * _AIR_DENSITY
            * cfg.chamber_air_cp
            * (T_c - ambient_temp)
        )
        dT_c = (Q_hotend_to_c + Q_bed_to_c - Q_walls - Q_vent) / (
            cfg.chamber_air_mass * cfg.chamber_air_cp
        )

        # --- Forward Euler integration ---
        self.hotend_temp = float(np.clip(T_h + dT_h * dt, _TEMP_MIN, _TEMP_MAX))
        self.bed_temp = float(np.clip(T_b + dT_b * dt, _TEMP_MIN, _TEMP_MAX))
        self.chamber_temp = float(np.clip(T_c + dT_c * dt, _TEMP_MIN, _TEMP_MAX))

        # --- Energy tracking ---
        self.hotend_power = P_heater_h
        self.bed_power = P_heater_b

        return self.hotend_temp, self.bed_temp, self.chamber_temp

    # ------------------------------------------------------------------ #
    #  Interface (inter-layer) temperature estimate                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def interface_temperature(
        extrusion_temp: float,
        chamber_temp: float,
        layer_time: float,
        thermal_diffusivity: float,
        z_offset: float = 0.0001,
    ) -> float:
        """Estimate the interface temperature between the freshly-extruded
        layer and the previously-deposited material.

        Uses the semi-infinite-solid complementary error function solution:

            T(z, t) = T_chamber + (T_extrusion - T_chamber)
                       * erfc(z / (2 * sqrt(alpha * t)))

        Parameters
        ----------
        extrusion_temp : float
            Freshly-extruded filament temperature (deg C).
        chamber_temp : float
            Surrounding chamber / ambient temperature (deg C).
        layer_time : float
            Time since the layer was deposited (s).
        thermal_diffusivity : float
            Material thermal diffusivity (m^2/s).
        z_offset : float, optional
            Depth below the surface at which to evaluate (m).
            Defaults to 0.1 mm.

        Returns
        -------
        float
            Estimated interface temperature (deg C).
        """
        t_safe = max(layer_time, 1e-6)
        z = z_offset
        alpha = thermal_diffusivity

        T: float = chamber_temp + (extrusion_temp - chamber_temp) * float(
            erfc(z / (2.0 * np.sqrt(alpha * t_safe)))
        )
        return T
