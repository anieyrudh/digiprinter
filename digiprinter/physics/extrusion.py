"""Cross-WLF viscosity model, nozzle flow, die swell, and retraction."""

from __future__ import annotations

import math

import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG
from digiprinter.materials.base import MaterialProperties


class ExtrusionModel:
    """Physics-based extrusion model for FDM 3-D printing.

    Implements Cross-WLF viscosity, Hagen-Poiseuille pressure drop through
    the nozzle, die-swell estimation, and retraction / unretraction timing
    with ooze prediction.
    """

    def __init__(self, config: PrinterConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        # Mutable state
        self.pressure_drop: float = 0.0
        self.die_swell: float = 1.0
        self.retracted: bool = False
        self.retraction_amount: float = 0.0

    # ------------------------------------------------------------------
    # Viscosity
    # ------------------------------------------------------------------

    @staticmethod
    def cross_wlf_viscosity(
        temperature: float,
        shear_rate: float,
        material: MaterialProperties,
    ) -> float:
        """Compute apparent viscosity using the Cross-WLF model.

        Parameters
        ----------
        temperature : float
            Melt temperature (deg C).
        shear_rate : float
            Shear rate (s-1).
        material : MaterialProperties
            Material with Cross-WLF parameters.

        Returns
        -------
        float
            Apparent viscosity (Pa-s), clamped to [1.0, 1e8].
        """
        D1 = material.d1
        A1 = material.a1
        A2 = material.a2
        T_star = material.t_star
        tau_star = material.tau_star
        n = material.n

        # Zero-shear viscosity via WLF shift
        denom = A2 + temperature - T_star
        denom = max(denom, 1.0)  # avoid division by zero
        eta_0 = D1 * math.exp(-A1 * (temperature - T_star) / denom)

        # Clamp shear rate to a small positive value
        gamma_dot = max(shear_rate, 0.01)

        # Cross model: apparent viscosity
        eta = eta_0 / (1.0 + (eta_0 * gamma_dot / tau_star) ** (1.0 - n))

        return float(np.clip(eta, 1.0, 1e8))

    # ------------------------------------------------------------------
    # Shear rate (Rabinowitsch correction)
    # ------------------------------------------------------------------

    @staticmethod
    def shear_rate(
        volume_flow_rate: float,
        nozzle_radius: float,
        n: float,
    ) -> float:
        """Wall shear rate with Rabinowitsch correction for a power-law fluid.

        Parameters
        ----------
        volume_flow_rate : float
            Volumetric flow rate Q (m3/s).
        nozzle_radius : float
            Inner radius of the nozzle (m).
        n : float
            Power-law index.

        Returns
        -------
        float
            Apparent wall shear rate (s-1).
        """
        n = max(n, 0.01)
        R = nozzle_radius
        Q = volume_flow_rate
        gamma_dot = (4.0 * Q / (math.pi * R ** 3)) * (3.0 * n + 1.0) / (4.0 * n)
        return gamma_dot

    # ------------------------------------------------------------------
    # Full nozzle-flow computation
    # ------------------------------------------------------------------

    def compute_flow(
        self,
        print_speed: float,
        layer_height: float,
        line_width: float,
        material: MaterialProperties,
        hotend_temp: float,
        nozzle_diameter: float | None = None,
    ) -> dict:
        """Compute all extrusion-related quantities for the current state.

        Parameters
        ----------
        print_speed : float
            Print head speed (mm/s).
        layer_height : float
            Layer height (mm).
        line_width : float
            Commanded line width (mm).
        material : MaterialProperties
            Active material.
        hotend_temp : float
            Current hotend temperature (deg C).
        nozzle_diameter : float | None
            Override nozzle diameter (mm); defaults to ``config.nozzle_diameter``.

        Returns
        -------
        dict
            Keys: volume_flow_mm3s, volume_flow_m3s, shear_rate, viscosity,
            pressure_drop, die_swell, actual_line_width, mass_flow_rate.
        """
        if nozzle_diameter is None:
            nozzle_diameter = self.config.nozzle_diameter

        # Target volume flow rate
        Q_mm3s = print_speed * layer_height * line_width  # mm3/s
        Q_m3s = Q_mm3s * 1e-9  # m3/s

        nozzle_radius_m = (nozzle_diameter / 2.0) * 1e-3  # m

        # Shear rate (Rabinowitsch-corrected)
        gamma_dot = self.shear_rate(Q_m3s, nozzle_radius_m, material.n)

        # Viscosity (Cross-WLF)
        viscosity = self.cross_wlf_viscosity(hotend_temp, gamma_dot, material)

        # Pressure drop (Hagen-Poiseuille)
        nozzle_length_m = self.config.nozzle_length * 1e-3  # m
        delta_P = (
            8.0 * viscosity * nozzle_length_m * Q_m3s
            / (math.pi * nozzle_radius_m ** 4)
        )

        # Die swell
        n = material.n
        B = 1.0 + 0.1 * (gamma_dot / 1000.0) * (1.0 - n)

        # Actual line width accounting for die swell
        actual_line_width = nozzle_diameter * B

        # Mass flow rate
        mass_flow_rate = Q_m3s * material.density  # kg/s

        # Update internal state
        self.pressure_drop = delta_P
        self.die_swell = B

        return {
            "volume_flow_mm3s": Q_mm3s,
            "volume_flow_m3s": Q_m3s,
            "shear_rate": gamma_dot,
            "viscosity": viscosity,
            "pressure_drop": delta_P,
            "die_swell": B,
            "actual_line_width": actual_line_width,
            "mass_flow_rate": mass_flow_rate,
        }

    # ------------------------------------------------------------------
    # Retraction / unretraction
    # ------------------------------------------------------------------

    def retract(self, distance: float, speed: float) -> float:
        """Perform a retraction move.

        Parameters
        ----------
        distance : float
            Retraction distance (mm).
        speed : float
            Retraction speed (mm/s).

        Returns
        -------
        float
            Time to complete the retraction (s).  Returns 0.0 if already
            retracted.
        """
        if self.retracted:
            return 0.0

        distance = float(np.clip(distance, 0.0, self.config.max_retraction_dist))
        self.retracted = True
        self.retraction_amount = distance
        return distance / max(speed, 0.1)

    def unretract(self, speed: float) -> float:
        """Undo a previous retraction.

        Parameters
        ----------
        speed : float
            Unretraction speed (mm/s).

        Returns
        -------
        float
            Time to complete the unretraction (s).  Returns 0.0 if not
            currently retracted.
        """
        if not self.retracted:
            return 0.0

        time = self.retraction_amount / max(speed, 0.1)
        self.retracted = False
        self.retraction_amount = 0.0
        return time

    # ------------------------------------------------------------------
    # Ooze / stringing prediction
    # ------------------------------------------------------------------

    def compute_ooze(
        self,
        viscosity: float,
        travel_distance: float,
        material: MaterialProperties,
        retraction_fraction: float,
    ) -> float:
        """Estimate stringing / ooze length during a travel move.

        Parameters
        ----------
        viscosity : float
            Current melt viscosity (Pa-s).
        travel_distance : float
            Length of the travel move (mm).
        material : MaterialProperties
            Active material.
        retraction_fraction : float
            Fraction of effective retraction applied (0 = none, 1 = full).

        Returns
        -------
        float
            Estimated stringing length (mm), >= 0.
        """
        L = (
            material.ooze_coefficient
            * viscosity ** (-0.5)
            * travel_distance
            * (1.0 - retraction_fraction * material.retraction_sensitivity)
        )
        return max(L, 0.0)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset mutable state to defaults."""
        self.pressure_drop = 0.0
        self.die_swell = 1.0
        self.retracted = False
        self.retraction_amount = 0.0
