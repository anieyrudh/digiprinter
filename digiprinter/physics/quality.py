"""Print quality metrics: adhesion, warping, stringing, dimensional accuracy."""

from __future__ import annotations

import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG
from digiprinter.materials.base import MaterialProperties

GAS_CONSTANT = 8.314  # J/(mol·K)


class QualityModel:
    """Aggregate print-quality tracker.

    Computes per-event quality metrics (layer adhesion, warping, stringing,
    dimensional accuracy) and maintains running accumulators so that
    summary scores can be queried at any point during a print.
    """

    def __init__(self, config: PrinterConfig = DEFAULT_CONFIG) -> None:
        self.config = config

        # Running quality accumulators
        self.adhesion_sum: float = 0.0
        self.adhesion_count: int = 0
        self.warp_total: float = 0.0
        self.string_total: float = 0.0
        self.dim_error_sum: float = 0.0
        self.dim_error_count: int = 0

    # ------------------------------------------------------------------
    # Adhesion
    # ------------------------------------------------------------------
    def compute_adhesion(
        self,
        interface_temp: float,
        contact_time: float,
        material: MaterialProperties,
        bed_adhesion_factor: float = 1.0,
    ) -> float:
        """Reptation-based inter-layer bonding quality.

        Parameters
        ----------
        interface_temp : float
            Temperature at the layer interface (deg C).
        contact_time : float
            Time the interface remains above Tg (seconds).
        material : MaterialProperties
            Active material.
        bed_adhesion_factor : float
            Multiplicative modifier for bed adhesion (0-1 typ.).

        Returns
        -------
        float
            Adhesion quality clamped to [0, 1].
        """
        # Reptation bonding strength
        sigma = (
            material.adhesion_coefficient
            * np.exp(
                -material.adhesion_activation_energy
                / (GAS_CONSTANT * max(interface_temp + 273.15, 1.0))
            )
            * max(contact_time, 0.0) ** 0.25
        )

        # Reference value at optimal nozzle temp midpoint and 1.0 s contact
        ref_temp = 0.5 * (
            material.nozzle_temp_range[0] + material.nozzle_temp_range[1]
        )
        sigma_ref = (
            material.adhesion_coefficient
            * np.exp(
                -material.adhesion_activation_energy
                / (GAS_CONSTANT * max(ref_temp + 273.15, 1.0))
            )
            * 1.0 ** 0.25
        )

        quality = float(np.clip(sigma / sigma_ref * bed_adhesion_factor, 0.0, 1.0))

        self.adhesion_sum += quality
        self.adhesion_count += 1
        return quality

    # ------------------------------------------------------------------
    # Warping
    # ------------------------------------------------------------------
    def compute_warping(
        self,
        delta_temp: float,
        part_footprint_area_mm2: float,
        material: MaterialProperties,
        bed_adhesion_quality: float,
    ) -> float:
        """Estimate warp deflection.

        Parameters
        ----------
        delta_temp : float
            Temperature differential (typically T_extrusion - T_ambient) in deg C.
        part_footprint_area_mm2 : float
            Area of the part footprint on the bed (mm^2).
        material : MaterialProperties
            Active material.
        bed_adhesion_quality : float
            Current bed adhesion quality [0, 1].

        Returns
        -------
        float
            Warp deflection in mm (>= 0).
        """
        w = (
            material.warp_coefficient
            * abs(delta_temp)
            * (part_footprint_area_mm2 / 1000.0)
            * (1.0 - bed_adhesion_quality * 0.8)
        )
        w = max(w, 0.0)
        self.warp_total += w
        return w

    # ------------------------------------------------------------------
    # Stringing
    # ------------------------------------------------------------------
    def compute_stringing(
        self,
        viscosity: float,
        travel_distance: float,
        material: MaterialProperties,
        retraction_fraction: float,
    ) -> float:
        """Estimate string length during a travel move.

        Parameters
        ----------
        viscosity : float
            Current melt viscosity (Pa-s).
        travel_distance : float
            Non-extrusion travel distance (mm).
        material : MaterialProperties
            Active material.
        retraction_fraction : float
            Fraction of optimal retraction applied [0, 1].

        Returns
        -------
        float
            Estimated string length in mm (>= 0).
        """
        L = (
            material.ooze_coefficient
            * viscosity ** (-0.5)
            * travel_distance
            * (1.0 - retraction_fraction * material.retraction_sensitivity)
        )
        L = max(L, 0.0)
        self.string_total += L
        return L

    # ------------------------------------------------------------------
    # Dimensional accuracy
    # ------------------------------------------------------------------
    def compute_dimensional_accuracy(
        self,
        target_width: float,
        actual_width: float,
    ) -> float:
        """Relative dimensional error of an extruded bead.

        Parameters
        ----------
        target_width : float
            Desired extrusion width (mm).
        actual_width : float
            Simulated extrusion width (mm).

        Returns
        -------
        float
            Relative error (0 = perfect).
        """
        error = abs(actual_width - target_width) / max(target_width, 0.001)
        self.dim_error_sum += error
        self.dim_error_count += 1
        return error

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------
    @property
    def average_adhesion(self) -> float:
        """Mean adhesion quality over all sampled interfaces."""
        return self.adhesion_sum / max(self.adhesion_count, 1)

    @property
    def average_dimensional_error(self) -> float:
        """Mean relative dimensional error over all sampled beads."""
        return self.dim_error_sum / max(self.dim_error_count, 1)

    @property
    def total_warping(self) -> float:
        """Cumulative warp deflection (mm)."""
        return self.warp_total

    @property
    def total_stringing(self) -> float:
        """Cumulative string length (mm)."""
        return self.string_total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def get_quality_scores(self) -> dict[str, float]:
        """Return a summary dictionary of all quality metrics.

        Keys
        ----
        adhesion : float
            Average adhesion quality (higher is better, max 1.0).
        warping : float
            Total warp deflection in mm (lower is better).
        stringing : float
            Total string length in mm (lower is better).
        dimensional_error : float
            Average relative dimensional error (lower is better).
        """
        return {
            "adhesion": self.average_adhesion,
            "warping": self.total_warping,
            "stringing": self.total_stringing,
            "dimensional_error": self.average_dimensional_error,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero all accumulators."""
        self.adhesion_sum = 0.0
        self.adhesion_count = 0
        self.warp_total = 0.0
        self.string_total = 0.0
        self.dim_error_sum = 0.0
        self.dim_error_count = 0
