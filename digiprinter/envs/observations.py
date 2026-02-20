"""Observation construction helpers for the RL environment."""
from __future__ import annotations

import numpy as np

from digiprinter.printer.state import PrinterState
from digiprinter.config import PrinterConfig


def build_observation(state: PrinterState, config: PrinterConfig) -> np.ndarray:
    """Build a 34-float observation vector from the current printer state.

    All values are normalised to roughly [0, 1] and clipped to [0, 1.5]
    before being returned as an np.float32 array.

    Index layout
    ------------
    0-4   : Temperatures (5)
    5-13  : Motion (9)
    14-17 : Extrusion (4)
    18-19 : Cooling (2)
    20-24 : Quality (5)
    25-27 : Progress (3)
    28-33 : G-code context (6)
    """
    obs = np.zeros(34, dtype=np.float32)

    # --- Temperatures (indices 0-4) ---
    obs[0] = state.hotend_temp / config.hotend_max_temp
    obs[1] = state.bed_temp / config.bed_max_temp
    obs[2] = state.chamber_temp / 80.0  # reasonable max chamber temp
    obs[3] = state.hotend_target / config.hotend_max_temp
    obs[4] = state.bed_target / config.bed_max_temp

    # --- Motion (indices 5-13) ---
    obs[5] = state.x / config.build_x
    obs[6] = state.y / config.build_y
    obs[7] = state.z / config.build_z
    obs[8] = state.vx / config.max_speed_xy
    obs[9] = state.vy / config.max_speed_xy
    obs[10] = state.vz / config.max_speed_z
    obs[11] = state.current_speed / config.max_speed_xy
    obs[12] = state.hotend_duty
    obs[13] = state.bed_duty

    # --- Extrusion (indices 14-17) ---
    obs[14] = state.flow_rate / 30.0  # typical max mm^3/s
    obs[15] = state.viscosity / 10000.0  # linear normalisation
    obs[16] = state.pressure_drop / 1e7  # normalise to ~1
    obs[17] = state.die_swell - 1.0  # centred at 0

    # --- Cooling (indices 18-19) ---
    obs[18] = state.fan_speed
    obs[19] = state.fan_target

    # --- Quality (indices 20-24) ---
    obs[20] = state.adhesion_quality
    obs[21] = state.warping_amount / 1.0  # mm, will be clipped
    obs[22] = state.stringing_amount / 10.0  # mm total, will be clipped
    obs[23] = state.dimensional_error  # already 0-1ish
    obs[24] = 1.0 if state.fault else 0.0

    # --- Progress (indices 25-27) ---
    obs[25] = state.current_layer / max(state.total_layers, 1)
    obs[26] = state.layer_progress
    obs[27] = state.gcode_line / max(state.total_gcode_lines, 1)

    # --- G-code context (indices 28-33) ---
    obs[28] = float(state.is_printing)
    obs[29] = float(state.retracted)
    obs[30] = state.retraction_amount / config.max_retraction_dist
    obs[31] = state.nozzle_wear
    obs[32] = state.filament_diameter_actual / config.filament_diameter
    obs[33] = state.bed_adhesion_factor

    # Clip to [0, 1.5] for safety (slight overflows are tolerated)
    np.clip(obs, 0.0, 1.5, out=obs)

    return obs


def observation_space(config: PrinterConfig):
    """Return a gymnasium Box describing the 34-dim observation space.

    The upper bound is 1.5 rather than 1.0 to tolerate slight overflows
    in normalised quantities.
    """
    import gymnasium
    return gymnasium.spaces.Box(
        low=0.0,
        high=1.5,
        shape=(34,),
        dtype=np.float32,
    )
