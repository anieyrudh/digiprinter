from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PrinterState:
    """Full mutable simulation state."""
    # Time
    sim_time: float = 0.0
    step_count: int = 0

    # Temperatures (°C)
    hotend_temp: float = 22.0
    bed_temp: float = 22.0
    chamber_temp: float = 22.0
    ambient_temp: float = 22.0

    # Temperature targets
    hotend_target: float = 0.0
    bed_target: float = 0.0

    # Position (mm)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity (mm/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    current_speed: float = 0.0

    # Extrusion
    e_position: float = 0.0  # extruder position mm
    flow_rate: float = 0.0  # mm³/s
    mass_flow_rate: float = 0.0  # kg/s
    viscosity: float = 1000.0  # Pa·s
    pressure_drop: float = 0.0  # Pa
    die_swell: float = 1.0  # ratio
    retracted: bool = False
    retraction_amount: float = 0.0  # mm

    # Cooling
    fan_speed: float = 0.0  # 0-1
    fan_target: float = 0.0

    # Quality metrics (running averages)
    adhesion_quality: float = 1.0  # 0-1
    warping_amount: float = 0.0  # mm
    stringing_amount: float = 0.0  # mm total
    dimensional_error: float = 0.0  # fraction

    # Print progress
    current_layer: int = 0
    total_layers: int = 0
    layer_progress: float = 0.0  # 0-1 within current layer
    gcode_line: int = 0
    total_gcode_lines: int = 0

    # Energy tracking
    total_energy_j: float = 0.0
    hotend_power: float = 0.0  # W
    bed_power: float = 0.0  # W

    # Heater duties
    hotend_duty: float = 0.0  # 0-1
    bed_duty: float = 0.0

    # Flags
    is_printing: bool = False
    is_homing: bool = False
    fault: str = ""  # empty = no fault

    # Domain randomization params (set at reset)
    nozzle_wear: float = 1.0
    filament_diameter_actual: float = 1.75
    bed_adhesion_factor: float = 1.0

    def reset(self, ambient_temp: float = 22.0) -> None:
        """Reset state to power-on defaults."""
        self.sim_time = 0.0
        self.step_count = 0
        self.hotend_temp = ambient_temp
        self.bed_temp = ambient_temp
        self.chamber_temp = ambient_temp
        self.ambient_temp = ambient_temp
        self.hotend_target = 0.0
        self.bed_target = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.current_speed = 0.0
        self.e_position = 0.0
        self.flow_rate = 0.0
        self.mass_flow_rate = 0.0
        self.viscosity = 1000.0
        self.pressure_drop = 0.0
        self.die_swell = 1.0
        self.retracted = False
        self.retraction_amount = 0.0
        self.fan_speed = 0.0
        self.fan_target = 0.0
        self.adhesion_quality = 1.0
        self.warping_amount = 0.0
        self.stringing_amount = 0.0
        self.dimensional_error = 0.0
        self.current_layer = 0
        self.total_layers = 0
        self.layer_progress = 0.0
        self.gcode_line = 0
        self.total_gcode_lines = 0
        self.total_energy_j = 0.0
        self.hotend_power = 0.0
        self.bed_power = 0.0
        self.hotend_duty = 0.0
        self.bed_duty = 0.0
        self.is_printing = False
        self.is_homing = False
        self.fault = ""
