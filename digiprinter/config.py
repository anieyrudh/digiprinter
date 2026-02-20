"""PrinterConfig — all Prusa Core One+ specifications."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrinterConfig:
    """Hardware specifications for the Prusa Core One+ 3D printer."""

    # --- Build volume (mm) ---
    build_x: float = 250.0
    build_y: float = 210.0
    build_z: float = 220.0

    # --- Nozzle ---
    nozzle_diameter: float = 0.4  # mm
    nozzle_length: float = 5.0  # mm (melt zone)
    nozzle_radius: float = 0.2  # mm (nozzle_diameter / 2)
    filament_diameter: float = 1.75  # mm

    # --- Motion (CoreXY) ---
    max_speed_xy: float = 500.0  # mm/s
    max_speed_z: float = 30.0  # mm/s
    max_accel_xy: float = 10000.0  # mm/s²
    max_accel_z: float = 500.0  # mm/s²
    max_jerk_xy: float = 10.0  # mm/s (junction deviation proxy)
    steps_per_mm_xy: float = 100.0  # steps/mm (belt-driven)
    steps_per_mm_z: float = 400.0  # steps/mm (leadscrew)

    # --- Input shaper ---
    resonance_freq: float = 50.0  # Hz (typical CoreXY)
    damping_ratio: float = 0.05  # ζ (light damping)

    # --- Hotend ---
    hotend_mass: float = 0.015  # kg (heater block + nozzle)
    hotend_specific_heat: float = 500.0  # J/(kg·K) (aluminum/steel mix)
    hotend_heater_power: float = 50.0  # W
    hotend_h_conv: float = 5.0  # W/(m²·K) convective coeff to chamber
    hotend_area: float = 0.001  # m² (surface area for convection)
    hotend_max_temp: float = 300.0  # °C
    hotend_pid_kp: float = 20.0
    hotend_pid_ki: float = 1.0
    hotend_pid_kd: float = 5.0

    # --- Heated bed ---
    bed_mass: float = 1.5  # kg (steel sheet + heatbed)
    bed_specific_heat: float = 500.0  # J/(kg·K)
    bed_heater_power: float = 200.0  # W
    bed_h_conv: float = 10.0  # W/(m²·K) convective coeff to chamber
    bed_area: float = 0.0525  # m² (250×210 mm)
    bed_emissivity: float = 0.9  # powder-coated steel
    bed_max_temp: float = 120.0  # °C

    # --- Chamber ---
    chamber_volume: float = 0.05  # m³ (~50L enclosed)
    chamber_air_mass: float = 0.06  # kg (air at ~20°C)
    chamber_air_cp: float = 1005.0  # J/(kg·K)
    chamber_wall_h: float = 3.0  # W/(m²·K)
    chamber_wall_area: float = 0.8  # m² (total wall surface)
    chamber_vent_flow: float = 0.0  # m³/s (controllable vent)

    # --- Cooling fan ---
    fan_max_flow: float = 0.005  # m³/s (50mm fan)
    fan_h_coefficient: float = 50.0  # W/(m²·K) at 100% fan
    fan_print_area: float = 0.0001  # m² (effective cooling area on part)

    # --- Retraction ---
    default_retraction_dist: float = 0.8  # mm
    default_retraction_speed: float = 35.0  # mm/s
    max_retraction_dist: float = 5.0  # mm

    # --- Default print settings ---
    default_layer_height: float = 0.2  # mm
    default_print_speed: float = 100.0  # mm/s
    default_travel_speed: float = 250.0  # mm/s
    default_first_layer_speed: float = 30.0  # mm/s

    # --- Safety ---
    thermal_runaway_threshold: float = 15.0  # °C above target triggers fault
    thermal_runaway_timeout: float = 30.0  # seconds
    max_heater_duty: float = 1.0  # 0-1

    # --- Simulation ---
    default_dt: float = 0.001  # s (1 ms time step)
    min_dt: float = 0.0001  # s
    max_dt: float = 0.01  # s

    # --- Domain randomization ranges ---
    ambient_temp_range: tuple[float, float] = (18.0, 28.0)
    filament_diameter_std: float = 0.02  # mm std deviation
    nozzle_wear_range: tuple[float, float] = (1.0, 1.1)  # multiplier
    heater_efficiency_range: tuple[float, float] = (0.9, 1.0)
    thermistor_offset_range: tuple[float, float] = (-2.0, 2.0)  # °C
    bed_adhesion_range: tuple[float, float] = (0.7, 1.0)


# Singleton default config
DEFAULT_CONFIG = PrinterConfig()
