from __future__ import annotations

from digiprinter.materials.base import MaterialProperties

NYLON_PA6: MaterialProperties = MaterialProperties(
    name="Nylon_PA6",
    # bulk thermal
    density=1140.0,                     # kg/m³
    specific_heat=1700.0,               # J/(kg·K)
    thermal_conductivity=0.25,          # W/(m·K)
    thermal_diffusivity=1.29e-7,        # m²/s
    # transition temperatures
    glass_transition_temp=50.0,         # °C
    melt_temp=260.0,                    # °C
    crystallization_temp=195.0,         # °C — semi-crystalline
    # processing window
    nozzle_temp_range=(250.0, 275.0),   # °C
    bed_temp_range=(70.0, 90.0),        # °C
    # Cross-WLF viscosity
    d1=8.0e11,                          # Pa·s
    a1=25.0,
    a2=51.6,                            # °C
    t_star=130.0,                       # °C
    tau_star=35000.0,                   # Pa
    n=0.32,
    # shrinkage / warping
    shrinkage_factor=0.015,             # high — nylon shrinks significantly
    warp_coefficient=0.55,
    # adhesion
    adhesion_coefficient=6e5,           # Pa
    adhesion_activation_energy=58000.0, # J/mol
    # retraction / ooze
    retraction_sensitivity=0.6,
    ooze_coefficient=0.45,
    # cooling
    fan_speed_default=0.3,
)
