from __future__ import annotations

from digiprinter.materials.base import MaterialProperties

PETG: MaterialProperties = MaterialProperties(
    name="PETG",
    # bulk thermal
    density=1270.0,                     # kg/m³
    specific_heat=1700.0,               # J/(kg·K)
    thermal_conductivity=0.17,          # W/(m·K)
    thermal_diffusivity=7.9e-8,         # m²/s
    # transition temperatures
    glass_transition_temp=80.0,         # °C
    melt_temp=230.0,                    # °C
    crystallization_temp=None,          # amorphous PETG — no crystallisation
    # processing window
    nozzle_temp_range=(220.0, 250.0),   # °C
    bed_temp_range=(75.0, 90.0),        # °C
    # Cross-WLF viscosity
    d1=5.0e11,                          # Pa·s
    a1=18.0,
    a2=55.0,                            # °C
    t_star=110.0,                       # °C
    tau_star=30000.0,                   # Pa
    n=0.35,
    # shrinkage / warping
    shrinkage_factor=0.004,
    warp_coefficient=0.35,
    # adhesion
    adhesion_coefficient=8.0e5,         # Pa
    adhesion_activation_energy=55000.0, # J/mol
    # retraction / ooze
    retraction_sensitivity=0.5,
    ooze_coefficient=0.5,
    # cooling
    fan_speed_default=0.5,
)
