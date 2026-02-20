from __future__ import annotations

from digiprinter.materials.base import MaterialProperties

ABS: MaterialProperties = MaterialProperties(
    name="ABS",
    # bulk thermal
    density=1040.0,                     # kg/m³
    specific_heat=1400.0,               # J/(kg·K)
    thermal_conductivity=0.17,          # W/(m·K)
    thermal_diffusivity=1.17e-7,        # m²/s
    # transition temperatures
    glass_transition_temp=105.0,        # °C
    melt_temp=230.0,                    # °C
    crystallization_temp=None,          # amorphous — no crystallisation
    # processing window
    nozzle_temp_range=(230.0, 260.0),   # °C
    bed_temp_range=(90.0, 110.0),       # °C
    # Cross-WLF viscosity
    d1=3.0e11,                          # Pa·s
    a1=22.0,
    a2=51.6,                            # °C
    t_star=120.0,                       # °C
    tau_star=28000.0,                   # Pa
    n=0.28,
    # shrinkage / warping
    shrinkage_factor=0.007,
    warp_coefficient=0.65,              # high — ABS warps aggressively
    # adhesion
    adhesion_coefficient=7e5,           # Pa
    adhesion_activation_energy=52000.0, # J/mol
    # retraction / ooze
    retraction_sensitivity=0.7,
    ooze_coefficient=0.4,
    # cooling
    fan_speed_default=0.0,              # ABS hates fan — keep it off
)
