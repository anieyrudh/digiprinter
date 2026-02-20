from __future__ import annotations

from digiprinter.materials.base import MaterialProperties

PLA: MaterialProperties = MaterialProperties(
    name="PLA",
    # bulk thermal
    density=1240.0,                     # kg/m³
    specific_heat=1800.0,               # J/(kg·K)
    thermal_conductivity=0.13,          # W/(m·K)
    thermal_diffusivity=5.8e-8,         # m²/s
    # transition temperatures
    glass_transition_temp=60.0,         # °C
    melt_temp=170.0,                    # °C
    crystallization_temp=None,          # amorphous PLA — no crystallisation
    # processing window
    nozzle_temp_range=(190.0, 220.0),   # °C
    bed_temp_range=(50.0, 70.0),        # °C
    # Cross-WLF viscosity
    d1=1.0e12,                          # Pa·s
    a1=20.0,
    a2=51.6,                            # °C
    t_star=100.0,                       # °C
    tau_star=25000.0,                   # Pa
    n=0.3,
    # shrinkage / warping
    shrinkage_factor=0.003,
    warp_coefficient=0.15,
    # adhesion
    adhesion_coefficient=1.0e6,         # Pa
    adhesion_activation_energy=50000.0, # J/mol
    # retraction / ooze
    retraction_sensitivity=0.8,
    ooze_coefficient=0.3,
    # cooling
    fan_speed_default=1.0,
)
