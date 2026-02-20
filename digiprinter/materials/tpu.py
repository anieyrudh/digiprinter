from __future__ import annotations

from digiprinter.materials.base import MaterialProperties

TPU_95A: MaterialProperties = MaterialProperties(
    name="TPU_95A",
    # bulk thermal
    density=1210.0,                     # kg/m³
    specific_heat=1500.0,               # J/(kg·K)
    thermal_conductivity=0.19,          # W/(m·K)
    thermal_diffusivity=1.05e-7,        # m²/s
    # transition temperatures
    glass_transition_temp=-40.0,        # °C
    melt_temp=220.0,                    # °C
    crystallization_temp=None,          # amorphous — no crystallisation
    # processing window
    nozzle_temp_range=(220.0, 245.0),   # °C
    bed_temp_range=(40.0, 60.0),        # °C
    # Cross-WLF viscosity
    d1=2.0e10,                          # Pa·s
    a1=15.0,
    a2=60.0,                            # °C
    t_star=90.0,                        # °C
    tau_star=20000.0,                   # Pa
    n=0.4,
    # shrinkage / warping
    shrinkage_factor=0.002,
    warp_coefficient=0.05,              # very low — flexible material
    # adhesion
    adhesion_coefficient=1.2e6,         # Pa
    adhesion_activation_energy=45000.0, # J/mol
    # retraction / ooze
    retraction_sensitivity=0.1,         # barely retracts — flexible filament
    ooze_coefficient=0.8,               # oozes a lot
    # cooling
    fan_speed_default=0.5,
)
