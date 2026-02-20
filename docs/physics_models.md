# DigiPrinter Physics Models

This document describes the physics models used in the DigiPrinter digital twin of the Prusa Core One+ 3D printer. Each model is implemented as a standalone Python class under `digiprinter/physics/` and composed together by the `SimulationEngine`.

---

## Table of Contents

1. [Thermal Dynamics](#thermal-dynamics)
2. [Extrusion Model](#extrusion-model)
3. [Motion Model](#motion-model)
4. [Quality Metrics](#quality-metrics)
5. [Cooling Model](#cooling-model)
6. [Material Database](#material-database)
7. [Domain Randomization](#domain-randomization)
8. [References](#references)

---

## Thermal Dynamics

**Source:** `digiprinter/physics/thermal.py` -- `ThermalModel`

The thermal subsystem uses a lumped-parameter approach with three coupled nodes: the hotend, the heated bed, and the enclosed chamber. Each node is modeled as a thermal mass exchanging heat with its neighbors via convection, radiation, and mass transport.

### Hotend Energy Balance

The hotend temperature evolves according to:

```
dT_h/dt = (P_heater - Q_conv_h - Q_filament) / (m_h * c_h)
```

where:

- `P_heater = P_max * duty * efficiency` -- electrical power delivered by the heater cartridge (50 W max)
- `Q_conv_h = h_conv * A_h * (T_h - T_chamber)` -- convective heat loss to the chamber
- `Q_filament = m_dot * c_p * (T_h - T_filament_in)` -- heat absorbed by incoming filament
- `m_h` = 0.015 kg (heater block + nozzle mass)
- `c_h` = 500 J/(kg K) (aluminum/steel composite specific heat)

### Bed Energy Balance

```
dT_b/dt = (P_bed - Q_conv_b - Q_rad_b) / (m_b * c_b)
```

where:

- `P_bed = P_bed_max * duty * efficiency` -- bed heater power (200 W max)
- `Q_conv_b = h_conv_b * A_b * (T_b - T_chamber)` -- convective loss
- `Q_rad_b = epsilon * sigma * A_b * ((T_b + 273.15)^4 - (T_c + 273.15)^4)` -- radiative loss (Stefan-Boltzmann law)
- `epsilon` = 0.9 (powder-coated steel emissivity)
- `sigma` = 5.67e-8 W/(m^2 K^4) (Stefan-Boltzmann constant)

### Chamber Energy Balance

```
dT_c/dt = (Q_hotend_to_c + Q_bed_to_c - Q_walls - Q_vent) / (m_air * c_air)
```

where:

- `Q_walls = h_wall * A_wall * (T_c - T_ambient)` -- heat loss through enclosure walls
- `Q_vent = V_dot_vent * rho_air * c_air * (T_c - T_ambient)` -- ventilation loss

### PID Temperature Control

The hotend and bed use standard PID controllers to regulate temperature:

```
duty(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt
```

Hotend PID gains: `Kp = 20.0`, `Ki = 1.0`, `Kd = 5.0`. The duty cycle output is clamped to [0, 1].

### Interface Temperature (Inter-Layer Bonding)

The temperature at the interface between a freshly-deposited layer and the layer below is estimated using the semi-infinite solid complementary error function solution:

```
T(z, t) = T_chamber + (T_extrusion - T_chamber) * erfc(z / (2 * sqrt(alpha * t)))
```

where `alpha` is the thermal diffusivity (m^2/s) and `z` is the depth below the surface (default 0.1 mm).

### Integration Method

All thermal ODEs are integrated using **forward Euler** with a configurable time step (default dt = 1 ms). Temperatures are clamped to [-40, 400] deg C to prevent numerical blow-up.

---

## Extrusion Model

**Source:** `digiprinter/physics/extrusion.py` -- `ExtrusionModel`

### Cross-WLF Viscosity

The apparent viscosity of the polymer melt is computed using the Cross-WLF model, which captures both temperature dependence (via the WLF shift) and shear-thinning behavior (via the Cross equation):

**Zero-shear viscosity (WLF shift):**

```
eta_0(T) = D1 * exp(-A1 * (T - T*) / (A2 + T - T*))
```

**Apparent viscosity (Cross model):**

```
eta(T, gamma_dot) = eta_0(T) / (1 + (eta_0(T) * gamma_dot / tau*)^(1 - n))
```

Parameters:

| Parameter | Description | PLA | PETG |
|-----------|-------------|-----|------|
| `D1` | Reference viscosity prefactor (Pa s) | 1.0e12 | 5.0e11 |
| `A1` | WLF coefficient 1 | 20.0 | 18.0 |
| `A2` | WLF coefficient 2 (deg C) | 51.6 | 55.0 |
| `T*` | Reference temperature (deg C) | 100.0 | 110.0 |
| `tau*` | Critical shear stress (Pa) | 25,000 | 30,000 |
| `n` | Power-law index | 0.3 | 0.35 |

The viscosity is clamped to [1.0, 1e8] Pa s to maintain numerical stability.

### Wall Shear Rate (Rabinowitsch Correction)

The apparent wall shear rate for a power-law fluid in a circular nozzle is:

```
gamma_dot_w = (4Q / (pi * R^3)) * (3n + 1) / (4n)
```

where `Q` is the volumetric flow rate and `R` is the nozzle inner radius.

### Hagen-Poiseuille Pressure Drop

The pressure drop through the nozzle is computed assuming fully-developed laminar flow:

```
delta_P = 8 * eta * L * Q / (pi * R^4)
```

where `L` is the nozzle melt-zone length (5 mm) and `R` is the nozzle radius (0.2 mm).

### Die Swell

The extrudate swells upon exiting the nozzle due to elastic recovery. The die swell ratio is estimated as:

```
B = 1.0 + 0.1 * (gamma_dot / 1000) * (1 - n)
```

The actual deposited line width is `actual_width = nozzle_diameter * B`.

### Retraction and Ooze

Retraction pulls the filament back by a configurable distance (default 0.8 mm at 35 mm/s) to relieve nozzle pressure. The stringing/ooze length during travel moves is modeled as:

```
L_ooze = C_ooze * eta^(-0.5) * d_travel * (1 - f_retract * S_retract)
```

where:
- `C_ooze` is the material ooze coefficient
- `f_retract` is the fraction of retraction applied (0 to 1)
- `S_retract` is the retraction sensitivity

---

## Motion Model

**Source:** `digiprinter/physics/motion.py` -- `MotionModel`

### CoreXY Kinematics

The Prusa Core One+ uses a CoreXY belt-driven mechanism. The kinematic transformations between motor positions (A, B) and Cartesian coordinates (X, Y) are:

**Forward kinematics:**

```
X = 0.5 * (A + B)
Y = 0.5 * (A - B)
```

**Inverse kinematics:**

```
A = X + Y
B = X - Y
```

Both motors contribute to both X and Y motion, enabling high-speed printing with reduced moving mass on the toolhead.

### Trapezoidal Velocity Planner

Each move segment follows a trapezoidal velocity profile with three phases:

1. **Acceleration:** constant acceleration from current speed to cruise speed
2. **Cruise:** constant velocity at the commanded feedrate
3. **Deceleration:** constant deceleration to zero

```
Acceleration phase:   s(t) = 0.5 * a * t^2,             v(t) = a * t
Cruise phase:         s(t) = s_accel + v_max * t_cruise,  v(t) = v_max
Deceleration phase:   s(t) = s_total - 0.5 * a * t_d^2,  v(t) = v_max - a * t_d
```

If the move distance is too short to reach the requested feedrate, a **triangular profile** is used instead, with `v_peak = sqrt(a * distance)`.

Key motion parameters:

| Parameter | Value |
|-----------|-------|
| Max XY speed | 500 mm/s |
| Max Z speed | 30 mm/s |
| Max XY acceleration | 10,000 mm/s^2 |
| Max Z acceleration | 500 mm/s^2 |

### Junction Speed (Cornering)

The maximum allowable speed at the junction of two consecutive move segments uses a deviation-based cornering model:

```
v_junction = sqrt(deviation * accel * sin(theta/2) / (1 - sin(theta/2)))
```

where `theta` is the angle between the two direction vectors and `deviation` is the junction deviation parameter.

### ZV Input Shaper

A Zero Vibration (ZV) input shaper suppresses ringing at the printer's resonance frequency. The shaper convolves the position trajectory with two impulses:

```
Impulse amplitudes:  a_1 = K / (1 + K),    a_2 = 1 / (1 + K)
Impulse delays:      t_1 = 0,              t_2 = 0.5 / f_res
```

where:

```
K = exp(-zeta * pi / sqrt(1 - zeta^2))
```

Default parameters: `f_res = 50 Hz`, `zeta = 0.05` (light damping). This produces shaped commands that cancel the dominant vibration mode, reducing ringing artifacts on printed surfaces.

---

## Quality Metrics

**Source:** `digiprinter/physics/quality.py` -- `QualityModel`

The quality model tracks four independent metrics that accumulate over the course of a print.

### Adhesion (Reptation-Based Bonding)

Inter-layer bond strength is modeled using polymer reptation theory. The bonding quality depends on the interface temperature and the contact time above the glass transition temperature:

```
sigma = C_adhesion * exp(-E_a / (R * T_interface)) * t_contact^0.25
```

where:
- `C_adhesion` is the material adhesion coefficient (Pa)
- `E_a` is the activation energy for chain diffusion (J/mol)
- `R` = 8.314 J/(mol K) (gas constant)
- `T_interface` is in Kelvin
- The 0.25 exponent reflects the t^(1/4) scaling of reptation-based healing

The raw bonding strength is normalized against a reference value computed at the material's optimal nozzle temperature midpoint with 1.0 s contact time, then clamped to [0, 1].

### Warping

Warp deflection is estimated from the thermal gradient, part footprint area, and bed adhesion:

```
w = C_warp * |delta_T| * (A_footprint / 1000) * (1 - 0.8 * Q_adhesion)
```

where:
- `C_warp` is the material warp coefficient (PLA: 0.15, PETG: 0.35)
- `delta_T` is the temperature differential (extrusion temp - ambient)
- Higher bed adhesion reduces warping by a factor of up to 80%

### Stringing

String length during travel moves follows the same ooze model used in the extrusion module:

```
L_string = C_ooze * eta^(-0.5) * d_travel * (1 - f_retract * S_retract)
```

Lower viscosity (higher temperature) increases stringing; retraction reduces it.

### Dimensional Accuracy

Dimensional error is the relative deviation between commanded and actual extrusion width:

```
error = |w_actual - w_target| / w_target
```

The actual width accounts for die swell, nozzle wear, and flow rate variations.

### Aggregate Quality Score

All four metrics are tracked as running accumulators and can be queried at any time:

| Metric | Direction | Range |
|--------|-----------|-------|
| Adhesion (mean) | Higher is better | [0, 1] |
| Warping (total) | Lower is better | [0, inf) mm |
| Stringing (total) | Lower is better | [0, inf) mm |
| Dimensional error (mean) | Lower is better | [0, inf) |

---

## Cooling Model

**Source:** `digiprinter/physics/cooling.py` -- `CoolingModel`

### Fan Dynamics

The part-cooling fan is modeled with a **first-order lag** (time constant tau = 0.5 s) to represent mechanical spin-up and spin-down:

```
fan_speed(t + dt) = fan_speed(t) + (target - fan_speed(t)) * (1 - exp(-dt / tau))
```

The fan speed is clamped to [0, 1].

### Convective Cooling

Heat loss from the print surface combines natural convection (always present) and forced convection (proportional to fan speed):

```
h_eff = h_fan * fan_fraction + h_natural
Q_conv = h_eff * A_print * (T_surface - T_ambient)
```

where:
- `h_fan` = 50 W/(m^2 K) at 100% fan
- `h_natural` = 5 W/(m^2 K) (natural convection baseline)
- `A_print` = 0.0001 m^2 (effective cooling footprint on the part)

### Radiative Cooling

Radiative heat loss from the print surface follows the Stefan-Boltzmann law:

```
Q_rad = epsilon * sigma * A * (T_surface^4 - T_ambient^4)
```

where all temperatures are in Kelvin (T_K = T_C + 273.15).

### Effective Cooling Rate

The total cooling rate is the sum of both mechanisms:

```
Q_total = Q_conv + Q_rad
```

Both components use the same effective surface area. The radiative component becomes significant only at elevated surface temperatures (above approximately 150 deg C).

---

## Material Database

**Source:** `digiprinter/materials/pla.py`, `digiprinter/materials/petg.py`

All materials are defined as frozen `MaterialProperties` dataclass instances. The base class lives in `digiprinter/materials/base.py`.

### PLA (Polylactic Acid)

| Property | Value | Unit |
|----------|-------|------|
| Density | 1240 | kg/m^3 |
| Specific heat | 1800 | J/(kg K) |
| Thermal conductivity | 0.13 | W/(m K) |
| Thermal diffusivity | 5.8e-8 | m^2/s |
| Glass transition (Tg) | 60 | deg C |
| Melt temperature | 170 | deg C |
| Nozzle temp range | 190 - 220 | deg C |
| Bed temp range | 50 - 70 | deg C |
| D1 (Cross-WLF) | 1.0e12 | Pa s |
| A1 | 20.0 | -- |
| A2 | 51.6 | deg C |
| T* | 100.0 | deg C |
| tau* | 25,000 | Pa |
| n (power-law index) | 0.3 | -- |
| Shrinkage factor | 0.003 | -- |
| Warp coefficient | 0.15 | -- |
| Adhesion coefficient | 1.0e6 | Pa |
| Adhesion activation energy | 50,000 | J/mol |
| Retraction sensitivity | 0.8 | -- |
| Ooze coefficient | 0.3 | -- |
| Default fan speed | 1.0 (100%) | -- |

### PETG (Polyethylene Terephthalate Glycol-Modified)

| Property | Value | Unit |
|----------|-------|------|
| Density | 1270 | kg/m^3 |
| Specific heat | 1700 | J/(kg K) |
| Thermal conductivity | 0.17 | W/(m K) |
| Thermal diffusivity | 7.9e-8 | m^2/s |
| Glass transition (Tg) | 80 | deg C |
| Melt temperature | 230 | deg C |
| Nozzle temp range | 220 - 250 | deg C |
| Bed temp range | 75 - 90 | deg C |
| D1 (Cross-WLF) | 5.0e11 | Pa s |
| A1 | 18.0 | -- |
| A2 | 55.0 | deg C |
| T* | 110.0 | deg C |
| tau* | 30,000 | Pa |
| n (power-law index) | 0.35 | -- |
| Shrinkage factor | 0.004 | -- |
| Warp coefficient | 0.35 | -- |
| Adhesion coefficient | 8.0e5 | Pa |
| Adhesion activation energy | 55,000 | J/mol |
| Retraction sensitivity | 0.5 | -- |
| Ooze coefficient | 0.5 | -- |
| Default fan speed | 0.5 (50%) | -- |

**Key differences:** PETG has higher glass transition and melt temperatures, is more prone to warping and stringing (higher warp and ooze coefficients), and is less sensitive to retraction. PETG requires lower fan speeds to maintain inter-layer adhesion.

---

## Domain Randomization

**Source:** `digiprinter/config.py` -- `PrinterConfig`

Domain randomization is applied at the start of each episode to improve policy robustness. The following parameters are randomized within their configured ranges:

| Parameter | Default Range | Unit | Effect |
|-----------|---------------|------|--------|
| Ambient temperature | [18.0, 28.0] | deg C | Shifts all thermal equilibria |
| Filament diameter std | 0.02 | mm | Varies flow rate (1.75 +/- 0.02 mm) |
| Nozzle wear multiplier | [1.0, 1.1] | -- | Widens effective nozzle diameter |
| Heater efficiency | [0.9, 1.0] | -- | Reduces effective heater power |
| Thermistor offset | [-2.0, 2.0] | deg C | Adds measurement bias |
| Bed adhesion factor | [0.7, 1.0] | -- | Varies first-layer adhesion |

The `ThermalModel` exposes `heater_efficiency` and `thermistor_offset` as mutable attributes for runtime randomization. Nozzle wear is applied as a multiplier on the effective nozzle diameter in the extrusion model.

---

## References

1. M. L. Williams, R. F. Landel, and J. D. Ferry, "The Temperature Dependence of Relaxation Mechanisms in Amorphous Polymers and Other Glass-forming Liquids," *Journal of the American Chemical Society*, vol. 77, no. 14, pp. 3701-3707, 1955. (WLF equation)

2. M. M. Cross, "Rheology of Non-Newtonian Fluids: A New Flow Equation for Pseudoplastic Systems," *Journal of Colloid Science*, vol. 20, no. 5, pp. 417-437, 1965. (Cross viscosity model)

3. P.-G. de Gennes, "Reptation of a Polymer Chain in the Presence of Fixed Obstacles," *Journal of Chemical Physics*, vol. 55, pp. 572-579, 1971. (Reptation theory for inter-layer bonding)

4. N. C. Singer and W. P. Seering, "Preshaping Command Inputs to Reduce System Vibration," *Journal of Dynamic Systems, Measurement, and Control*, vol. 112, pp. 76-82, 1990. (ZV input shaping)

5. J. P. Kruth, L. Froyen, J. Van Vaerenbergh, P. Mercelis, M. Rombouts, and B. Lauwers, "Selective laser melting of iron-based powder," *Journal of Materials Processing Technology*, vol. 149, pp. 616-622, 2004. (Thermal modeling approach for additive manufacturing)

6. C. Bellehumeur, L. Li, Q. Sun, and P. Gu, "Modeling of Bond Formation Between Polymer Filaments in the Fused Deposition Modeling Process," *Journal of Manufacturing Processes*, vol. 6, no. 2, pp. 170-178, 2004. (FDM bonding model)

7. Prusa Research, "Original Prusa CORE One Specifications," https://www.prusa3d.com/product/original-prusa-core-one/ (Hardware parameters)
