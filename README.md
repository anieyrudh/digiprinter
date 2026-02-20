# DigiPrinter

Physics-based digital twin of the Prusa Core One+ 3D printer for training AI/RL agents.

## Features

- **Lumped-parameter thermal dynamics** -- coupled hotend, bed, and chamber nodes with PID control
- **Cross-WLF extrusion model** -- shear-thinning viscosity, Hagen-Poiseuille pressure drop, die swell prediction
- **CoreXY kinematics** -- trapezoidal velocity planner with ZV input shaper for vibration suppression
- **Print quality metrics** -- adhesion (reptation theory), warping, stringing, dimensional accuracy
- **G-code parser and interpreter** -- reads standard G-code and drives the simulation step-by-step
- **Domain randomization** -- per-episode randomization of ambient temperature, filament diameter, nozzle wear, heater efficiency, thermistor offset, and bed adhesion
- **Single-agent (Gymnasium) and multi-agent (PettingZoo) RL environments** with shaped multi-component rewards
- **Matplotlib live visualization dashboard** via `render_mode="human"`
- **75k+ physics steps/sec**, **13k+ episodes/hour** throughput

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Requires Python 3.11 or later. Core dependencies: NumPy, SciPy, Gymnasium, Stable-Baselines3, PettingZoo, Ray RLlib.

## Usage

### Single-Agent Training

```python
import gymnasium
from digiprinter.envs.single_agent import PrusaCoreOneEnv

env = PrusaCoreOneEnv()
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

The environment is also registered as `PrusaCoreOne-v0` for use with `gymnasium.make()`.

### Train with Stable-Baselines3

```bash
python scripts/train_single_agent.py --timesteps 100000
```

### Multi-Agent Training

```bash
python scripts/train_multi_agent.py --timesteps 100000
```

### Visualization

```python
env = PrusaCoreOneEnv(render_mode="human")
```

This opens a live Matplotlib dashboard showing temperatures, motion, extrusion state, and quality metrics in real time.

### Benchmark

```bash
python scripts/benchmark_physics.py
```

## Architecture

```
digiprinter/
    config.py               # PrinterConfig dataclass (hardware specs, domain randomization ranges)
    physics/
        thermal.py          # Lumped-parameter thermal model (hotend, bed, chamber)
        extrusion.py        # Cross-WLF viscosity, Hagen-Poiseuille flow, die swell
        motion.py           # CoreXY kinematics, trapezoidal planner, ZV input shaper
        cooling.py          # Part cooling fan model
        quality.py          # Adhesion, warping, stringing, dimensional accuracy
    materials/
        base.py             # MaterialProperties dataclass
        pla.py              # PLA material profile
        petg.py             # PETG material profile
    printer/
        state.py            # PrinterState (full simulation state vector)
        hardware.py         # PID controller and hardware abstractions
    gcode/
        parser.py           # G-code tokenizer and parser
        interpreter.py      # G-code interpreter driving simulation actions
        library.py          # Built-in G-code programs (calibration cube, etc.)
    simulation/
        engine.py           # SimulationEngine coupling all physics modules
        time_stepper.py     # Adaptive time-stepping
        events.py           # Event manager for simulation callbacks
    envs/
        single_agent.py     # Gymnasium environment (PrusaCoreOneEnv)
        multi_agent.py      # PettingZoo ParallelEnv (PrusaCoreOneMultiAgentEnv)
        observations.py     # 34-float observation vector construction
        rewards.py          # Weighted multi-component reward function
        renderer.py         # Matplotlib live visualization
        wrappers.py         # Environment wrappers
    utils/
        math_helpers.py     # Shared math utilities
scripts/
    train_single_agent.py   # SB3 training script
    train_multi_agent.py    # Multi-agent training script
    benchmark_physics.py    # Physics throughput benchmark
tests/
    test_thermal.py         # Thermal model tests
    test_extrusion.py       # Extrusion model tests
    test_motion.py          # Motion model tests
    test_quality.py         # Quality metrics tests
    test_gcode.py           # G-code parser/interpreter tests
    test_single_agent_env.py# Single-agent environment tests
    test_multi_agent_env.py # Multi-agent environment tests
docs/
    physics_models.md       # Detailed physics model documentation with equations
```

## Physics Models

All physics models are documented in detail in [docs/physics_models.md](docs/physics_models.md).

**Thermal dynamics** -- Three coupled lumped-parameter nodes (hotend, bed, chamber) with convection, radiation, and filament mass transport. PID controllers manage heater duty cycles. Hotend heater: 50 W max; bed heater: 200 W max.

**Extrusion** -- Cross-WLF viscosity model captures shear-thinning behavior of molten polymer. Hagen-Poiseuille flow through the nozzle computes pressure drop and volumetric flow rate. Die swell prediction accounts for extrudate expansion at the nozzle exit.

**Motion** -- CoreXY kinematic transform maps Cartesian commands to belt-driven motor steps. A trapezoidal velocity planner enforces acceleration and jerk limits (10,000 mm/s^2 max XY acceleration, 500 mm/s max XY speed). A ZV (Zero Vibration) input shaper filters commanded motion to suppress resonance.

**Quality metrics** -- Layer adhesion is modeled via reptation theory (polymer chain interdiffusion). Warping is computed from thermal gradients and material shrinkage coefficients. Stringing is estimated from retraction behavior and ooze dynamics. Dimensional accuracy tracks deviation from commanded geometry.

## RL Environment

### Observation Space (34 floats)

| Index | Group          | Description                                      |
|-------|----------------|--------------------------------------------------|
| 0-4   | Temperatures   | Hotend, bed, chamber temps; hotend/bed targets   |
| 5-13  | Motion         | XYZ position, XYZ velocity, speed, heater duties |
| 14-17 | Extrusion      | Flow rate, viscosity, pressure drop, die swell   |
| 18-19 | Cooling        | Fan speed, fan target                            |
| 20-24 | Quality        | Adhesion, warping, stringing, dimensional error, surface quality |
| 25-27 | Progress       | Layer progress, total progress, time             |
| 28-33 | G-code context | Current command parameters                       |

All values are normalized to approximately [0, 1] and clipped to [0, 1.5].

### Action Space

**Single-agent** -- `Box(-1, 1, (6,), float32)`

| Index | Action                    | Mapped Range       |
|-------|---------------------------|--------------------|
| 0     | Speed modifier            | 0.5x -- 1.5x      |
| 1     | Flow modifier             | 0.8x -- 1.2x      |
| 2     | Hotend temperature offset | -20 -- +20 deg C   |
| 3     | Bed temperature offset    | -10 -- +10 deg C   |
| 4     | Fan override              | 0.0 -- 1.0         |
| 5     | Retraction modifier       | 0.5x -- 1.5x      |

**Multi-agent** -- Three cooperative agents sharing observations:

| Agent             | Action Dim | Controls                                          |
|-------------------|------------|---------------------------------------------------|
| `thermal_agent`   | 4          | Hotend offset, bed offset, fan override, vent     |
| `motion_agent`    | 2          | Speed modifier, acceleration modifier             |
| `extrusion_agent` | 2          | Flow modifier, retraction modifier                |

### Reward Function

Weighted sum of 8 components (weights sum to ~1.0):

| Component              | Weight | Description                                         |
|------------------------|--------|-----------------------------------------------------|
| Adhesion               | 0.25   | Layer bonding quality (0-1)                         |
| Dimensional accuracy   | 0.20   | Deviation from target geometry                      |
| Warping                | 0.15   | Thermal warping penalty                             |
| Stringing              | 0.10   | Ooze and stringing penalty                          |
| Speed                  | 0.10   | Printing throughput reward                          |
| Thermal stability      | 0.10   | Temperature oscillation penalty                     |
| Energy efficiency      | 0.05   | Power consumption penalty                           |
| Safety                 | 0.05   | Fault condition penalty (-10 on fault)              |

## Materials

Two material profiles are included:

| Material | Nozzle Temp   | Bed Temp    | Key Properties                                |
|----------|---------------|-------------|-----------------------------------------------|
| **PLA**  | 190 -- 220 C  | 50 -- 70 C  | Low shrinkage (0.3%), high adhesion, full fan  |
| **PETG** | 220 -- 250 C  | 75 -- 90 C  | Higher shrinkage (0.4%), more ooze, half fan   |

Each profile includes density, specific heat, thermal conductivity, glass transition temperature, Cross-WLF viscosity parameters, shrinkage/warp coefficients, adhesion parameters, and retraction sensitivity.

## Domain Randomization

The following parameters are randomized at the start of each episode to improve policy robustness and sim-to-real transfer:

| Parameter              | Range / Std       | Effect                                  |
|------------------------|-------------------|-----------------------------------------|
| Ambient temperature    | 18.0 -- 28.0 C    | Shifts all thermal equilibria           |
| Filament diameter      | 1.75 +/- 0.02 mm | Varies volumetric flow rate             |
| Nozzle wear multiplier | 1.0 -- 1.1       | Widens effective nozzle diameter        |
| Heater efficiency      | 0.9 -- 1.0       | Reduces effective heater power          |
| Thermistor offset      | -2.0 -- +2.0 C    | Adds temperature measurement bias      |
| Bed adhesion factor    | 0.7 -- 1.0       | Varies first-layer adhesion strength    |

## Testing

```bash
python -m pytest tests/ -v
```

68 tests covering all physics modules (thermal, extrusion, motion, quality), G-code parsing and interpretation, and both single-agent and multi-agent RL environments.

## License

MIT
