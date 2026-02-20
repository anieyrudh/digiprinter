#!/usr/bin/env python3
"""Benchmark physics simulation performance."""

import time
import argparse
import numpy as np
from digiprinter.config import DEFAULT_CONFIG
from digiprinter.materials import PLA
from digiprinter.simulation.engine import SimulationEngine
from digiprinter.gcode.library import calibration_cube_gcode


def benchmark_physics_step(num_steps: int = 100_000) -> float:
    """Benchmark raw physics stepping speed."""
    engine = SimulationEngine(DEFAULT_CONFIG, PLA)
    engine.reset(ambient_temp=22.0)
    engine.engine_state = engine.state  # alias

    # Set some realistic state
    engine.state.hotend_temp = 210.0
    engine.state.hotend_target = 210.0
    engine.state.bed_temp = 60.0
    engine.state.bed_target = 60.0
    engine.state.fan_speed = 1.0

    start = time.perf_counter()
    for _ in range(num_steps):
        engine.step_physics(0.001)
    elapsed = time.perf_counter() - start

    steps_per_sec = num_steps / elapsed
    return steps_per_sec


def benchmark_gcode_execution() -> tuple[float, int]:
    """Benchmark full G-code execution speed."""
    engine = SimulationEngine(DEFAULT_CONFIG, PLA)
    engine.reset(ambient_temp=22.0)
    gcode = calibration_cube_gcode()
    num_actions = engine.load_gcode(gcode)

    start = time.perf_counter()
    while not engine.done:
        engine.step_action()
    elapsed = time.perf_counter() - start

    actions_per_sec = num_actions / max(elapsed, 1e-6)
    return actions_per_sec, num_actions


def main():
    parser = argparse.ArgumentParser(description="Benchmark digiprinter physics")
    parser.add_argument("--physics-steps", type=int, default=100_000)
    args = parser.parse_args()

    print("=" * 60)
    print("DigiPrinter Physics Benchmark")
    print("=" * 60)

    # Physics step benchmark
    print(f"\nBenchmarking {args.physics_steps:,} physics steps...")
    steps_per_sec = benchmark_physics_step(args.physics_steps)
    print(f"  Physics steps/sec: {steps_per_sec:,.0f}")
    print(f"  Target: 10,000+ steps/sec")
    print(f"  {'PASS' if steps_per_sec > 10000 else 'BELOW TARGET'}")

    # G-code execution benchmark
    print(f"\nBenchmarking calibration cube G-code execution...")
    actions_per_sec, num_actions = benchmark_gcode_execution()
    print(f"  G-code actions: {num_actions}")
    print(f"  Actions/sec: {actions_per_sec:,.0f}")

    # Estimate episodes per hour
    episode_time = num_actions / max(actions_per_sec, 1)
    episodes_per_hour = 3600 / max(episode_time, 0.001)
    print(f"  Estimated episode time: {episode_time:.2f}s")
    print(f"  Estimated episodes/hour: {episodes_per_hour:,.0f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
