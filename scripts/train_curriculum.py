#!/usr/bin/env python3
"""Curriculum learning: train on progressively harder G-code."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Curriculum training on diverse G-code")
    parser.add_argument("--timesteps-per-stage", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/curriculum")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 not installed. Install with: pip install stable-baselines3")
        sys.exit(1)

    from digiprinter.envs.single_agent import PrusaCoreOneEnv
    from digiprinter.envs.wrappers import ClipAction
    from digiprinter.gcode.library import (
        single_line_gcode, calibration_cube_gcode,
        overhang_test_gcode, benchy_simplified_gcode,
        spiral_vase_gcode,
    )

    # Curriculum stages: easy -> hard
    stages = [
        ("Stage 1: Single line", single_line_gcode()),
        ("Stage 2: Calibration cube", calibration_cube_gcode()),
        ("Stage 3: Overhang test", overhang_test_gcode()),
        ("Stage 4: Simplified Benchy", benchy_simplified_gcode()),
        ("Stage 5: Spiral vase", spiral_vase_gcode(height=10.0)),  # Shorter for speed
    ]

    model = None
    for i, (stage_name, gcode) in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"{stage_name}")
        print(f"{'='*60}")

        def make_env(g=gcode):
            env = PrusaCoreOneEnv(gcode=g)
            env = ClipAction(env)
            return env

        env = DummyVecEnv([make_env])

        if model is None:
            model = SAC("MlpPolicy", env, learning_rate=3e-4,
                       buffer_size=100_000, batch_size=256, verbose=1,
                       seed=args.seed)
        else:
            model.set_env(env)

        model.learn(total_timesteps=args.timesteps_per_stage,
                   reset_num_timesteps=False)
        model.save(f"{args.save_path}/curriculum_stage_{i+1}")
        print(f"Stage {i+1} complete, model saved.")
        env.close()

    model.save(f"{args.save_path}/curriculum_final")
    print(f"\nCurriculum training complete! Final model: {args.save_path}/curriculum_final")

if __name__ == "__main__":
    main()
