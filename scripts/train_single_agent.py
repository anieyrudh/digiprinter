#!/usr/bin/env python3
"""Train a single-agent SAC policy on the Prusa Core One+ digital twin."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Train SAC on PrusaCoreOne-v0")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/sac_prusa")
    parser.add_argument("--log-dir", type=str, default="runs/sac")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("stable-baselines3 not installed. Install with: pip install stable-baselines3")
        sys.exit(1)

    from digiprinter.envs.single_agent import PrusaCoreOneEnv
    from digiprinter.envs.wrappers import NormalizeObservation, ClipAction

    # Create training env
    def make_env():
        env = PrusaCoreOneEnv()
        env = ClipAction(env)
        return env

    env = DummyVecEnv([make_env])

    # Create eval env
    eval_env = DummyVecEnv([make_env])

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_path,
        log_path=args.log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=args.save_path,
        name_prefix="sac_prusa",
    )

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    print(f"Training SAC for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    model.save(f"{args.save_path}/sac_final")
    print(f"Model saved to {args.save_path}/sac_final")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
