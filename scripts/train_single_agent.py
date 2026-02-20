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
    # Weights & Biases (optional)
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="digiprinter",
                        help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    args = parser.parse_args()

    # ---- Weights & Biases setup (optional) --------------------------------
    wandb = None
    wandb_callback = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print(
                "wandb is not installed. Install with:\n"
                "  pip install wandb\n"
                "Then run `wandb login` to authenticate.\n"
                "Continuing without W&B logging."
            )
            args.wandb = False

    if args.wandb and wandb is not None:
        hyperparams = {
            "timesteps": args.timesteps,
            "seed": args.seed,
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "algorithm": "SAC",
            "policy": "MlpPolicy",
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=hyperparams,
            sync_tensorboard=True,
        )

        # Try to use SB3's official WandbCallback; fall back to a lightweight
        # custom callback if the integration module is unavailable.
        try:
            from wandb.integration.sb3 import WandbCallback
            wandb_callback = WandbCallback(
                gradient_save_freq=0,
                verbose=2,
            )
        except ImportError:
            from stable_baselines3.common.callbacks import BaseCallback

            class _SimpleWandbCallback(BaseCallback):
                """Logs training metrics to W&B every *log_freq* calls."""

                def __init__(self, log_freq: int = 1000, verbose: int = 0):
                    super().__init__(verbose)
                    self.log_freq = log_freq

                def _on_step(self) -> bool:
                    if self.n_calls % self.log_freq == 0:
                        logs = {}
                        # SB3 stores recent info in self.locals
                        infos = self.locals.get("infos", [])
                        if infos:
                            for key in ("reward", "episode_reward", "ep_rew_mean"):
                                if key in infos[-1]:
                                    logs[key] = infos[-1][key]
                        if self.locals.get("log_interval") is not None:
                            for k, v in self.logger.name_to_value.items():
                                logs[k] = v
                        if logs:
                            wandb.log(logs, step=self.num_timesteps)
                    return True

            wandb_callback = _SimpleWandbCallback(log_freq=1000)
            print("wandb.integration.sb3 not found; using lightweight custom callback.")

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

    # Assemble callbacks
    callbacks = [eval_callback, checkpoint_callback]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    print(f"Training SAC for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
    )

    model.save(f"{args.save_path}/sac_final")
    print(f"Model saved to {args.save_path}/sac_final")

    # ---- Final wandb logging & cleanup ------------------------------------
    if args.wandb and wandb is not None:
        # Log the best eval reward recorded by EvalCallback
        try:
            import numpy as np
            eval_log = f"{args.log_dir}/evaluations.npz"
            data = np.load(eval_log)
            final_mean_reward = float(data["results"].mean(axis=1)[-1])
            wandb.log({"eval/final_mean_reward": final_mean_reward})
            print(f"Final eval mean reward: {final_mean_reward:.2f}")
        except Exception as exc:
            print(f"Could not load eval results for wandb: {exc}")
        wandb.finish()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
