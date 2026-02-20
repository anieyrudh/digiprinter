#!/usr/bin/env python3
"""Train multi-agent PPO with CTDE on the Prusa Core One+ digital twin."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Train multi-agent PPO on PrusaCoreOne")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/multi_agent")
    args = parser.parse_args()

    try:
        import ray
        from ray import tune
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env import PettingZooEnv
        from ray.tune.registry import register_env
    except ImportError:
        print("Ray RLlib not installed. Install with: pip install 'ray[rllib]'")
        sys.exit(1)

    from digiprinter.envs.multi_agent import PrusaCoreOneMultiAgentEnv

    def env_creator(config):
        return PrusaCoreOneMultiAgentEnv()

    ray.init(ignore_reinit_error=True)
    register_env("PrusaCoreOneMultiAgent", lambda config: PettingZooEnv(env_creator(config)))

    config = (
        PPOConfig()
        .environment(env="PrusaCoreOneMultiAgent")
        .rollouts(num_rollout_workers=args.num_workers)
        .training(
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies={
                "thermal_policy": (None, None, None, {}),
                "motion_policy": (None, None, None, {}),
                "extrusion_policy": (None, None, None, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: {
                "thermal_agent": "thermal_policy",
                "motion_agent": "motion_policy",
                "extrusion_agent": "extrusion_policy",
            }[agent_id],
        )
        .framework("torch")
        .debugging(seed=args.seed)
    )

    stop = {"timesteps_total": args.timesteps}

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=args.checkpoint_dir,
        verbose=1,
    )

    print("Training complete!")
    best_checkpoint = results.get_best_checkpoint(
        results.get_best_trial("episode_reward_mean", mode="max"),
        "episode_reward_mean",
        mode="max",
    )
    print(f"Best checkpoint: {best_checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()
