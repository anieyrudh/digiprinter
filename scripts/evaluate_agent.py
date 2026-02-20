#!/usr/bin/env python3
"""Evaluate trained agent against baselines.

Runs three policies (trained SAC, random, zero-action) across multiple
episodes and prints a comparison table of key performance metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Ensure package is importable when running from the scripts/ directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from digiprinter.envs.single_agent import PrusaCoreOneEnv
from digiprinter.envs.wrappers import ClipAction


# ---------------------------------------------------------------------------
# Policy runner
# ---------------------------------------------------------------------------

def run_policy(
    env: PrusaCoreOneEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    seed: int,
) -> list[dict]:
    """Run a policy for *num_episodes* episodes, returning per-episode metrics.

    Parameters
    ----------
    env : PrusaCoreOneEnv
        The Gymnasium environment instance.
    policy_fn : callable
        Maps an observation array to an action array.
    num_episodes : int
        How many episodes to run.
    seed : int
        Base seed; episode *i* uses ``seed + i``.

    Returns
    -------
    list[dict]
        One result dict per episode with keys: reward, length, adhesion,
        warping, stringing, dim_error, fault.
    """
    results: list[dict] = []

    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        total_reward = 0.0
        steps = 0

        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        results.append({
            "reward": total_reward,
            "length": steps,
            "adhesion": info.get("adhesion", 0.0),
            "warping": info.get("warping", 0.0),
            "stringing": info.get("stringing", 0.0),
            "dim_error": info.get("dimensional_error", 0.0),
            "fault": info.get("fault", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results_dict: dict[str, list[dict]]) -> None:
    """Print a formatted comparison table across policies.

    Parameters
    ----------
    results_dict : dict[str, list[dict]]
        Mapping of policy name to list of episode result dicts.
    """
    metrics = [
        ("reward", "Total Reward"),
        ("length", "Episode Length"),
        ("adhesion", "Adhesion"),
        ("warping", "Warping"),
        ("stringing", "Stringing"),
        ("dim_error", "Dim. Error"),
    ]

    policy_names = list(results_dict.keys())

    # Column widths
    metric_col_w = 16
    policy_col_w = 24

    # Header
    header = f"{'Metric':<{metric_col_w}}"
    for name in policy_names:
        header += f"{name:^{policy_col_w}}"
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    # Rows for numeric metrics (mean +/- std)
    for key, label in metrics:
        row = f"{label:<{metric_col_w}}"
        for name in policy_names:
            vals = [ep[key] for ep in results_dict[name]]
            mean = np.mean(vals)
            std = np.std(vals)
            cell = f"{mean:8.2f} +/- {std:6.2f}"
            row += f"{cell:^{policy_col_w}}"
        print(row)

    # Fault rate row
    row = f"{'Fault Rate':<{metric_col_w}}"
    for name in policy_names:
        faults = [1 if ep["fault"] else 0 for ep in results_dict[name]]
        rate = np.mean(faults) * 100.0
        cell = f"{rate:5.1f}%"
        row += f"{cell:^{policy_col_w}}"
    print(row)

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAC agent against random and zero-action baselines.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/sac_prusa/best_model.zip",
        help="Path to the trained SB3 SAC model (default: checkpoints/sac_prusa/best_model.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per policy (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    # ----- Environment factory --------------------------------------------
    def make_env() -> PrusaCoreOneEnv:
        env = PrusaCoreOneEnv()
        env = ClipAction(env)
        return env

    # ----- Try loading the trained model ----------------------------------
    trained_model = None
    try:
        from stable_baselines3 import SAC

        model_path = Path(args.model_path)
        if model_path.exists():
            trained_model = SAC.load(str(model_path))
            print(f"Loaded trained model from {model_path}")
        else:
            print(f"WARNING: Model file not found at {model_path} -- skipping trained agent.")
    except ImportError:
        print("WARNING: stable-baselines3 is not installed -- skipping trained agent.")
        print("  Install with:  pip install stable-baselines3")

    # ----- Define policies ------------------------------------------------
    policies: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    if trained_model is not None:
        def _trained_policy(obs: np.ndarray) -> np.ndarray:
            action, _ = trained_model.predict(obs, deterministic=True)
            return action

        policies["SAC (trained)"] = _trained_policy

    # Random policy: sample from action space each step.
    # We need a reference to the env for action_space.sample(), so we create
    # a closure around the env that will be used at evaluation time.
    _eval_env_ref: list[PrusaCoreOneEnv] = []

    def _random_policy(obs: np.ndarray) -> np.ndarray:
        return _eval_env_ref[0].action_space.sample()

    policies["Random"] = _random_policy

    # Zero-action baseline (all slicer defaults, no RL modification)
    def _zero_policy(obs: np.ndarray) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    policies["Zero-Action"] = _zero_policy

    # ----- Run evaluation -------------------------------------------------
    all_results: dict[str, list[dict]] = {}

    for policy_name, policy_fn in policies.items():
        print(f"\nEvaluating: {policy_name}  ({args.episodes} episodes) ...")
        env = make_env()
        _eval_env_ref.clear()
        _eval_env_ref.append(env)

        results = run_policy(env, policy_fn, args.episodes, seed=args.seed)
        all_results[policy_name] = results

        # Quick per-policy summary
        rewards = [r["reward"] for r in results]
        print(f"  mean reward = {np.mean(rewards):.2f},  std = {np.std(rewards):.2f}")

        env.close()

    # ----- Print comparison -----------------------------------------------
    print_comparison(all_results)


if __name__ == "__main__":
    main()
