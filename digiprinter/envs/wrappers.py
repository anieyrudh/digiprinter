from __future__ import annotations

import gymnasium
import numpy as np


class NormalizeObservation(gymnasium.ObservationWrapper):
    """Wraps a Gymnasium env, normalizes observations to [0, 1] using running mean/std."""

    def __init__(self, env, clip: float = 5.0):
        super().__init__(env)
        self.clip = clip
        obs_shape = env.observation_space.shape
        self._mean = np.zeros(obs_shape, dtype=np.float64)
        self._var = np.ones(obs_shape, dtype=np.float64)
        self._count = 1e-4

    def observation(self, obs):
        # Update running stats using Welford's algorithm
        batch_mean = obs
        batch_var = 0
        batch_count = 1
        delta = batch_mean - self._mean
        total = self._count + batch_count
        self._mean += delta * batch_count / total
        m2 = (
            self._var * self._count
            + batch_var * batch_count
            + delta**2 * self._count * batch_count / total
        )
        self._var = m2 / total
        self._count = total

        normalized = (obs - self._mean) / np.sqrt(self._var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)


class ClipAction(gymnasium.ActionWrapper):
    """Clips actions to the action space bounds."""

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


class RewardScaling(gymnasium.RewardWrapper):
    """Scales reward by a constant factor."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


def make_wrapped_env(
    env_id: str = "PrusaCoreOne-v0",
    normalize_obs: bool = True,
    clip_actions: bool = True,
    reward_scale: float = 1.0,
    **kwargs,
) -> gymnasium.Env:
    env = gymnasium.make(env_id, **kwargs)
    if clip_actions:
        env = ClipAction(env)
    if normalize_obs:
        env = NormalizeObservation(env)
    if reward_scale != 1.0:
        env = RewardScaling(env, reward_scale)
    return env
