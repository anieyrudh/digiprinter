"""Tests for single-agent Gymnasium environment."""
import pytest
import numpy as np


class TestSingleAgentEnv:
    def setup_method(self):
        from digiprinter.envs.single_agent import PrusaCoreOneEnv
        self.env = PrusaCoreOneEnv()

    def teardown_method(self):
        self.env.close()

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (34,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_valid(self):
        self.env.reset(seed=42)
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.shape == (34,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_action_space_valid(self):
        assert self.env.action_space.shape == (6,)
        assert self.env.action_space.dtype == np.float32

    def test_observation_space_valid(self):
        assert self.env.observation_space.shape == (34,)

    def test_random_agent_no_crash(self):
        """Random agent should run without crashing."""
        obs, _ = self.env.reset(seed=42)
        for _ in range(20):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break

    def test_zero_action_reasonable(self):
        """Zero action (slicer defaults) should produce reasonable results."""
        obs, _ = self.env.reset(seed=42)
        action = np.zeros(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Should not crash and reward should be finite
        assert np.isfinite(reward)

    def test_deterministic_reset(self):
        """Same seed should produce same initial observation."""
        obs1, _ = self.env.reset(seed=123)
        obs2, _ = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)
