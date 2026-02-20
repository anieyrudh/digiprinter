"""Tests for multi-agent PettingZoo environment."""
import pytest
import numpy as np


class TestMultiAgentEnv:
    def setup_method(self):
        from digiprinter.envs.multi_agent import PrusaCoreOneMultiAgentEnv
        self.env = PrusaCoreOneMultiAgentEnv()

    def test_agents_defined(self):
        assert "thermal_agent" in self.env.possible_agents
        assert "motion_agent" in self.env.possible_agents
        assert "extrusion_agent" in self.env.possible_agents

    def test_action_spaces(self):
        assert self.env.action_space("thermal_agent").shape == (4,)
        assert self.env.action_space("motion_agent").shape == (2,)
        assert self.env.action_space("extrusion_agent").shape == (2,)

    def test_observation_spaces(self):
        for agent in self.env.possible_agents:
            assert self.env.observation_space(agent).shape == (34,)

    def test_reset(self):
        observations, infos = self.env.reset(seed=42)
        assert set(observations.keys()) == set(self.env.possible_agents)
        for agent in self.env.possible_agents:
            assert observations[agent].shape == (34,)

    def test_step(self):
        self.env.reset(seed=42)
        actions = {
            "thermal_agent": self.env.action_space("thermal_agent").sample(),
            "motion_agent": self.env.action_space("motion_agent").sample(),
            "extrusion_agent": self.env.action_space("extrusion_agent").sample(),
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        for agent in self.env.possible_agents:
            assert agent in obs
            assert agent in rewards
            assert isinstance(rewards[agent], (int, float))

    def test_random_agents_no_crash(self):
        self.env.reset(seed=42)
        for _ in range(10):
            if not self.env.agents:
                break
            actions = {
                agent: self.env.action_space(agent).sample()
                for agent in self.env.agents
            }
            obs, rewards, terms, truncs, infos = self.env.step(actions)
