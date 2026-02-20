"""Multi-agent PettingZoo ParallelEnv for Prusa Core One+ cooperative RL."""

from __future__ import annotations

import functools

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG
from digiprinter.materials import PLA
from digiprinter.materials.base import MaterialProperties
from digiprinter.simulation.engine import SimulationEngine
from digiprinter.envs.observations import build_observation, observation_space
from digiprinter.envs.rewards import RewardCalculator, RewardWeights
from digiprinter.gcode.library import calibration_cube_gcode


class PrusaCoreOneMultiAgentEnv(ParallelEnv):
    """PettingZoo ParallelEnv with three cooperative agents controlling
    different subsystems of the Prusa Core One+ digital twin.

    Agents
    ------
    ``thermal_agent``
        Controls hotend temperature offset, bed temperature offset,
        fan override, and vent control.  Action shape: ``(4,)``.

    ``motion_agent``
        Controls speed and acceleration modifiers.
        Action shape: ``(2,)``.

    ``extrusion_agent``
        Controls flow and retraction modifiers.
        Action shape: ``(2,)``.

    All agents share a common 34-dimensional observation vector and
    receive a shared cooperative reward with small per-agent shaping
    bonuses that encourage each agent to attend to its own subsystem.
    """

    metadata = {"render_modes": ["human"], "name": "PrusaCoreOneMultiAgent-v0"}

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        config: PrinterConfig | None = None,
        material: MaterialProperties | None = None,
        gcode: str | None = None,
        render_mode: str | None = None,
        max_steps: int = 50_000,
    ) -> None:
        super().__init__()

        self.config: PrinterConfig = config or DEFAULT_CONFIG
        self.material: MaterialProperties = material or PLA
        self.gcode_text: str = gcode or calibration_cube_gcode()
        self.max_steps: int = max_steps
        self.render_mode: str | None = render_mode

        self.possible_agents: list[str] = [
            "thermal_agent",
            "motion_agent",
            "extrusion_agent",
        ]
        self.agents: list[str] = list(self.possible_agents)

        # Simulation back-end
        self.engine = SimulationEngine(self.config, self.material)
        self.reward_calc = RewardCalculator(config=self.config)

        # Internal bookkeeping
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------ #
    #  Spaces                                                             #
    # ------------------------------------------------------------------ #

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.spaces.Box:
        """Return the shared 34-dim observation space (identical for all agents)."""
        return observation_space(self.config)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.spaces.Box:
        """Return the agent-specific action space.

        - thermal_agent:   Box(-1, 1, (4,)) — hotend_offset, bed_offset,
                           fan_override, vent_control
        - motion_agent:    Box(-1, 1, (2,)) — speed_modifier, accel_modifier
        - extrusion_agent: Box(-1, 1, (2,)) — flow_modifier, retraction_modifier
        """
        if agent == "thermal_agent":
            return gymnasium.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        elif agent == "motion_agent":
            return gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        elif agent == "extrusion_agent":
            return gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown agent: {agent!r}")

    # ------------------------------------------------------------------ #
    #  reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment for a new episode.

        Applies domain randomization to several physical parameters so
        that agents learn a robust cooperative policy.

        Returns
        -------
        observations : dict[str, np.ndarray]
            Mapping from agent name to the shared observation vector.
        infos : dict[str, dict]
            Mapping from agent name to the info dictionary.
        """
        # Seed the RNG
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # Pick a random ambient temperature from the configured range
        ambient_temp = float(
            self._rng.uniform(*self.config.ambient_temp_range)
        )

        # Derive a deterministic sub-seed for the engine's event manager
        engine_seed = int(self._rng.integers(2**31))

        # Full engine reset (clears gcode, state, sub-models)
        self.engine.reset(ambient_temp=ambient_temp, seed=engine_seed)

        # ----- Domain randomization --------------------------------------
        state = self.engine.state
        state.filament_diameter_actual = (
            self.config.filament_diameter
            + self._rng.normal(0.0, self.config.filament_diameter_std)
        )
        state.nozzle_wear = float(
            self._rng.uniform(*self.config.nozzle_wear_range)
        )
        state.bed_adhesion_factor = float(
            self._rng.uniform(*self.config.bed_adhesion_range)
        )

        # Thermal-model randomization
        self.engine.thermal.heater_efficiency = float(
            self._rng.uniform(*self.config.heater_efficiency_range)
        )
        self.engine.thermal.thermistor_offset = float(
            self._rng.uniform(*self.config.thermistor_offset_range)
        )

        # Load the G-code program into the engine
        self.engine.load_gcode(self.gcode_text)

        # Reset the reward calculator (clears history)
        self.reward_calc.reset()

        self._step_count = 0
        self.agents = list(self.possible_agents)

        obs = build_observation(self.engine.state, self.config)
        info = self.engine.get_info()

        observations = {agent: obs.copy() for agent in self.agents}
        infos = {agent: dict(info) for agent in self.agents}

        return observations, infos

    # ------------------------------------------------------------------ #
    #  step                                                               #
    # ------------------------------------------------------------------ #

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Execute one G-code action with RL-modulated print parameters.

        Each agent contributes its own subset of control signals, which
        are combined and applied to the simulation engine.

        Parameters
        ----------
        actions : dict[str, np.ndarray]
            Mapping from agent name to its action array.

        Returns
        -------
        observations, rewards, terminations, truncations, infos
        """
        # ----- Decode per-agent actions -----------------------------------
        # Thermal agent: shape (4,)
        thermal_action = np.asarray(
            actions.get("thermal_agent", np.zeros(4, dtype=np.float32)),
            dtype=np.float32,
        )
        hotend_offset = float(thermal_action[0]) * 20.0      # -1->-20, 1->+20
        bed_offset = float(thermal_action[1]) * 10.0          # -1->-10, 1->+10
        fan_override = 0.5 * (float(thermal_action[2]) + 1.0) # -1->0,   1->1
        vent_control = 0.5 * (float(thermal_action[3]) + 1.0) # -1->0,   1->1

        # Motion agent: shape (2,)
        motion_action = np.asarray(
            actions.get("motion_agent", np.zeros(2, dtype=np.float32)),
            dtype=np.float32,
        )
        speed_mod = 0.5 + (float(motion_action[0]) + 1.0) * 0.5  # -1->0.5, 1->1.5
        accel_mod = 0.5 + (float(motion_action[1]) + 1.0) * 0.5  # -1->0.5, 1->1.5

        # Extrusion agent: shape (2,)
        extrusion_action = np.asarray(
            actions.get("extrusion_agent", np.zeros(2, dtype=np.float32)),
            dtype=np.float32,
        )
        flow_mod = 0.8 + (float(extrusion_action[0]) + 1.0) * 0.2     # -1->0.8, 1->1.2
        retract_mod = 0.5 + (float(extrusion_action[1]) + 1.0) * 0.5  # -1->0.5, 1->1.5

        # ----- Apply modifiers to engine state ----------------------------
        state = self.engine.state
        mat = self.material

        # Hotend target: offset clamped to material processing window
        new_hotend = state.hotend_target + hotend_offset
        new_hotend = float(np.clip(
            new_hotend,
            mat.nozzle_temp_range[0],
            mat.nozzle_temp_range[1],
        ))
        state.hotend_target = new_hotend

        # Bed target: offset clamped to material processing window
        new_bed = state.bed_target + bed_offset
        new_bed = float(np.clip(
            new_bed,
            mat.bed_temp_range[0],
            mat.bed_temp_range[1],
        ))
        state.bed_target = new_bed

        # Fan override applied to cooling model
        fan_value = float(np.clip(fan_override, 0.0, 1.0))
        state.fan_target = fan_value
        self.engine.cooling.fan_target = fan_value

        # Vent control applied to chamber ventilation.
        # The thermal model reads chamber_vent_flow from its config each
        # step, so we swap in a config copy with the RL-controlled value.
        vent_value = float(np.clip(vent_control, 0.0, 1.0))
        vent_config = PrinterConfig(
            **{
                **{f.name: getattr(self.config, f.name)
                   for f in self.config.__dataclass_fields__.values()},
                "chamber_vent_flow": vent_value * 0.01,  # scale to m^3/s
            }
        )
        self.engine.thermal.config = vent_config

        # Patch the next queued SimulationAction in-place so the engine
        # sees the RL-modified values for speed, flow, and retraction.
        if self.engine.action_index < len(self.engine.gcode_actions):
            next_action = self.engine.gcode_actions[self.engine.action_index]

            # Scale feedrate (speed modifier)
            if next_action.feedrate is not None:
                next_action.feedrate = next_action.feedrate * float(speed_mod)

            # Scale extrusion amount (flow modifier)
            if next_action.e is not None:
                next_action.e = next_action.e * float(flow_mod)

            # Scale retraction distance (retraction modifier)
            if next_action.action_type in ("retract", "unretract"):
                if next_action.e is not None:
                    next_action.e = next_action.e * float(retract_mod)

        # ----- Execute one G-code action ----------------------------------
        _action_obj, _action_info = self.engine.step_action()

        # ----- Observation ------------------------------------------------
        obs = build_observation(self.engine.state, self.config)

        # ----- Reward -----------------------------------------------------
        # Build a combined 6-element action vector for the reward calculator
        combined_action = np.array([
            motion_action[0],      # speed
            extrusion_action[0],   # flow
            thermal_action[0],     # hotend offset
            thermal_action[1],     # bed offset
            thermal_action[2],     # fan
            extrusion_action[1],   # retraction
        ], dtype=np.float32)

        shared_reward, reward_components = self.reward_calc.compute(
            self.engine.state, combined_action,
        )

        # Per-agent shaping bonuses (small signals to guide specialisation)
        thermal_bonus = 0.01 * reward_components.get("thermal_stability", 0.0)
        motion_bonus = 0.01 * reward_components.get("speed", 0.0)
        extrusion_bonus = 0.01 * reward_components.get("dimensional_accuracy", 0.0)

        agent_bonuses = {
            "thermal_agent": thermal_bonus,
            "motion_agent": motion_bonus,
            "extrusion_agent": extrusion_bonus,
        }

        # ----- Termination / truncation -----------------------------------
        self._step_count += 1
        terminated = self.engine.done
        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            self.agents = []

        # ----- Build per-agent return dicts (keyed on possible_agents) ----
        info = self.engine.get_info()
        info["reward_components"] = reward_components

        observations = {agent: obs.copy() for agent in self.possible_agents}
        rewards = {
            agent: float(shared_reward) + agent_bonuses[agent]
            for agent in self.possible_agents
        }
        terminations = {agent: bool(terminated) for agent in self.possible_agents}
        truncations = {agent: bool(truncated) for agent in self.possible_agents}
        infos = {agent: dict(info) for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    #  render                                                             #
    # ------------------------------------------------------------------ #

    def render(self) -> None:
        """Print a one-line status summary when ``render_mode='human'``."""
        if self.render_mode != "human":
            return

        s = self.engine.state
        info = self.engine.get_info()
        print(
            f"step={self._step_count:>6d}  "
            f"progress={info['progress']:.1%}  "
            f"hotend={s.hotend_temp:.1f}/{s.hotend_target:.0f}\u00b0C  "
            f"bed={s.bed_temp:.1f}/{s.bed_target:.0f}\u00b0C  "
            f"fan={s.fan_speed:.0%}  "
            f"speed={s.current_speed:.1f}mm/s  "
            f"fault={'YES' if s.fault else 'no'}"
        )
