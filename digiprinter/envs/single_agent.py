"""Single-agent Gymnasium environment for Prusa Core One+ digital twin."""

from __future__ import annotations

import gymnasium
import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG
from digiprinter.materials import PLA
from digiprinter.materials.base import MaterialProperties
from digiprinter.simulation.engine import SimulationEngine
from digiprinter.envs.observations import observation_space, build_observation
from digiprinter.envs.rewards import RewardCalculator
from digiprinter.gcode.library import calibration_cube_gcode

# ---------------------------------------------------------------------------
# Environment registration
# ---------------------------------------------------------------------------
gymnasium.register(
    id="PrusaCoreOne-v0",
    entry_point="digiprinter.envs.single_agent:PrusaCoreOneEnv",
)


class PrusaCoreOneEnv(gymnasium.Env):
    """Gymnasium environment that wraps the Prusa Core One+ simulation engine.

    **Action space** — ``Box(-1, 1, (6,), float32)``

    ======  ==============================  ===================
    Index   Meaning                          Mapped range
    ======  ==============================  ===================
    0       Speed modifier                   0.5x -- 1.5x
    1       Flow modifier                    0.8x -- 1.2x
    2       Hotend temperature offset         -20 -- +20 deg C
    3       Bed temperature offset            -10 -- +10 deg C
    4       Fan override (direct fraction)   0.0  -- 1.0
    5       Retraction modifier              0.5x -- 1.5x
    ======  ==============================  ===================

    **Observation space** — ``Box(0, 1.5, (34,), float32)``
    (see :func:`digiprinter.envs.observations.observation_space`).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

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

        # Simulation back-end
        self.engine = SimulationEngine(self.config, self.material)
        self.reward_calc = RewardCalculator(config=self.config)

        # Internal bookkeeping
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        # ----- Spaces -----------------------------------------------------
        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = observation_space(self.config)

    # ------------------------------------------------------------------ #
    #  reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode.

        Applies domain randomization to several physical parameters so
        that the agent learns a robust policy.
        """
        super().reset(seed=seed)

        # Use the Gymnasium-managed np_random (seeded by super().reset) so
        # that sequential resets without an explicit seed stay deterministic.
        self._rng = np.random.default_rng(self.np_random.integers(2**31))

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

        obs = build_observation(self.engine.state, self.config)
        info = self.engine.get_info()
        return obs, info

    # ------------------------------------------------------------------ #
    #  step                                                               #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one G-code action with RL-modulated print parameters.

        Parameters
        ----------
        action : np.ndarray
            A 6-element float32 array in [-1, 1].

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float32)

        # ----- Map normalised actions to physical modifiers ---------------
        speed_mod = 0.5 + (action[0] + 1.0) * 0.5       # -1->0.5, 1->1.5
        flow_mod = 0.8 + (action[1] + 1.0) * 0.2        # -1->0.8, 1->1.2
        hotend_offset = float(action[2]) * 20.0           # -1->-20, 1->+20
        bed_offset = float(action[3]) * 10.0              # -1->-10, 1->+10
        fan_override = (action[4] + 1.0) * 0.5            # -1->0,   1->1
        retract_mod = 0.5 + (action[5] + 1.0) * 0.5      # -1->0.5, 1->1.5

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
        reward, reward_components = self.reward_calc.compute(
            self.engine.state, action,
        )

        # ----- Termination / truncation -----------------------------------
        self._step_count += 1
        terminated = self.engine.done
        truncated = self._step_count >= self.max_steps

        # ----- Info -------------------------------------------------------
        info = self.engine.get_info()
        info["reward_components"] = reward_components

        return obs, float(reward), bool(terminated), bool(truncated), info

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
            f"hotend={s.hotend_temp:.1f}/{s.hotend_target:.0f}°C  "
            f"bed={s.bed_temp:.1f}/{s.bed_target:.0f}°C  "
            f"fan={s.fan_speed:.0%}  "
            f"speed={s.current_speed:.1f}mm/s  "
            f"fault={'YES' if s.fault else 'no'}"
        )
