"""Main simulation engine coupling all physics modules."""

from __future__ import annotations

import numpy as np

from digiprinter.config import PrinterConfig, DEFAULT_CONFIG
from digiprinter.materials.base import MaterialProperties
from digiprinter.materials import PLA
from digiprinter.physics.thermal import ThermalModel
from digiprinter.physics.extrusion import ExtrusionModel
from digiprinter.physics.motion import MotionModel
from digiprinter.physics.cooling import CoolingModel
from digiprinter.physics.quality import QualityModel
from digiprinter.printer.state import PrinterState
from digiprinter.printer.hardware import PIDController
from digiprinter.gcode.parser import GCodeParser
from digiprinter.gcode.interpreter import GCodeInterpreter, SimulationAction
from digiprinter.simulation.time_stepper import AdaptiveTimeStepper
from digiprinter.simulation.events import EventManager


class SimulationEngine:
    """High-level simulation engine that couples G-code interpretation with
    thermal, extrusion, motion, cooling, and quality physics models.

    Typical usage::

        engine = SimulationEngine()
        engine.load_gcode(gcode_text)
        while not engine.done:
            action, info = engine.step_action()
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        config: PrinterConfig = DEFAULT_CONFIG,
        material: MaterialProperties = PLA,
    ) -> None:
        self.config = config
        self.material = material

        # Physics sub-models
        self.thermal = ThermalModel(config)
        self.extrusion = ExtrusionModel(config)
        self.motion = MotionModel(config)
        self.cooling = CoolingModel(config)
        self.quality = QualityModel(config)

        # Printer state
        self.state = PrinterState()

        # PID controllers
        self.hotend_pid = PIDController(
            kp=config.hotend_pid_kp,
            ki=config.hotend_pid_ki,
            kd=config.hotend_pid_kd,
            output_min=0.0,
            output_max=config.max_heater_duty,
        )
        self.bed_pid = PIDController(
            kp=config.hotend_pid_kp,
            ki=config.hotend_pid_ki,
            kd=config.hotend_pid_kd,
            output_min=0.0,
            output_max=config.max_heater_duty,
        )

        # G-code pipeline
        self.parser = GCodeParser()
        self.interpreter = GCodeInterpreter()

        # Adaptive time stepper and event manager
        self.time_stepper = AdaptiveTimeStepper(config)
        self.event_manager = EventManager()

        # Action queue
        self.gcode_actions: list[SimulationAction] = []
        self.action_index: int = 0

        # Cached quality scores — only recomputed after extruding moves
        self._cached_quality_scores: dict[str, float] = {
            "adhesion": 0.0,
            "warping": 0.0,
            "stringing": 0.0,
            "dimensional_error": 0.0,
        }
        self._quality_dirty: bool = False

    # ------------------------------------------------------------------ #
    #  G-code loading                                                     #
    # ------------------------------------------------------------------ #

    def load_gcode(self, gcode_text: str) -> int:
        """Parse and interpret a G-code program.

        Parameters
        ----------
        gcode_text : str
            Full G-code program as a multi-line string.

        Returns
        -------
        int
            Number of executable simulation actions produced.
        """
        commands = self.parser.parse_file(gcode_text)
        self.gcode_actions = self.interpreter.interpret_program(commands)
        self.action_index = 0
        self.state.total_gcode_lines = len(self.gcode_actions)
        return len(self.gcode_actions)

    # ------------------------------------------------------------------ #
    #  Step: execute one G-code action                                    #
    # ------------------------------------------------------------------ #

    def step_action(self) -> tuple[SimulationAction | None, dict]:
        """Execute the next G-code action and advance the simulation.

        Returns
        -------
        tuple[SimulationAction | None, dict]
            The action that was executed (or ``None`` if finished) and an
            info dictionary containing quality scores, temperatures, and
            other diagnostics.
        """
        if self.action_index >= len(self.gcode_actions):
            return None, {}

        action = self.gcode_actions[self.action_index]
        info: dict = {}
        dt = self.time_stepper.dt

        # ----- Dispatch by action type --------------------------------
        if action.action_type in ("set_hotend_temp", "wait_hotend_temp"):
            if action.temperature is not None:
                self.state.hotend_target = action.temperature

            if action.action_type == "wait_hotend_temp":
                self._wait_for_temp("hotend")

        elif action.action_type in ("set_bed_temp", "wait_bed_temp"):
            if action.temperature is not None:
                self.state.bed_target = action.temperature

            if action.action_type == "wait_bed_temp":
                self._wait_for_temp("bed")

        elif action.action_type == "set_fan":
            if action.fan_speed is not None:
                self.state.fan_target = action.fan_speed
                self.cooling.fan_target = action.fan_speed

        elif action.action_type == "home":
            self.motion.reset()
            self.state.x = 0.0
            self.state.y = 0.0
            self.state.z = 0.0
            self.state.vx = 0.0
            self.state.vy = 0.0
            self.state.vz = 0.0
            self.state.is_homing = True

        elif action.action_type == "move":
            self._execute_move(action, extrude=True)
            self._quality_dirty = True

        elif action.action_type == "travel":
            # Compute stringing/ooze during travel moves
            x_t = action.x if action.x is not None else self.state.x
            y_t = action.y if action.y is not None else self.state.y
            travel_dist = np.sqrt(
                (x_t - self.state.x) ** 2 + (y_t - self.state.y) ** 2
            )
            if travel_dist > 0.1:
                retract_frac = 1.0 if self.state.retracted else 0.0
                string_len = self.quality.compute_stringing(
                    self.state.viscosity, travel_dist,
                    self.material, retract_frac,
                )
                self.state.stringing_amount = self.quality.total_stringing
                self._quality_dirty = True
            self._execute_move(action, extrude=False)

        elif action.action_type == "retract":
            retract_dist = abs(action.e) if action.e is not None else self.config.default_retraction_dist
            retract_speed = action.feedrate if action.feedrate is not None else self.config.default_retraction_speed
            retract_time = self.extrusion.retract(retract_dist, retract_speed)
            self.state.retracted = True
            self.state.retraction_amount = retract_dist
            # Step physics for the retraction duration
            if retract_time > 0.0:
                self.step_physics(retract_time)

        elif action.action_type == "unretract":
            unretract_speed = action.feedrate if action.feedrate is not None else self.config.default_retraction_speed
            unretract_time = self.extrusion.unretract(unretract_speed)
            self.state.retracted = False
            self.state.retraction_amount = 0.0
            if unretract_time > 0.0:
                self.step_physics(unretract_time)

        # Update bookkeeping
        self.action_index += 1
        self.state.gcode_line = self.action_index

        # Check for fault events
        new_faults = self.event_manager.check_events(self.state.sim_time)
        for fault in new_faults:
            self.event_manager.apply_fault(fault, self.state)
        # Apply ongoing active faults
        for fault in self.event_manager.active_faults:
            if fault not in new_faults:
                self.event_manager.apply_fault(fault, self.state)

        info = self.get_info()
        return action, info

    # ------------------------------------------------------------------ #
    #  Internal: move execution                                           #
    # ------------------------------------------------------------------ #

    def _execute_move(self, action: SimulationAction, extrude: bool) -> None:
        """Plan and execute a move, stepping physics for each waypoint.

        For moves with many waypoints the full physics update (PID, thermal,
        extrusion, cooling) is batched: it runs every *N*-th waypoint while
        the intermediate waypoints only update the kinematic position and
        accumulate time.  This preserves accuracy for short moves while
        greatly reducing computation for long, straight segments.
        """
        dt = self.time_stepper.dt

        # Resolve target position (keep current if axis not specified)
        x_target = action.x if action.x is not None else self.motion.x
        y_target = action.y if action.y is not None else self.motion.y
        z_target = action.z if action.z is not None else self.motion.z
        feedrate = action.feedrate if action.feedrate is not None else self.config.default_print_speed

        # Plan the move and get waypoints
        waypoints = self.motion.execute_move(
            x_target, y_target, z_target, feedrate, dt,
        )

        if not waypoints:
            return

        n_waypoints = len(waypoints)

        # Determine batching stride: step physics every *stride* waypoints.
        # Short moves (< 20 waypoints) keep per-waypoint physics; longer
        # moves batch more aggressively (up to every 10th waypoint).
        if n_waypoints < 20:
            stride = 1
        elif n_waypoints < 100:
            stride = 5
        else:
            stride = 10

        accumulated_dt = 0.0  # time accumulated since last physics step

        # Process each waypoint
        for i, (wx, wy, wz, speed) in enumerate(waypoints):
            # Update position in state (always — this is cheap)
            self.state.x = wx
            self.state.y = wy
            self.state.z = wz
            self.state.current_speed = speed

            accumulated_dt += dt

            # Full physics update on stride boundaries and the final waypoint
            is_physics_step = (i % stride == 0) or (i == n_waypoints - 1)

            if is_physics_step:
                step_dt = accumulated_dt  # cover the whole accumulated interval

                # PID update
                hotend_duty = self.hotend_pid.update(
                    self.state.hotend_target, self.state.hotend_temp, step_dt,
                )
                bed_duty = self.bed_pid.update(
                    self.state.bed_target, self.state.bed_temp, step_dt,
                )
                self.state.hotend_duty = hotend_duty
                self.state.bed_duty = bed_duty

                # Thermal step
                mass_flow = self.state.mass_flow_rate if extrude else 0.0
                self.thermal.step(
                    step_dt,
                    hotend_duty,
                    bed_duty,
                    self.cooling.fan_speed,
                    mass_flow,
                    self.state.ambient_temp,
                    self.state.ambient_temp,
                )

                # Sync thermal state back
                self.state.hotend_temp = self.thermal.hotend_temp
                self.state.bed_temp = self.thermal.bed_temp
                self.state.chamber_temp = self.thermal.chamber_temp

                # Extrusion flow (only for extrusion moves)
                if extrude and speed > 0.0:
                    layer_height = self.config.default_layer_height
                    line_width = self.config.nozzle_diameter
                    flow = self.extrusion.compute_flow(
                        speed, layer_height, line_width,
                        self.material, self.state.hotend_temp,
                    )
                    self.state.flow_rate = flow["volume_flow_mm3s"]
                    self.state.mass_flow_rate = flow["mass_flow_rate"]
                    self.state.viscosity = flow["viscosity"]
                    self.state.pressure_drop = flow["pressure_drop"]
                    self.state.die_swell = flow["die_swell"]

                    # Quality: dimensional accuracy
                    self.quality.compute_dimensional_accuracy(
                        line_width, flow["actual_line_width"],
                    )

                    # Quality: adhesion (inter-layer bonding)
                    # Estimate layer time as time since last Z change.
                    # The interface temp depends on how long the previous
                    # layer has been cooling — approximate as total sim
                    # time divided by layers (capped for realism).
                    layers = max(self.state.current_layer, 1)
                    layer_time = max(self.state.sim_time / layers, 0.5)
                    layer_time = min(layer_time, 30.0)  # cap at 30s
                    interface_temp = ThermalModel.interface_temperature(
                        self.state.hotend_temp,
                        self.state.chamber_temp,
                        layer_time,
                        self.material.thermal_diffusivity,
                    )
                    adhesion = self.quality.compute_adhesion(
                        interface_temp, layer_time, self.material,
                        self.state.bed_adhesion_factor,
                    )
                    self.state.adhesion_quality = adhesion

                    # Quality: warping (per-segment footprint, not full bed)
                    delta_t = self.state.hotend_temp - self.state.ambient_temp
                    # Segment footprint: line_width * segment_length
                    seg_len = speed * step_dt  # mm traveled this segment
                    seg_footprint = line_width * max(seg_len, 0.1)  # mm²
                    warp = self.quality.compute_warping(
                        delta_t, seg_footprint, self.material, adhesion,
                    )
                    self.state.warping_amount = self.quality.total_warping

                # Cooling update
                self.cooling.update_fan(step_dt, self.state.fan_target)
                self.state.fan_speed = self.cooling.fan_speed

                # Energy tracking
                self.state.hotend_power = self.thermal.hotend_power
                self.state.bed_power = self.thermal.bed_power
                self.state.total_energy_j += (
                    self.thermal.hotend_power + self.thermal.bed_power
                ) * step_dt

                accumulated_dt = 0.0  # reset accumulator

            # Advance simulation clock (always, per-waypoint)
            self.state.sim_time += dt
            self.state.step_count += 1

        self.state.is_printing = extrude

    # ------------------------------------------------------------------ #
    #  Internal: wait for temperature                                     #
    # ------------------------------------------------------------------ #

    def _wait_for_temp(self, heater: str) -> None:
        """Simulate until the specified heater is within 2 deg C of target.

        Uses a fast-forward strategy: a coarse time step (0.1 s) while
        the temperature is far from the target (> 5 deg C away), switching
        to the normal fine time step once close.  This dramatically reduces
        the number of physics iterations during long thermal equilibration
        phases (M109 / M190).  A 120 s safety cap prevents infinite loops.
        """
        fine_dt = self.time_stepper.dt
        coarse_dt = 0.1  # 100x larger than typical 0.001 s fine step
        close_threshold = 5.0  # switch to fine dt within this range
        tolerance = 2.0  # target reached when within this range
        max_wait_time = 120.0  # seconds of simulated time before giving up
        elapsed = 0.0

        while elapsed < max_wait_time:
            # Read the relevant temperature and target
            if heater == "hotend":
                current = self.state.hotend_temp
                target = self.state.hotend_target
            else:
                current = self.state.bed_temp
                target = self.state.bed_target

            temp_error = abs(current - target)

            # Check convergence
            if temp_error <= tolerance:
                break

            # Choose time step based on distance from target
            dt = fine_dt if temp_error <= close_threshold else coarse_dt

            self.step_physics(dt)
            elapsed += dt

    # ------------------------------------------------------------------ #
    #  Single physics time step (no G-code advance)                      #
    # ------------------------------------------------------------------ #

    def step_physics(self, dt: float) -> None:
        """Advance the physics by a single time step without consuming a
        G-code action.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        # PID updates
        hotend_duty = self.hotend_pid.update(
            self.state.hotend_target, self.state.hotend_temp, dt,
        )
        bed_duty = self.bed_pid.update(
            self.state.bed_target, self.state.bed_temp, dt,
        )
        self.state.hotend_duty = hotend_duty
        self.state.bed_duty = bed_duty

        # Thermal step
        self.thermal.step(
            dt,
            hotend_duty,
            bed_duty,
            self.cooling.fan_speed,
            self.state.mass_flow_rate,
            self.state.ambient_temp,
            self.state.ambient_temp,
        )

        # Sync temperatures
        self.state.hotend_temp = self.thermal.hotend_temp
        self.state.bed_temp = self.thermal.bed_temp
        self.state.chamber_temp = self.thermal.chamber_temp

        # Cooling update
        self.cooling.update_fan(dt, self.state.fan_target)
        self.state.fan_speed = self.cooling.fan_speed

        # Energy tracking
        self.state.hotend_power = self.thermal.hotend_power
        self.state.bed_power = self.thermal.bed_power
        self.state.total_energy_j += (
            self.thermal.hotend_power + self.thermal.bed_power
        ) * dt

        # Advance simulation clock
        self.state.sim_time += dt
        self.state.step_count += 1

        # Event check
        new_faults = self.event_manager.check_events(self.state.sim_time)
        for fault in new_faults:
            self.event_manager.apply_fault(fault, self.state)

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        ambient_temp: float = 22.0,
        seed: int | None = None,
    ) -> PrinterState:
        """Reset the entire simulation to its initial state.

        Parameters
        ----------
        ambient_temp : float
            Starting ambient / room temperature (deg C).
        seed : int | None
            If provided, re-seed the event manager RNG for reproducibility.

        Returns
        -------
        PrinterState
            The freshly-reset printer state.
        """
        # Sub-model resets
        self.thermal.reset(ambient_temp)
        self.extrusion.reset()
        self.motion.reset()
        self.cooling.reset()
        self.quality.reset()

        # State reset
        self.state.reset(ambient_temp)

        # PID reset
        self.hotend_pid.reset()
        self.bed_pid.reset()

        # Interpreter reset
        self.interpreter.reset()

        # Event manager reset
        self.event_manager.reset()
        if seed is not None:
            self.event_manager._rng = np.random.default_rng(seed)

        # Action queue reset
        self.gcode_actions = []
        self.action_index = 0

        # Quality cache reset
        self._cached_quality_scores = {
            "adhesion": 0.0,
            "warping": 0.0,
            "stringing": 0.0,
            "dimensional_error": 0.0,
        }
        self._quality_dirty = False

        # Time stepper reset
        self.time_stepper.reset()

        return self.state

    # ------------------------------------------------------------------ #
    #  Info / status                                                      #
    # ------------------------------------------------------------------ #

    def get_info(self) -> dict:
        """Return a summary dictionary of the current simulation state.

        Keys include quality scores, temperatures, energy consumption,
        progress fraction, and fault status.

        Quality scores are only recomputed when the last action was an
        extruding move (``_quality_dirty`` flag), avoiding redundant work
        during temperature waits, travels, fan changes, etc.
        """
        total = max(len(self.gcode_actions), 1)
        progress = self.action_index / total

        if self._quality_dirty:
            self._cached_quality_scores = self.quality.get_quality_scores()
            self._quality_dirty = False

        quality_scores = self._cached_quality_scores

        return {
            # Quality
            "adhesion": quality_scores["adhesion"],
            "warping": quality_scores["warping"],
            "stringing": quality_scores["stringing"],
            "dimensional_error": quality_scores["dimensional_error"],
            # Temperatures
            "hotend_temp": self.state.hotend_temp,
            "bed_temp": self.state.bed_temp,
            "chamber_temp": self.state.chamber_temp,
            # Energy
            "total_energy_j": self.state.total_energy_j,
            "hotend_power": self.state.hotend_power,
            "bed_power": self.state.bed_power,
            # Progress
            "progress": progress,
            "action_index": self.action_index,
            "total_actions": len(self.gcode_actions),
            "sim_time": self.state.sim_time,
            "step_count": self.state.step_count,
            # Fault
            "fault": self.state.fault,
            "active_faults": len(self.event_manager.active_faults),
        }

    # ------------------------------------------------------------------ #
    #  Done property                                                      #
    # ------------------------------------------------------------------ #

    @property
    def done(self) -> bool:
        """Return ``True`` when all actions have been consumed or a fault
        has occurred."""
        return (
            self.action_index >= len(self.gcode_actions)
            or self.state.fault != ""
        )
