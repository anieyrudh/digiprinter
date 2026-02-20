"""G-code interpreter.

Converts parsed GCodeCommand objects into high-level SimulationAction
objects that the physics / simulation layer can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from digiprinter.gcode.parser import GCodeCommand


@dataclass
class SimulationAction:
    """A single action the simulation engine should execute."""

    action_type: str
    # "move", "travel", "set_hotend_temp", "wait_hotend_temp",
    # "set_bed_temp", "wait_bed_temp", "set_fan", "retract",
    # "unretract", "home", "set_absolute", "set_relative", "noop"

    x: float | None = None
    y: float | None = None
    z: float | None = None
    e: float | None = None
    feedrate: float | None = None  # mm/s (converted from G-code F in mm/min)
    temperature: float | None = None
    fan_speed: float | None = None  # 0.0 - 1.0 (from S 0-255)


class GCodeInterpreter:
    """Stateful interpreter that walks a sequence of GCodeCommands and
    produces SimulationActions.

    The interpreter tracks modal state (absolute vs. relative mode,
    current feedrate, extruder position) so that each command can be
    resolved to a concrete action.
    """

    def __init__(self) -> None:
        self.absolute_mode: bool = True  # G90 / G91
        self.absolute_xyz: bool = True
        self.current_feedrate: float = 60.0  # mm/s default
        self.last_e: float = 0.0
        self.retracted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(self, cmd: GCodeCommand) -> SimulationAction:
        """Interpret a single G-code command and return a SimulationAction."""

        handler = self._DISPATCH.get(cmd.command)
        if handler is not None:
            return handler(self, cmd)
        return SimulationAction(action_type="noop")

    def interpret_program(
        self, commands: list[GCodeCommand]
    ) -> list[SimulationAction]:
        """Interpret a full program, filtering out noops.

        Parameters
        ----------
        commands:
            Ordered list of GCodeCommand objects (e.g. from GCodeParser.parse_file).

        Returns
        -------
        List of SimulationAction objects with all noops removed.
        """
        actions: list[SimulationAction] = []
        for cmd in commands:
            action = self.interpret(cmd)
            if action.action_type != "noop":
                actions.append(action)
        return actions

    def reset(self) -> None:
        """Reset all interpreter state to defaults."""
        self.absolute_mode = True
        self.absolute_xyz = True
        self.current_feedrate = 60.0
        self.last_e = 0.0
        self.retracted = False

    # ------------------------------------------------------------------
    # Command handlers (private)
    # ------------------------------------------------------------------

    def _handle_move(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle G0 and G1 (linear move / travel)."""
        params = cmd.params

        # Update feedrate if provided (F is in mm/min, convert to mm/s)
        if "F" in params:
            self.current_feedrate = params["F"] / 60.0

        x = params.get("X")
        y = params.get("Y")
        z = params.get("Z")
        e_param = params.get("E")

        feedrate = self.current_feedrate

        # Determine extrusion delta
        if e_param is not None:
            if self.absolute_mode:
                e_delta = e_param - self.last_e
                self.last_e = e_param
            else:
                e_delta = e_param
                self.last_e += e_param
        else:
            e_delta = 0.0

        has_xyz = x is not None or y is not None or z is not None

        # Pure retraction / unretraction (no XYZ movement)
        if not has_xyz and e_param is not None:
            if e_delta < 0.0:
                self.retracted = True
                return SimulationAction(
                    action_type="retract",
                    e=e_delta,
                    feedrate=feedrate,
                )
            elif e_delta > 0.0 and self.retracted:
                self.retracted = False
                return SimulationAction(
                    action_type="unretract",
                    e=e_delta,
                    feedrate=feedrate,
                )

        # Movement with extrusion
        if has_xyz and e_delta > 0.0:
            return SimulationAction(
                action_type="move",
                x=x,
                y=y,
                z=z,
                e=e_delta,
                feedrate=feedrate,
            )

        # Travel (no extrusion or negative E)
        return SimulationAction(
            action_type="travel",
            x=x,
            y=y,
            z=z,
            e=e_delta if e_delta != 0.0 else None,
            feedrate=feedrate,
        )

    def _handle_home(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle G28 (home axes)."""
        self.last_e = 0.0
        return SimulationAction(action_type="home")

    def _handle_g90(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle G90 (absolute positioning)."""
        self.absolute_xyz = True
        self.absolute_mode = True
        return SimulationAction(action_type="set_absolute")

    def _handle_g91(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle G91 (relative positioning)."""
        self.absolute_xyz = False
        self.absolute_mode = False
        return SimulationAction(action_type="set_relative")

    def _handle_m82(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M82 (absolute extruder)."""
        self.absolute_mode = True
        return SimulationAction(action_type="set_absolute")

    def _handle_m83(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M83 (relative extruder)."""
        self.absolute_mode = False
        return SimulationAction(action_type="set_relative")

    def _handle_m104(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M104 (set hotend temperature, no wait)."""
        temp = cmd.params.get("S", 0.0)
        return SimulationAction(action_type="set_hotend_temp", temperature=temp)

    def _handle_m109(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M109 (set hotend temperature and wait)."""
        temp = cmd.params.get("S", 0.0)
        return SimulationAction(action_type="wait_hotend_temp", temperature=temp)

    def _handle_m140(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M140 (set bed temperature, no wait)."""
        temp = cmd.params.get("S", 0.0)
        return SimulationAction(action_type="set_bed_temp", temperature=temp)

    def _handle_m190(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M190 (set bed temperature and wait)."""
        temp = cmd.params.get("S", 0.0)
        return SimulationAction(action_type="wait_bed_temp", temperature=temp)

    def _handle_m106(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M106 (set fan speed, S 0-255 -> 0.0-1.0)."""
        s_val = cmd.params.get("S", 255.0)
        fan_speed = max(0.0, min(1.0, s_val / 255.0))
        return SimulationAction(action_type="set_fan", fan_speed=fan_speed)

    def _handle_m107(self, cmd: GCodeCommand) -> SimulationAction:
        """Handle M107 (fan off)."""
        return SimulationAction(action_type="set_fan", fan_speed=0.0)

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    _DISPATCH: dict[str, callable] = {
        "G0": _handle_move,
        "G1": _handle_move,
        "G28": _handle_home,
        "G90": _handle_g90,
        "G91": _handle_g91,
        "M82": _handle_m82,
        "M83": _handle_m83,
        "M104": _handle_m104,
        "M109": _handle_m109,
        "M140": _handle_m140,
        "M190": _handle_m190,
        "M106": _handle_m106,
        "M107": _handle_m107,
    }
