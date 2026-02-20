"""Tests for G-code parser and interpreter."""
import pytest
from digiprinter.gcode.parser import GCodeParser, GCodeCommand
from digiprinter.gcode.interpreter import GCodeInterpreter, SimulationAction
from digiprinter.gcode.library import calibration_cube_gcode, single_line_gcode


class TestGCodeParser:
    def setup_method(self):
        self.parser = GCodeParser()

    def test_parse_g1_move(self):
        cmd = self.parser.parse_line("G1 X10.5 Y20.0 E0.5 F1200", 1)
        assert cmd is not None
        assert cmd.command == "G1"
        assert cmd.params["X"] == 10.5
        assert cmd.params["Y"] == 20.0
        assert cmd.params["E"] == 0.5
        assert cmd.params["F"] == 1200.0

    def test_parse_m104(self):
        cmd = self.parser.parse_line("M104 S210", 1)
        assert cmd.command == "M104"
        assert cmd.params["S"] == 210.0

    def test_parse_comment_only(self):
        cmd = self.parser.parse_line("; this is a comment", 1)
        assert cmd is None

    def test_parse_empty_line(self):
        cmd = self.parser.parse_line("", 1)
        assert cmd is None

    def test_parse_inline_comment(self):
        cmd = self.parser.parse_line("G1 X10 ; move", 1)
        assert cmd is not None
        assert cmd.params["X"] == 10.0
        assert "move" in cmd.comment

    def test_parse_file(self):
        gcode = "G28\nM104 S210\nG1 X10 Y20 F1200\n"
        commands = self.parser.parse_file(gcode)
        assert len(commands) == 3


class TestGCodeInterpreter:
    def setup_method(self):
        self.interp = GCodeInterpreter()
        self.parser = GCodeParser()

    def _interpret(self, line: str) -> SimulationAction:
        cmd = self.parser.parse_line(line, 0)
        return self.interp.interpret(cmd)

    def test_g1_move_with_extrusion(self):
        action = self._interpret("G1 X10 Y10 E1.0 F1200")
        assert action.action_type == "move"
        assert action.x == 10.0
        assert action.feedrate == 20.0  # 1200/60

    def test_g0_travel(self):
        action = self._interpret("G0 X50 Y50 F3000")
        assert action.action_type == "travel"

    def test_m104_set_temp(self):
        action = self._interpret("M104 S210")
        assert action.action_type == "set_hotend_temp"
        assert action.temperature == 210.0

    def test_m109_wait_temp(self):
        action = self._interpret("M109 S210")
        assert action.action_type == "wait_hotend_temp"

    def test_m106_fan(self):
        action = self._interpret("M106 S128")
        assert action.action_type == "set_fan"
        assert abs(action.fan_speed - 128/255) < 0.01

    def test_m107_fan_off(self):
        action = self._interpret("M107")
        assert action.action_type == "set_fan"
        assert action.fan_speed == 0.0

    def test_home(self):
        action = self._interpret("G28")
        assert action.action_type == "home"

    def test_retraction_detection(self):
        # First extrude forward
        self._interpret("G1 E5.0 F1200")
        # Then retract
        action = self._interpret("G1 E4.0 F1200")
        assert action.action_type == "retract"

    def test_interpret_program(self):
        gcode = "G28\nM104 S210\nG1 X10 Y10 E1 F1200\n"
        commands = self.parser.parse_file(gcode)
        actions = self.interp.interpret_program(commands)
        assert len(actions) >= 2  # home + set_temp + move (noops filtered)


class TestGCodeLibrary:
    def test_calibration_cube_generates_gcode(self):
        gcode = calibration_cube_gcode()
        assert len(gcode) > 100
        assert "G28" in gcode
        assert "M104" in gcode

    def test_single_line_generates_gcode(self):
        gcode = single_line_gcode()
        assert len(gcode) > 50
        assert "G1" in gcode

    def test_calibration_cube_parseable(self):
        parser = GCodeParser()
        gcode = calibration_cube_gcode()
        commands = parser.parse_file(gcode)
        assert len(commands) > 10
