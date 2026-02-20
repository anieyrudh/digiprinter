"""Tests for motion physics model."""
import pytest
import numpy as np
from digiprinter.physics.motion import MotionModel
from digiprinter.config import DEFAULT_CONFIG


class TestMotionModel:
    def setup_method(self):
        self.model = MotionModel()

    def test_forward_inverse_roundtrip(self):
        """Forward and inverse kinematics should be exact inverses."""
        for x, y in [(0, 0), (10, 20), (125, 105), (-5, 3)]:
            a, b = MotionModel.inverse_kinematics(x, y)
            x2, y2 = MotionModel.forward_kinematics(a, b)
            assert abs(x2 - x) < 1e-10
            assert abs(y2 - y) < 1e-10

    def test_inverse_forward_roundtrip(self):
        for a, b in [(30, 10), (0, 0), (100, -50)]:
            x, y = MotionModel.forward_kinematics(a, b)
            a2, b2 = MotionModel.inverse_kinematics(x, y)
            assert abs(a2 - a) < 1e-10
            assert abs(b2 - b) < 1e-10

    def test_plan_move_reaches_target(self):
        waypoints = self.model.plan_move(10.0, 10.0, 0.0, 100.0, 0.001)
        assert len(waypoints) > 0
        last = waypoints[-1]
        assert abs(last[0] - 10.0) < 0.5
        assert abs(last[1] - 10.0) < 0.5

    def test_plan_move_zero_distance(self):
        waypoints = self.model.plan_move(0.0, 0.0, 0.0, 100.0, 0.001)
        assert len(waypoints) == 0

    def test_speed_limit_respected(self):
        waypoints = self.model.plan_move(100.0, 0.0, 0.0, 1000.0, 0.001)
        for wp in waypoints:
            assert wp[3] <= DEFAULT_CONFIG.max_speed_xy + 1.0  # Small tolerance

    def test_execute_move_updates_position(self):
        self.model.execute_move(50.0, 30.0, 0.0, 100.0, 0.001)
        assert abs(self.model.x - 50.0) < 0.01
        assert abs(self.model.y - 30.0) < 0.01

    def test_execute_move_updates_motors(self):
        self.model.execute_move(10.0, 5.0, 0.0, 100.0, 0.001)
        expected_a, expected_b = MotionModel.inverse_kinematics(10.0, 5.0)
        assert abs(self.model.motor_a - expected_a) < 0.01
        assert abs(self.model.motor_b - expected_b) < 0.01

    def test_input_shaper_preserves_count(self):
        waypoints = self.model.plan_move(20.0, 0.0, 0.0, 100.0, 0.001)
        shaped = self.model.apply_input_shaper(waypoints, 0.001)
        assert len(shaped) == len(waypoints)

    def test_junction_speed(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])  # 90 degree turn
        js = self.model.junction_speed(v1, v2, 0.05)
        assert js > 0
        assert js < DEFAULT_CONFIG.max_speed_xy

    def test_reset(self):
        self.model.execute_move(10.0, 10.0, 5.0, 100.0, 0.001)
        self.model.reset()
        assert self.model.x == 0.0
        assert self.model.y == 0.0
        assert self.model.z == 0.0
