from __future__ import annotations
import numpy as np
from digiprinter.config import PrinterConfig


class PIDController:
    """Discrete PID controller with anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float, output_min: float = 0.0, output_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement

        # Proportional
        p = self.kp * error

        # Integral with anti-windup (clamp)
        self._integral += error * dt
        i = self.ki * self._integral

        # Derivative (on error, with initialization guard)
        if not self._initialized:
            d = 0.0
            self._initialized = True
        else:
            d = self.kd * (error - self._prev_error) / max(dt, 1e-10)

        self._prev_error = error
        output = p + i + d

        # Clamp output and apply anti-windup
        clamped = np.clip(output, self.output_min, self.output_max)
        if clamped != output:
            # Anti-windup: undo the integral that caused saturation
            self._integral -= error * dt

        return float(clamped)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False


class ThermistorSensor:
    """Simulated thermistor with noise and offset."""

    def __init__(self, noise_std: float = 0.1, offset: float = 0.0):
        self.noise_std = noise_std
        self.offset = offset
        self._rng = np.random.default_rng()

    def read(self, true_temp: float) -> float:
        noise = self._rng.normal(0.0, self.noise_std)
        return true_temp + self.offset + noise

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
