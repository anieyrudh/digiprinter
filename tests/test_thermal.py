"""Tests for thermal physics model."""
import pytest
import numpy as np
from digiprinter.physics.thermal import ThermalModel
from digiprinter.config import PrinterConfig, DEFAULT_CONFIG


class TestThermalModel:
    def setup_method(self):
        self.model = ThermalModel()

    def test_initial_temperature(self):
        assert self.model.hotend_temp == 22.0
        assert self.model.bed_temp == 22.0
        assert self.model.chamber_temp == 22.0

    def test_hotend_heats_up(self):
        """Hotend should reach ~210°C in roughly 30s with full power."""
        for _ in range(30000):  # 30s at 1ms steps
            self.model.step(0.001, hotend_duty=1.0, bed_duty=0.0,
                          fan_fraction=0.0, mass_flow_rate=0.0,
                          filament_temp_in=22.0, ambient_temp=22.0)
        assert self.model.hotend_temp > 150.0  # Should be well above ambient
        assert self.model.hotend_temp < 400.0  # Should not exceed max

    def test_bed_heats_up(self):
        """Bed should heat up significantly with full power over 60s."""
        for _ in range(60000):  # 60s at 1ms steps
            self.model.step(0.001, hotend_duty=0.0, bed_duty=1.0,
                          fan_fraction=0.0, mass_flow_rate=0.0,
                          filament_temp_in=22.0, ambient_temp=22.0)
        # Bed has high thermal mass (1.5kg), so it heats slowly
        assert self.model.bed_temp > 30.0  # Should be well above ambient
        assert self.model.bed_temp < 200.0

    def test_cooldown(self):
        """Temperatures should decrease when heaters are off."""
        # Heat up first
        self.model.hotend_temp = 200.0
        self.model.bed_temp = 80.0
        initial_hotend = self.model.hotend_temp
        initial_bed = self.model.bed_temp
        for _ in range(10000):
            self.model.step(0.001, hotend_duty=0.0, bed_duty=0.0,
                          fan_fraction=1.0, mass_flow_rate=0.0,
                          filament_temp_in=22.0, ambient_temp=22.0)
        assert self.model.hotend_temp < initial_hotend
        assert self.model.bed_temp < initial_bed

    def test_interface_temperature(self):
        """Interface temp should be between chamber and extrusion temps."""
        t_if = ThermalModel.interface_temperature(
            extrusion_temp=210.0, chamber_temp=40.0,
            layer_time=1.0, thermal_diffusivity=5.8e-8)
        assert 40.0 < t_if < 210.0

    def test_thermistor_offset(self):
        self.model.thermistor_offset = 2.0
        assert self.model.hotend_temp_measured == self.model.hotend_temp + 2.0

    def test_reset(self):
        self.model.hotend_temp = 200.0
        self.model.reset(25.0)
        assert self.model.hotend_temp == 25.0
        assert self.model.bed_temp == 25.0
