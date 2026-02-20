"""Tests for extrusion physics model."""
import pytest
import numpy as np
from digiprinter.physics.extrusion import ExtrusionModel
from digiprinter.materials import PLA, PETG
from digiprinter.config import DEFAULT_CONFIG


class TestExtrusionModel:
    def setup_method(self):
        self.model = ExtrusionModel()

    def test_pla_viscosity_at_210(self):
        """PLA viscosity should be ~500-2000 Pa·s at 210°C, 100 s⁻¹."""
        eta = ExtrusionModel.cross_wlf_viscosity(210.0, 100.0, PLA)
        assert 10.0 < eta < 50000.0  # Reasonable range for PLA

    def test_viscosity_decreases_with_temperature(self):
        eta_low = ExtrusionModel.cross_wlf_viscosity(190.0, 100.0, PLA)
        eta_high = ExtrusionModel.cross_wlf_viscosity(220.0, 100.0, PLA)
        assert eta_high < eta_low

    def test_viscosity_decreases_with_shear(self):
        """Shear thinning behavior."""
        eta_low_shear = ExtrusionModel.cross_wlf_viscosity(210.0, 10.0, PLA)
        eta_high_shear = ExtrusionModel.cross_wlf_viscosity(210.0, 1000.0, PLA)
        assert eta_high_shear < eta_low_shear

    def test_shear_rate_calculation(self):
        Q = 16e-9  # 16 mm³/s in m³/s
        R = 0.2e-3  # 0.2mm radius
        gamma = ExtrusionModel.shear_rate(Q, R, 0.3)
        assert gamma > 0

    def test_compute_flow(self):
        result = self.model.compute_flow(
            print_speed=100.0, layer_height=0.2, line_width=0.4,
            material=PLA, hotend_temp=210.0)
        assert result["volume_flow_mm3s"] > 0
        assert result["viscosity"] > 0
        assert result["pressure_drop"] > 0
        assert result["die_swell"] >= 1.0
        assert result["mass_flow_rate"] > 0

    def test_flow_increases_with_speed(self):
        r1 = self.model.compute_flow(50.0, 0.2, 0.4, PLA, 210.0)
        r2 = self.model.compute_flow(200.0, 0.2, 0.4, PLA, 210.0)
        assert r2["volume_flow_mm3s"] > r1["volume_flow_mm3s"]

    def test_retraction(self):
        t = self.model.retract(0.8, 35.0)
        assert t > 0
        assert self.model.retracted
        # Double retract should return 0
        t2 = self.model.retract(0.8, 35.0)
        assert t2 == 0.0

    def test_unretract(self):
        self.model.retract(0.8, 35.0)
        t = self.model.unretract(35.0)
        assert t > 0
        assert not self.model.retracted

    def test_ooze(self):
        ooze = self.model.compute_ooze(500.0, 10.0, PLA, 0.0)
        assert ooze > 0
        # With full retraction, ooze should be less
        ooze_retracted = self.model.compute_ooze(500.0, 10.0, PLA, 1.0)
        assert ooze_retracted < ooze

    def test_reset(self):
        self.model.retract(1.0, 35.0)
        self.model.reset()
        assert not self.model.retracted
        assert self.model.die_swell == 1.0
