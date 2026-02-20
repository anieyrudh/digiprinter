"""Tests for quality physics model."""
import pytest
import numpy as np
from digiprinter.physics.quality import QualityModel
from digiprinter.materials import PLA, PETG


class TestQualityModel:
    def setup_method(self):
        self.model = QualityModel()

    def test_adhesion_high_at_optimal_temp(self):
        """Adhesion should be high at optimal temperature."""
        q = self.model.compute_adhesion(200.0, 1.0, PLA)
        assert q > 0.5

    def test_adhesion_low_at_low_temp(self):
        """Adhesion should be low below glass transition."""
        q = self.model.compute_adhesion(30.0, 1.0, PLA)
        assert q < 0.3

    def test_adhesion_increases_with_temp(self):
        q_low = self.model.compute_adhesion(100.0, 1.0, PLA)
        self.model.reset()
        q_high = self.model.compute_adhesion(200.0, 1.0, PLA)
        assert q_high > q_low

    def test_warping_increases_with_delta_t(self):
        w1 = self.model.compute_warping(50.0, 400.0, PLA, 0.8)
        w2 = self.model.compute_warping(150.0, 400.0, PLA, 0.8)
        assert w2 > w1

    def test_warping_reduced_by_adhesion(self):
        w_low_adh = self.model.compute_warping(100.0, 400.0, PLA, 0.2)
        self.model.reset()
        w_high_adh = self.model.compute_warping(100.0, 400.0, PLA, 1.0)
        assert w_high_adh < w_low_adh

    def test_stringing_positive(self):
        s = self.model.compute_stringing(500.0, 10.0, PLA, 0.0)
        assert s > 0

    def test_stringing_reduced_by_retraction(self):
        s_no_ret = self.model.compute_stringing(500.0, 10.0, PLA, 0.0)
        self.model.reset()
        s_ret = self.model.compute_stringing(500.0, 10.0, PLA, 1.0)
        assert s_ret < s_no_ret

    def test_dimensional_accuracy(self):
        err = self.model.compute_dimensional_accuracy(0.4, 0.42)
        assert err == pytest.approx(0.05, abs=0.01)

    def test_quality_scores(self):
        self.model.compute_adhesion(200.0, 1.0, PLA)
        self.model.compute_warping(100.0, 400.0, PLA, 0.8)
        self.model.compute_stringing(500.0, 10.0, PLA, 0.5)
        self.model.compute_dimensional_accuracy(0.4, 0.41)
        scores = self.model.get_quality_scores()
        assert "adhesion" in scores
        assert "warping" in scores
        assert "stringing" in scores
        assert "dimensional_error" in scores

    def test_reset(self):
        self.model.compute_adhesion(200.0, 1.0, PLA)
        self.model.reset()
        assert self.model.adhesion_sum == 0.0
        assert self.model.adhesion_count == 0
