# tests/test_metrics.py

import math
import pandas as pd
import pytest

from src.metrics import brier_score_1x2, log_loss_1x2, calibration_table


class TestBrierScore:
    def test_perfect_prediction(self):
        # Predict 100% home win, actual is home win
        score = brier_score_1x2(1.0, 0.0, 0.0, "H")
        assert math.isclose(score, 0.0, abs_tol=1e-9)

    def test_worst_prediction(self):
        # Predict 0% home win, actual is home win
        score = brier_score_1x2(0.0, 0.5, 0.5, "H")
        # (0-1)^2 + (0.5-0)^2 + (0.5-0)^2 = 1 + 0.25 + 0.25 = 1.5 / 3 = 0.5
        assert math.isclose(score, 0.5, rel_tol=1e-6)

    def test_uniform_prediction(self):
        # Equal probabilities
        score = brier_score_1x2(1/3, 1/3, 1/3, "D")
        # (1/3 - 0)^2 + (1/3 - 1)^2 + (1/3 - 0)^2 = 1/9 + 4/9 + 1/9 = 6/9 / 3 = 2/9
        assert math.isclose(score, 2/9, rel_tol=1e-6)


class TestLogLoss:
    def test_confident_correct(self):
        ll = log_loss_1x2(0.9, 0.05, 0.05, "H")
        assert ll < 0.2

    def test_confident_wrong(self):
        ll = log_loss_1x2(0.01, 0.01, 0.98, "H")
        assert ll > 3.0

    def test_uniform(self):
        ll = log_loss_1x2(1/3, 1/3, 1/3, "A")
        assert math.isclose(ll, -math.log(1/3), rel_tol=1e-6)


class TestCalibrationTable:
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "p_home": [0.6, 0.7, 0.4],
            "p_draw": [0.2, 0.2, 0.3],
            "p_away": [0.2, 0.1, 0.3],
            "pred_outcome": ["H", "H", "H"],
            "actual_outcome": ["H", "A", "H"],
        })
        calib = calibration_table(df, n_bins=5)
        assert "avg_conf" in calib.columns
        assert "win_rate" in calib.columns
        assert "n" in calib.columns
