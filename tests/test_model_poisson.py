# tests/test_model_poisson.py

import math
import numpy as np
import pytest

from src.model_poisson import (
    PoissonConfig,
    poisson_pmf,
    scoreline_grid,
    scoreline_grid_dc,
    outcome_probs,
    top_scorelines,
    expected_goals_proxy,
    dixon_coles_tau,
)


class TestPoissonPMF:
    def test_pmf_zero(self):
        # P(X=0 | lambda=1) = e^-1
        assert math.isclose(poisson_pmf(0, 1.0), math.exp(-1), rel_tol=1e-9)

    def test_pmf_sums_approximately_one(self):
        lam = 2.5
        total = sum(poisson_pmf(k, lam) for k in range(20))
        assert math.isclose(total, 1.0, rel_tol=1e-6)

    def test_pmf_negative_lambda_clamps(self):
        assert poisson_pmf(0, -1.0) == 1.0  # lam clamped to 0 => e^0 * 0^0 / 0! = 1


class TestScorelineGrid:
    def test_grid_sums_to_one(self):
        grid = scoreline_grid(1.5, 1.2, max_goals=5)
        assert math.isclose(grid.sum(), 1.0, rel_tol=1e-6)

    def test_grid_shape(self):
        grid = scoreline_grid(1.0, 1.0, max_goals=7)
        assert grid.shape == (8, 8)

    def test_symmetric_lambdas_give_equal_outcomes(self):
        grid = scoreline_grid(1.5, 1.5, max_goals=5)
        probs = outcome_probs(grid)
        assert math.isclose(probs["p_home_win"], probs["p_away_win"], rel_tol=1e-6)

    def test_high_home_lambda_favors_home(self):
        grid = scoreline_grid(3.0, 0.5, max_goals=5)
        probs = outcome_probs(grid)
        assert probs["p_home_win"] > probs["p_away_win"]
        assert probs["p_home_win"] > probs["p_draw"]


class TestDixonColes:
    def test_tau_values(self):
        rho = -0.1
        assert dixon_coles_tau(0, 0, rho) == 1 - rho  # 1.1
        assert dixon_coles_tau(0, 1, rho) == 1 + rho  # 0.9
        assert dixon_coles_tau(1, 0, rho) == 1 + rho  # 0.9
        assert dixon_coles_tau(1, 1, rho) == 1 - rho  # 1.1
        assert dixon_coles_tau(2, 3, rho) == 1.0

    def test_dc_grid_sums_to_one(self):
        grid = scoreline_grid_dc(1.5, 1.2, max_goals=5, rho=-0.1)
        assert math.isclose(grid.sum(), 1.0, rel_tol=1e-6)


class TestOutcomeProbs:
    def test_sums_to_one(self):
        grid = scoreline_grid(1.3, 1.1, max_goals=5)
        probs = outcome_probs(grid)
        total = probs["p_home_win"] + probs["p_draw"] + probs["p_away_win"]
        assert math.isclose(total, 1.0, rel_tol=1e-6)


class TestTopScorelines:
    def test_returns_correct_count(self):
        grid = scoreline_grid(1.5, 1.0, max_goals=5)
        top = top_scorelines(grid, top_n=3)
        assert len(top) == 3

    def test_sorted_descending(self):
        grid = scoreline_grid(1.5, 1.0, max_goals=5)
        top = top_scorelines(grid, top_n=5)
        probs = [p for _, p in top]
        assert probs == sorted(probs, reverse=True)


class TestExpectedGoalsProxy:
    def test_symmetric_inputs_give_similar_lambdas(self):
        cfg = PoissonConfig()
        lam_h, lam_a = expected_goals_proxy(
            home_gf=1.5, home_ga=1.0,
            away_gf=1.5, away_ga=1.0,
            league_avg_team_goals=1.25,
            cfg=cfg,
        )
        assert math.isclose(lam_h, lam_a, rel_tol=1e-6)

    def test_nan_inputs_fallback_to_league_avg(self):
        cfg = PoissonConfig()
        lam_h, lam_a = expected_goals_proxy(
            home_gf=float("nan"), home_ga=float("nan"),
            away_gf=float("nan"), away_ga=float("nan"),
            league_avg_team_goals=1.25,
            cfg=cfg,
        )
        # With all NaN -> fallback to league avg -> both lambdas should be equal
        assert math.isclose(lam_h, lam_a, rel_tol=1e-6)

    def test_lambda_clamping(self):
        cfg = PoissonConfig(lambda_min=0.2, lambda_max=4.0)
        lam_h, lam_a = expected_goals_proxy(
            home_gf=10.0, home_ga=0.1,
            away_gf=0.01, away_ga=10.0,
            league_avg_team_goals=1.25,
            cfg=cfg,
        )
        assert lam_h <= cfg.lambda_max
        assert lam_a >= cfg.lambda_min
