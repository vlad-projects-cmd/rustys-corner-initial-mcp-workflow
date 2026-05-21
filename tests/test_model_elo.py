# tests/test_model_elo.py

import math
import pandas as pd
import numpy as np
import pytest

from src.model_elo import (
    EloConfig,
    EloState,
    win_probability,
    elo_1x2_probs,
    update_elo,
    build_elo_ratings,
    predict_match_elo,
    actual_score,
    goal_diff_multiplier,
    regress_ratings,
)


class TestWinProbability:
    def test_equal_ratings_gives_50_50(self):
        p = win_probability(1500, 1500)
        assert math.isclose(p, 0.5, rel_tol=1e-6)

    def test_higher_rating_favored(self):
        p = win_probability(1600, 1400)
        assert p > 0.5

    def test_symmetric(self):
        p1 = win_probability(1600, 1400)
        p2 = win_probability(1400, 1600)
        assert math.isclose(p1 + p2, 1.0, rel_tol=1e-6)

    def test_large_difference(self):
        p = win_probability(1800, 1200)
        assert p > 0.9


class TestElo1x2Probs:
    def test_sums_to_one(self):
        p_h, p_d, p_a = elo_1x2_probs(1500, 1500, 65.0)
        assert math.isclose(p_h + p_d + p_a, 1.0, rel_tol=1e-6)

    def test_home_advantage_favors_home(self):
        p_h, p_d, p_a = elo_1x2_probs(1500, 1500, 65.0)
        assert p_h > p_a

    def test_equal_without_home_advantage(self):
        p_h, p_d, p_a = elo_1x2_probs(1500, 1500, 0.0)
        assert math.isclose(p_h, p_a, rel_tol=1e-2)

    def test_strong_team_dominates(self):
        p_h, p_d, p_a = elo_1x2_probs(1700, 1300, 65.0)
        assert p_h > 0.6
        assert p_a < 0.15

    def test_draw_probability_reasonable(self):
        p_h, p_d, p_a = elo_1x2_probs(1500, 1500, 65.0)
        # Draws typically 20-30% in football
        assert 0.15 < p_d < 0.35


class TestUpdateElo:
    def test_winner_gains_rating(self):
        state = EloState(cfg=EloConfig())
        state.set_rating(1, 1500)
        state.set_rating(2, 1500)

        new_h, new_a = update_elo(state, 1, 2, home_goals=2, away_goals=0)
        assert new_h > 1500
        assert new_a < 1500

    def test_draw_moves_toward_expected(self):
        state = EloState(cfg=EloConfig())
        state.set_rating(1, 1500)
        state.set_rating(2, 1500)

        # With home advantage, home is expected to win.
        # A draw is slightly below expected for home -> home rating drops slightly
        new_h, new_a = update_elo(state, 1, 2, home_goals=1, away_goals=1)
        assert new_h < 1500  # draw underperformed expectation (home was favored)
        assert new_a > 1500

    def test_big_upset_moves_more(self):
        state1 = EloState(cfg=EloConfig())
        state1.set_rating(1, 1600)
        state1.set_rating(2, 1400)

        state2 = EloState(cfg=EloConfig())
        state2.set_rating(1, 1600)
        state2.set_rating(2, 1400)

        # Normal loss
        update_elo(state1, 1, 2, home_goals=0, away_goals=1)
        # Big loss
        update_elo(state2, 1, 2, home_goals=0, away_goals=4)

        # Big loss should drop home rating more
        assert state2.get_rating(1) < state1.get_rating(1)


class TestGoalDiffMultiplier:
    def test_one_goal_no_bonus(self):
        assert goal_diff_multiplier(1, 0.5) == 1.0

    def test_larger_margin_bigger_multiplier(self):
        m2 = goal_diff_multiplier(2, 0.5)
        m4 = goal_diff_multiplier(4, 0.5)
        assert m2 > 1.0
        assert m4 > m2


class TestActualScore:
    def test_home_win(self):
        assert actual_score(3, 1) == (1.0, 0.0)

    def test_away_win(self):
        assert actual_score(0, 2) == (0.0, 1.0)

    def test_draw(self):
        assert actual_score(1, 1) == (0.5, 0.5)


class TestBuildEloRatings:
    def _make_matches(self, n=20):
        np.random.seed(123)
        rows = []
        match_id = 1000
        base = pd.Timestamp("2025-01-01", tz="UTC")
        for i in range(n):
            rows.append({
                "match_id": match_id + i,
                "season": 2025,
                "competition_id": 2021,
                "matchday": i + 1,
                "utc_date": base + pd.Timedelta(days=i * 7),
                "status": "FINISHED",
                "home_team_id": 1 + (i % 4),
                "home_team_name": f"Team {1 + (i % 4)}",
                "away_team_id": 1 + ((i + 1) % 4),
                "away_team_name": f"Team {1 + ((i + 1) % 4)}",
                "home_goals_ft": np.random.poisson(1.5),
                "away_goals_ft": np.random.poisson(1.2),
            })
        return pd.DataFrame(rows)

    def test_builds_state(self):
        df = self._make_matches()
        cutoff = pd.Timestamp("2025-12-01", tz="UTC")
        state = build_elo_ratings(df, cutoff_utc=cutoff)

        assert len(state.ratings) == 4

    def test_ratings_diverge_from_initial(self):
        df = self._make_matches(40)
        cutoff = pd.Timestamp("2025-12-01", tz="UTC")
        state = build_elo_ratings(df, cutoff_utc=cutoff)

        # Not all teams should still be at 1500
        ratings = list(state.ratings.values())
        assert max(ratings) > 1500 or min(ratings) < 1500

    def test_predict_match_returns_probs(self):
        df = self._make_matches()
        cutoff = pd.Timestamp("2025-12-01", tz="UTC")
        state = build_elo_ratings(df, cutoff_utc=cutoff)

        pred = predict_match_elo(1, 2, state)
        assert "p_home_win" in pred
        assert "p_draw" in pred
        assert "p_away_win" in pred
        total = pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"]
        assert math.isclose(total, 1.0, rel_tol=1e-6)


class TestRegressRatings:
    def test_regression_moves_toward_mean(self):
        cfg = EloConfig(season_carryover=0.5)
        state = EloState(cfg=cfg)
        state.set_rating(1, 1600)
        state.set_rating(2, 1400)

        regress_ratings(state)
        assert state.get_rating(1) == 1550  # 0.5*1600 + 0.5*1500
        assert state.get_rating(2) == 1450  # 0.5*1400 + 0.5*1500
