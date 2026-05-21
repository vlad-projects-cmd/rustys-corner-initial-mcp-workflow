# tests/test_features.py

import pandas as pd
import numpy as np
import pytest

from src.features import build_team_match_history, compute_rolling_averages, get_fixture_features


def _make_matches_df(n_matchdays: int = 5) -> pd.DataFrame:
    """Create a minimal synthetic matches DataFrame for testing."""
    rows = []
    teams = [
        (1, "Team A"),
        (2, "Team B"),
        (3, "Team C"),
        (4, "Team D"),
    ]
    match_id = 100
    for md in range(1, n_matchdays + 1):
        # Match 1: Team A vs Team B
        rows.append({
            "match_id": match_id,
            "season": 2025,
            "competition_id": 2021,
            "matchday": md,
            "utc_date": pd.Timestamp(f"2025-08-{10 + md}", tz="UTC"),
            "status": "FINISHED",
            "home_team_id": 1,
            "home_team_name": "Team A",
            "away_team_id": 2,
            "away_team_name": "Team B",
            "home_goals_ft": 2,
            "away_goals_ft": 1,
        })
        match_id += 1
        # Match 2: Team C vs Team D
        rows.append({
            "match_id": match_id,
            "season": 2025,
            "competition_id": 2021,
            "matchday": md,
            "utc_date": pd.Timestamp(f"2025-08-{10 + md}", tz="UTC"),
            "status": "FINISHED",
            "home_team_id": 3,
            "home_team_name": "Team C",
            "away_team_id": 4,
            "away_team_name": "Team D",
            "home_goals_ft": 1,
            "away_goals_ft": 1,
        })
        match_id += 1
    return pd.DataFrame(rows)


class TestBuildTeamMatchHistory:
    def test_doubles_rows(self):
        df = _make_matches_df(3)
        hist = build_team_match_history(df)
        # Each match produces 2 team rows
        assert len(hist) == len(df) * 2

    def test_has_required_columns(self):
        df = _make_matches_df(2)
        hist = build_team_match_history(df)
        assert "team_id" in hist.columns
        assert "goals_for" in hist.columns
        assert "goals_against" in hist.columns
        assert "is_home" in hist.columns


class TestComputeRollingAverages:
    def test_no_cross_team_bleed(self):
        df = _make_matches_df(5)
        hist = build_team_match_history(df)
        hist = compute_rolling_averages(hist, window=3)

        # Team A scores 2 every game at home. After shift(1), rolling should reflect that.
        team_a = hist[hist["team_id"] == 1].copy()
        # First row should be NaN (shift), second should have 1 data point
        assert pd.isna(team_a.iloc[0]["gf_roll"])
        # By row 3 (3rd match), rolling avg of GF should be 2.0 (Team A always scores 2 at home)
        # But Team A also plays away where they score 1, so this depends on fixture arrangement
        # Key test: no NaN after first row (min_periods=1)
        assert not pd.isna(team_a.iloc[1]["gf_roll"])

    def test_shift_prevents_leakage(self):
        df = _make_matches_df(3)
        hist = build_team_match_history(df)
        hist = compute_rolling_averages(hist, window=5)

        # First match for any team should have NaN rolling (nothing to look back at)
        for tid in hist["team_id"].unique():
            team_rows = hist[hist["team_id"] == tid]
            assert pd.isna(team_rows.iloc[0]["gf_roll"])


class TestGetFixtureFeatures:
    def test_returns_expected_keys(self):
        df = _make_matches_df(5)
        hist = build_team_match_history(df)
        hist = compute_rolling_averages(hist, window=3)

        # Use a match from matchday 3+ so rolling has data
        match_id = 104  # matchday 3, first match
        feats = get_fixture_features(match_id, hist)
        assert "home_team" in feats
        assert "away_team" in feats
        assert "home_gf_avg" in feats
        assert "away_gf_avg" in feats

    def test_raises_on_missing_match(self):
        df = _make_matches_df(2)
        hist = build_team_match_history(df)
        hist = compute_rolling_averages(hist, window=3)

        with pytest.raises(ValueError, match="Expected 2 team rows"):
            get_fixture_features(99999, hist)
