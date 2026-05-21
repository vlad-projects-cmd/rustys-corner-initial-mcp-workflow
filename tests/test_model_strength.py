# tests/test_model_strength.py

import math
import pandas as pd
import numpy as np
import pytest

from src.model_strength import StrengthConfig, StrengthModel, fit_strength_model


def _make_training_data(n_matches: int = 40) -> pd.DataFrame:
    """Synthetic training data: 4 teams playing round-robin style."""
    np.random.seed(42)
    teams = [(1, "Strong FC"), (2, "Mid FC"), (3, "Weak FC"), (4, "Average FC")]
    # Strong FC scores more, Weak FC scores less
    attack_rates = {1: 2.2, 2: 1.3, 3: 0.8, 4: 1.2}

    rows = []
    match_id = 1000
    base_date = pd.Timestamp("2025-01-01", tz="UTC")

    for i in range(n_matches):
        h_idx = i % 4
        a_idx = (i + 1) % 4
        h_id, h_name = teams[h_idx]
        a_id, a_name = teams[a_idx]

        h_goals = int(np.random.poisson(attack_rates[h_id]))
        a_goals = int(np.random.poisson(attack_rates[a_id]))

        rows.append({
            "match_id": match_id,
            "season": 2025,
            "competition_id": 2021,
            "matchday": (i // 2) + 1,
            "utc_date": base_date + pd.Timedelta(days=i * 3),
            "status": "FINISHED",
            "home_team_id": h_id,
            "home_team_name": h_name,
            "away_team_id": a_id,
            "away_team_name": a_name,
            "home_goals_ft": h_goals,
            "away_goals_ft": a_goals,
        })
        match_id += 1

    return pd.DataFrame(rows)


class TestFitStrengthModel:
    def test_returns_valid_model(self):
        df = _make_training_data()
        cutoff = pd.Timestamp("2025-06-01", tz="UTC")
        model = fit_strength_model(df, cutoff_utc=cutoff)

        assert isinstance(model, StrengthModel)
        assert len(model.teams) == 4
        assert len(model.attack) == 4
        assert len(model.defence) == 4

    def test_strong_team_has_higher_attack(self):
        df = _make_training_data(80)
        cutoff = pd.Timestamp("2025-12-01", tz="UTC")
        model = fit_strength_model(df, cutoff_utc=cutoff)

        # Strong FC (id=1) should have higher attack than Weak FC (id=3)
        assert model.attack[1] > model.attack[3]

    def test_expected_goals_positive(self):
        df = _make_training_data()
        cutoff = pd.Timestamp("2025-06-01", tz="UTC")
        model = fit_strength_model(df, cutoff_utc=cutoff)

        lam_h, lam_a = model.expected_goals(1, 3)
        assert lam_h > 0
        assert lam_a > 0

    def test_empty_data_returns_fallback(self):
        df = pd.DataFrame(columns=[
            "match_id", "season", "competition_id", "matchday",
            "utc_date", "status", "home_team_id", "home_team_name",
            "away_team_id", "away_team_name", "home_goals_ft", "away_goals_ft",
        ])
        cutoff = pd.Timestamp("2025-06-01", tz="UTC")
        model = fit_strength_model(df, cutoff_utc=cutoff)

        assert model.mu == math.log(1.35)
        assert model.teams == []

    def test_identifiability_constraint(self):
        df = _make_training_data()
        cutoff = pd.Timestamp("2025-06-01", tz="UTC")
        model = fit_strength_model(df, cutoff_utc=cutoff)

        # Mean attack and defence should be ~0 (identifiability)
        mean_attack = sum(model.attack.values()) / len(model.attack)
        mean_defence = sum(model.defence.values()) / len(model.defence)
        assert abs(mean_attack) < 0.1
        assert abs(mean_defence) < 0.1
