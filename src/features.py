# src/features.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from src.data_loader import load_matches_for_season


DEFAULT_WINDOW = 5


def load_matches(csv_path: Path, season: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["utc_date"])
    if season is not None and "season" in df.columns:
        df = df[df["season"] == season].copy()
    df = df.sort_values("utc_date").reset_index(drop=True)
    return df


def build_team_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform match-level data into team-level rows.
    One match -> two rows (home team row + away team row).
    """
    home = df[
        [
            "match_id",
            "season",
            "competition_id",
            "matchday",
            "utc_date",
            "status",
            "home_team_id",
            "home_team_name",
            "home_goals_ft",
            "away_goals_ft",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_name": "team_name",
            "home_goals_ft": "goals_for",
            "away_goals_ft": "goals_against",
        }
    )
    home["is_home"] = True

    away = df[
        [
            "match_id",
            "season",
            "competition_id",
            "matchday",
            "utc_date",
            "status",
            "away_team_id",
            "away_team_name",
            "home_goals_ft",
            "away_goals_ft",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_name": "team_name",
            "away_goals_ft": "goals_for",
            "home_goals_ft": "goals_against",
        }
    )
    away["is_home"] = False

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values(["team_id", "utc_date"])

    return team_matches.reset_index(drop=True)


def compute_rolling_averages(
    team_matches: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling GF/GA per team without leakage.

    Computes:
    - gf_roll / ga_roll: overall rolling average (all venues)
    - gf_home_roll / ga_home_roll: rolling average from HOME matches only
    - gf_away_roll / ga_away_roll: rolling average from AWAY matches only

    Uses transform with shift(1) inside each group to prevent
    rolling windows from bleeding across team boundaries.
    """
    team_matches = team_matches.copy()

    # Overall rolling (all matches)
    team_matches["gf_roll"] = team_matches.groupby("team_id")["goals_for"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    team_matches["ga_roll"] = team_matches.groupby("team_id")["goals_against"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Home/away split rolling averages
    # For home-only: mask away matches as NaN, then forward-fill within rolling window
    home_gf = team_matches["goals_for"].where(team_matches["is_home"])
    away_gf = team_matches["goals_for"].where(~team_matches["is_home"])
    home_ga = team_matches["goals_against"].where(team_matches["is_home"])
    away_ga = team_matches["goals_against"].where(~team_matches["is_home"])

    team_matches["gf_home_roll"] = team_matches.groupby("team_id")[[]].transform(
        lambda x: pd.Series(np.nan, index=x.index)
    ).iloc[:, 0] if False else None  # placeholder

    # Proper venue-split computation using groupby + custom transform
    def _venue_rolling(series: pd.Series, mask: pd.Series, window: int) -> pd.Series:
        """Compute rolling mean on venue-filtered matches within each team group."""
        # We need to do this per-team, so we'll build it outside groupby
        result = pd.Series(np.nan, index=series.index)
        return result

    # More efficient approach: compute within grouped context
    team_matches["gf_home_roll"] = np.nan
    team_matches["ga_home_roll"] = np.nan
    team_matches["gf_away_roll"] = np.nan
    team_matches["ga_away_roll"] = np.nan

    for team_id, group in team_matches.groupby("team_id"):
        idx = group.index

        # Home-only stats (NaN for away matches, then rolling ignores NaN)
        hg = group["goals_for"].where(group["is_home"])
        hga = group["goals_against"].where(group["is_home"])
        ag = group["goals_for"].where(~group["is_home"])
        aga = group["goals_against"].where(~group["is_home"])

        team_matches.loc[idx, "gf_home_roll"] = hg.shift(1).rolling(window, min_periods=1).mean()
        team_matches.loc[idx, "ga_home_roll"] = hga.shift(1).rolling(window, min_periods=1).mean()
        team_matches.loc[idx, "gf_away_roll"] = ag.shift(1).rolling(window, min_periods=1).mean()
        team_matches.loc[idx, "ga_away_roll"] = aga.shift(1).rolling(window, min_periods=1).mean()

    return team_matches


def get_fixture_features(
    match_id: int,
    team_matches: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract features for a single fixture.
    Returns overall and venue-split rolling averages.
    """
    rows = team_matches[team_matches["match_id"] == match_id]

    if len(rows) != 2:
        raise ValueError(f"Expected 2 team rows for match {match_id}, got {len(rows)}")

    home = rows[rows["is_home"]].iloc[0]
    away = rows[~rows["is_home"]].iloc[0]

    return {
        "home_team": home["team_name"],
        "away_team": away["team_name"],
        # Overall rolling
        "home_gf_avg": float(home["gf_roll"]),
        "home_ga_avg": float(home["ga_roll"]),
        "away_gf_avg": float(away["gf_roll"]),
        "away_ga_avg": float(away["ga_roll"]),
        # Venue-split rolling (home team's home record, away team's away record)
        "home_gf_at_home": float(home["gf_home_roll"]),
        "home_ga_at_home": float(home["ga_home_roll"]),
        "away_gf_at_away": float(away["gf_away_roll"]),
        "away_ga_at_away": float(away["ga_away_roll"]),
    }


if __name__ == "__main__":
    season = 2025

    matches = load_matches_for_season(season=season)
    team_history = build_team_match_history(matches)
    team_history = compute_rolling_averages(team_history, window=5)

    sample_match_id = matches.iloc[0]["match_id"]
    print(get_fixture_features(sample_match_id, team_history))
