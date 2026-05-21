# src/features.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

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

    Uses transform with shift(1) inside each group to prevent
    rolling windows from bleeding across team boundaries.
    """
    team_matches = team_matches.copy()

    team_matches["gf_roll"] = team_matches.groupby("team_id")["goals_for"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    team_matches["ga_roll"] = team_matches.groupby("team_id")["goals_against"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    return team_matches


def get_fixture_features(
    match_id: int,
    team_matches: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract features for a single fixture.
    """
    rows = team_matches[team_matches["match_id"] == match_id]

    if len(rows) != 2:
        raise ValueError(f"Expected 2 team rows for match {match_id}, got {len(rows)}")

    home = rows[rows["is_home"]].iloc[0]
    away = rows[~rows["is_home"]].iloc[0]

    return {
        "home_team": home["team_name"],
        "away_team": away["team_name"],
        "home_gf_avg": float(home["gf_roll"]),
        "home_ga_avg": float(home["ga_roll"]),
        "away_gf_avg": float(away["gf_roll"]),
        "away_ga_avg": float(away["ga_roll"]),
    }


if __name__ == "__main__":
    season = 2025

    matches = load_matches_for_season(season=season)
    team_history = build_team_match_history(matches)
    team_history = compute_rolling_averages(team_history, window=5)

    sample_match_id = matches.iloc[0]["match_id"]
    print(get_fixture_features(sample_match_id, team_history))
