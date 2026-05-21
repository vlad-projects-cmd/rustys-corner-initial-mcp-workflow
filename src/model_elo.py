# src/model_elo.py
# Elo rating system for football match prediction.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import math
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    # K-factor: how much a single result moves the rating
    k: float = 30.0
    # Home advantage in Elo points (added to home team's rating for prediction)
    home_advantage: float = 65.0
    # Initial rating for new teams
    initial_rating: float = 1500.0
    # Season carry-over: how much to regress toward mean between seasons (0=full reset, 1=no regression)
    season_carryover: float = 0.6
    # Goal difference multiplier: scale K by goal margin
    goal_diff_weight: float = 0.5
    # Base-10 scaling factor (standard Elo uses 400)
    scale: float = 400.0


@dataclass
class EloState:
    """Mutable state tracking team ratings."""
    ratings: Dict[int, float] = field(default_factory=dict)
    cfg: EloConfig = field(default_factory=EloConfig)

    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.cfg.initial_rating)

    def set_rating(self, team_id: int, rating: float) -> None:
        self.ratings[team_id] = rating


def win_probability(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    """Expected score for team A against team B (0 to 1)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def elo_1x2_probs(
    home_rating: float,
    away_rating: float,
    home_advantage: float,
    scale: float = 400.0,
) -> Tuple[float, float, float]:
    """
    Convert Elo ratings to P(Home), P(Draw), P(Away).

    Uses the approach of:
    - Compute expected score for home (with home advantage)
    - Map expected score to 1X2 using empirical draw adjustment
    """
    eff_home = home_rating + home_advantage
    e_home = win_probability(eff_home, away_rating, scale)
    e_away = 1.0 - e_home

    # Draw probability estimation:
    # Draws are more likely when teams are evenly matched.
    # Use a simple model: P(Draw) peaks at ~28% when e_home=0.5, decreases as mismatch grows.
    # Based on empirical football data: P(D) ~ 0.28 * (1 - (2*|e_home - 0.5|)^1.5)
    mismatch = abs(e_home - 0.5) * 2.0  # 0 to 1
    p_draw = 0.26 * (1.0 - mismatch ** 1.3)
    p_draw = max(p_draw, 0.05)  # floor

    # Distribute remaining probability
    remaining = 1.0 - p_draw
    p_home = remaining * e_home
    p_away = remaining * e_away

    # Normalize to sum to exactly 1
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def goal_diff_multiplier(goal_diff: int, weight: float) -> float:
    """Scale K-factor by goal difference (bigger wins move ratings more)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    return 1.0 + weight * math.log(gd)


def actual_score(home_goals: int, away_goals: int) -> Tuple[float, float]:
    """Actual score for Elo update: 1=win, 0.5=draw, 0=loss."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def update_elo(
    state: EloState,
    home_team_id: int,
    away_team_id: int,
    home_goals: int,
    away_goals: int,
) -> Tuple[float, float]:
    """
    Update Elo ratings after a match. Returns (new_home_rating, new_away_rating).
    """
    cfg = state.cfg
    r_home = state.get_rating(home_team_id)
    r_away = state.get_rating(away_team_id)

    # Expected scores (with home advantage for prediction)
    e_home = win_probability(r_home + cfg.home_advantage, r_away, cfg.scale)
    e_away = 1.0 - e_home

    # Actual scores
    s_home, s_away = actual_score(home_goals, away_goals)

    # Goal difference multiplier
    gd_mult = goal_diff_multiplier(home_goals - away_goals, cfg.goal_diff_weight)

    # Update
    k_eff = cfg.k * gd_mult
    new_home = r_home + k_eff * (s_home - e_home)
    new_away = r_away + k_eff * (s_away - e_away)

    state.set_rating(home_team_id, new_home)
    state.set_rating(away_team_id, new_away)

    return new_home, new_away


def regress_ratings(state: EloState) -> None:
    """Regress all ratings toward the mean (call between seasons)."""
    mean = state.cfg.initial_rating
    carry = state.cfg.season_carryover
    for team_id in list(state.ratings.keys()):
        old = state.ratings[team_id]
        state.ratings[team_id] = carry * old + (1.0 - carry) * mean


def build_elo_ratings(
    matches: pd.DataFrame,
    cutoff_utc: pd.Timestamp,
    cfg: EloConfig = EloConfig(),
) -> EloState:
    """
    Process all finished matches before cutoff to build current Elo state.
    Handles season boundaries with regression.
    """
    df = matches.copy()
    df = df[df["status"] == "FINISHED"].copy()
    df = df.dropna(subset=["home_goals_ft", "away_goals_ft", "home_team_id", "away_team_id"])
    df = df[df["utc_date"] < cutoff_utc].copy()
    df = df.sort_values("utc_date").reset_index(drop=True)

    if df.empty:
        return EloState(cfg=cfg)

    state = EloState(cfg=cfg)

    # Track season transitions for regression
    prev_season = None

    for _, row in df.iterrows():
        current_season = row.get("season")
        if prev_season is not None and current_season != prev_season:
            regress_ratings(state)
        prev_season = current_season

        update_elo(
            state,
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
            home_goals=int(row["home_goals_ft"]),
            away_goals=int(row["away_goals_ft"]),
        )

    return state


def predict_match_elo(
    home_team_id: int,
    away_team_id: int,
    state: EloState,
) -> Dict[str, float]:
    """
    Predict 1X2 probabilities from current Elo ratings.
    """
    cfg = state.cfg
    r_home = state.get_rating(home_team_id)
    r_away = state.get_rating(away_team_id)

    p_home, p_draw, p_away = elo_1x2_probs(
        r_home, r_away, cfg.home_advantage, cfg.scale
    )

    return {
        "p_home_win": p_home,
        "p_draw": p_draw,
        "p_away_win": p_away,
        "elo_home": r_home,
        "elo_away": r_away,
        "elo_diff": r_home - r_away,
    }
