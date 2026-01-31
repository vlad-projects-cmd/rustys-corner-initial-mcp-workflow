# src/model_poisson.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PoissonConfig:
    max_goals: int = 5
    # Fallback if we cannot compute team rolling stats (e.g. GW1). If None, derive from league avg.
    fallback_team_goals: Optional[float] = None
    # Small epsilon to avoid weird edge cases
    eps: float = 1e-9


def poisson_pmf(k: int, lam: float) -> float:
    # PMF = e^-λ * λ^k / k!
    lam = max(lam, 0.0)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def scoreline_grid(lam_home: float, lam_away: float, max_goals: int) -> np.ndarray:
    """
    Returns a (max_goals+1) x (max_goals+1) matrix P(home_goals=i, away_goals=j)
    """
    home_probs = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)], dtype=float)
    away_probs = np.array([poisson_pmf(j, lam_away) for j in range(max_goals + 1)], dtype=float)
    grid = np.outer(home_probs, away_probs)

    # Normalize to sum to 1 over truncated grid (0..max_goals)
    s = grid.sum()
    if s > 0:
        grid = grid / s
    return grid


def outcome_probs(grid: np.ndarray) -> Dict[str, float]:
    """
    From the scoreline grid compute:
    - P(HomeWin): i > j
    - P(Draw): i == j
    - P(AwayWin): i < j
    """
    n = grid.shape[0]
    p_home = float(np.tril(grid, k=-1).sum())  # i > j → home win
    p_draw = float(np.trace(grid))             # i == j
    p_away = float(np.triu(grid, k=1).sum())   # i < j → away win
    return {"p_home_win": p_home, "p_draw": p_draw, "p_away_win": p_away}


def top_scorelines(grid: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Return top_n scorelines like [("1-0", 0.14), ("2-0", 0.12), ...]
    """
    max_i, max_j = grid.shape
    pairs = []
    for i in range(max_i):
        for j in range(max_j):
            pairs.append((i, j, float(grid[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    out = [(f"{i}-{j}", p) for i, j, p in pairs[:top_n]]
    return out


def compute_league_goal_rates(matches: pd.DataFrame) -> Dict[str, float]:
    """
    Compute league average goals per match and per team (simple baseline),
    using FINISHED matches with known goals.
    """
    finished = matches[(matches["status"] == "FINISHED")].copy()
    finished = finished.dropna(subset=["home_goals_ft", "away_goals_ft"])

    if finished.empty:
        # early season, nothing finished - use conservative default
        return {
            "avg_home_goals": 1.35,
            "avg_away_goals": 1.15,
            "avg_team_goals": 1.25,
            "avg_match_goals": 2.50,
        }

    avg_home = float(finished["home_goals_ft"].mean())
    avg_away = float(finished["away_goals_ft"].mean())
    avg_match = avg_home + avg_away
    avg_team = avg_match / 2.0
    return {
        "avg_home_goals": avg_home,
        "avg_away_goals": avg_away,
        "avg_team_goals": avg_team,
        "avg_match_goals": avg_match,
    }


def expected_goals_proxy(
    home_gf: float,
    home_ga: float,
    away_gf: float,
    away_ga: float,
    league_avg_team_goals: float,
    cfg: PoissonConfig,
) -> Tuple[float, float]:
    """
    Simple proxy:
      λ_home = (home_gf * away_ga) / league_avg_team_goals
      λ_away = (away_gf * home_ga) / league_avg_team_goals

    If any inputs are NaN, fall back to league averages (or cfg fallback).
    """
    base = cfg.fallback_team_goals if cfg.fallback_team_goals is not None else league_avg_team_goals

    def _safe(x: float) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return base
        return float(x)

    MIN_GA = 0.6

    home_gf = _safe(home_gf)
    away_gf = _safe(away_gf)

    home_ga = max(_safe(home_ga), MIN_GA)
    away_ga = max(_safe(away_ga), MIN_GA)
    
    ALPHA = 0.7  # trust recent form 70%, league avg 30%

    home_gf = ALPHA * home_gf + (1 - ALPHA) * league_avg_team_goals
    home_ga = ALPHA * home_ga + (1 - ALPHA) * league_avg_team_goals
    away_gf = ALPHA * away_gf + (1 - ALPHA) * league_avg_team_goals
    away_ga = ALPHA * away_ga + (1 - ALPHA) * league_avg_team_goals

    denom = max(league_avg_team_goals, cfg.eps)
    lam_home = (home_gf * away_ga) / denom
    lam_away = (away_gf * home_ga) / denom

    # clamp to reasonable range so early season doesn't go wild
    lam_home = float(np.clip(lam_home, 0.2, 4.0))
    lam_away = float(np.clip(lam_away, 0.2, 4.0))

    return lam_home, lam_away


def predict_match_from_features(
    features: Dict[str, float],
    league_avg_team_goals: float,
    cfg: PoissonConfig = PoissonConfig(),
    top_n_scorelines: int = 5,
) -> Dict[str, object]:
    lam_home, lam_away = expected_goals_proxy(
        home_gf=features["home_gf_avg"],
        home_ga=features["home_ga_avg"],
        away_gf=features["away_gf_avg"],
        away_ga=features["away_ga_avg"],
        league_avg_team_goals=league_avg_team_goals,
        cfg=cfg,
    )

    grid = scoreline_grid(lam_home, lam_away, cfg.max_goals)
    probs = outcome_probs(grid)
    scorelines = top_scorelines(grid, top_n=top_n_scorelines)

    return {
        "home_team": features["home_team"],
        "away_team": features["away_team"],
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        **probs,
        "top_scorelines": scorelines,
    }

if __name__ == "__main__":
    from pathlib import Path
    from src.features import load_matches, build_team_match_history, compute_rolling_averages, get_fixture_features

    from src.features import resolve_matches_path

    season = 2025
    csv_path = resolve_matches_path(competition_id=2021, season=season)
    matches = load_matches(csv_path, season=season)

    team_history = compute_rolling_averages(build_team_match_history(matches), window=5)

    league_rates = compute_league_goal_rates(matches)
    league_avg_team_goals = league_rates["avg_team_goals"]

    # pick a match from GW3 to see non-NaN rolling features
    gw = 5
    match_id = int(matches[matches["matchday"] == gw].iloc[0]["match_id"])

    feats = get_fixture_features(match_id, team_history)
    pred = predict_match_from_features(feats, league_avg_team_goals)

    print("Features:", feats)
    print("Prediction:", pred)
