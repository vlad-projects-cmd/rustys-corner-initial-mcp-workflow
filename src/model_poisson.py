# src/model_poisson.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PoissonConfig:
    max_goals: int = 5
    dc_rho: Optional[float] = None
    fallback_team_goals: Optional[float] = None
    eps: float = 1e-9
    # Shrinkage toward league average (0 = pure league avg, 1 = pure recent form)
    form_weight: float = 0.7
    # Floor for goals-against rolling average (prevents division blow-ups)
    min_ga: float = 0.6
    # Lambda clamp range
    lambda_min: float = 0.2
    lambda_max: float = 4.0


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(lam, 0.0)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dixon_coles_tau(h: int, a: int, rho: float) -> float:
    if h == 0 and a == 0:
        return 1 - rho
    if h == 0 and a == 1:
        return 1 + rho
    if h == 1 and a == 0:
        return 1 + rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def scoreline_grid_dc(
    lambda_home: float,
    lambda_away: float,
    max_goals: int,
    rho: float,
) -> np.ndarray:
    grid = np.zeros((max_goals + 1, max_goals + 1))

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)
            p *= dixon_coles_tau(h, a, rho)
            grid[h, a] = p

    grid /= grid.sum()
    return grid


def scoreline_grid(lam_home: float, lam_away: float, max_goals: int) -> np.ndarray:
    home_probs = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)], dtype=float)
    away_probs = np.array([poisson_pmf(j, lam_away) for j in range(max_goals + 1)], dtype=float)
    grid = np.outer(home_probs, away_probs)

    s = grid.sum()
    if s > 0:
        grid = grid / s
    return grid


def outcome_probs(grid: np.ndarray) -> Dict[str, float]:
    p_home = float(np.tril(grid, k=-1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, k=1).sum())
    return {"p_home_win": p_home, "p_draw": p_draw, "p_away_win": p_away}


def top_scorelines(grid: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
    max_i, max_j = grid.shape
    pairs = []
    for i in range(max_i):
        for j in range(max_j):
            pairs.append((i, j, float(grid[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(f"{i}-{j}", p) for i, j, p in pairs[:top_n]]


def compute_league_goal_rates(matches: pd.DataFrame) -> Dict[str, float]:
    finished = matches[(matches["status"] == "FINISHED")].copy()
    finished = finished.dropna(subset=["home_goals_ft", "away_goals_ft"])

    if finished.empty:
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
      lambda_home = (home_gf * away_ga) / league_avg_team_goals
      lambda_away = (away_gf * home_ga) / league_avg_team_goals

    Applies shrinkage toward league average and clamps to safe range.
    """
    base = cfg.fallback_team_goals if cfg.fallback_team_goals is not None else league_avg_team_goals

    def _safe(x: float) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return base
        return float(x)

    home_gf = _safe(home_gf)
    away_gf = _safe(away_gf)
    home_ga = max(_safe(home_ga), cfg.min_ga)
    away_ga = max(_safe(away_ga), cfg.min_ga)

    alpha = cfg.form_weight
    home_gf = alpha * home_gf + (1 - alpha) * league_avg_team_goals
    home_ga = alpha * home_ga + (1 - alpha) * league_avg_team_goals
    away_gf = alpha * away_gf + (1 - alpha) * league_avg_team_goals
    away_ga = alpha * away_ga + (1 - alpha) * league_avg_team_goals

    denom = max(league_avg_team_goals, cfg.eps)
    lam_home = (home_gf * away_ga) / denom
    lam_away = (away_gf * home_ga) / denom

    lam_home = float(np.clip(lam_home, cfg.lambda_min, cfg.lambda_max))
    lam_away = float(np.clip(lam_away, cfg.lambda_min, cfg.lambda_max))

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

    if cfg.dc_rho is not None:
        grid = scoreline_grid_dc(lam_home, lam_away, max_goals=cfg.max_goals, rho=cfg.dc_rho)
    else:
        grid = scoreline_grid(lam_home, lam_away, max_goals=cfg.max_goals)

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
    from src.data_loader import load_matches_for_season
    from src.features import build_team_match_history, compute_rolling_averages, get_fixture_features

    season = 2025
    matches = load_matches_for_season(season=season)
    team_history = compute_rolling_averages(build_team_match_history(matches), window=5)

    league_rates = compute_league_goal_rates(matches)
    league_avg_team_goals = league_rates["avg_team_goals"]

    gw = 5
    match_id = int(matches[matches["matchday"] == gw].iloc[0]["match_id"])

    feats = get_fixture_features(match_id, team_history)
    pred = predict_match_from_features(feats, league_avg_team_goals)

    print("Features:", feats)
    print("Prediction:", pred)
