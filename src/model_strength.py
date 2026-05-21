from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrengthConfig:
    # Exponential time decay: weight = exp(-ln(2) * age_days / half_life_days)
    half_life_days: float = 60.0
    # L2 regularization strength (shrinkage)
    l2: float = 1.0
    # Optimizer settings (simple GD; good enough for this scale)
    max_iter: int = 250
    lr: float = 0.01
    tol: float = 1e-6
    # Gradient clipping max norm
    grad_clip_norm: float = 10.0
    # Parameter clamp ranges
    mu_clamp: tuple[float, float] = (-2.0, 2.0)
    home_adv_clamp: tuple[float, float] = (-1.0, 1.0)
    team_param_clamp: tuple[float, float] = (-3.0, 3.0)


@dataclass(frozen=True)
class StrengthModel:
    mu: float
    home_adv: float
    teams: List[int]
    attack: Dict[int, float]
    defence: Dict[int, float]
    cfg: StrengthConfig

    def expected_goals(self, home_team_id: int, away_team_id: int) -> Tuple[float, float]:
        a_h = self.attack.get(home_team_id, 0.0)
        d_h = self.defence.get(home_team_id, 0.0)
        a_a = self.attack.get(away_team_id, 0.0)
        d_a = self.defence.get(away_team_id, 0.0)

        lam_home = math.exp(self.mu + self.home_adv + a_h - d_a)
        lam_away = math.exp(self.mu + a_a - d_h)
        return lam_home, lam_away


def _decay_weights(dates: pd.Series, cutoff: pd.Timestamp, half_life_days: float) -> np.ndarray:
    # age in days
    age_days = (cutoff - dates).dt.total_seconds() / (3600.0 * 24.0)
    age_days = np.clip(age_days.to_numpy(dtype=float), 0.0, None)
    return np.exp(-math.log(2.0) * age_days / float(half_life_days))


def fit_strength_model(
    matches: pd.DataFrame,
    cutoff_utc: pd.Timestamp,
    cfg: StrengthConfig = StrengthConfig(),
) -> StrengthModel:
    """
    Fit Poisson attack/defence model with time decay + L2.

    Uses:
      - status == FINISHED
      - utc_date < cutoff_utc
      - home_goals_ft/away_goals_ft present
    """
    df = matches.copy()
    df = df[df["status"] == "FINISHED"].copy()
    df = df[df["utc_date"] < cutoff_utc].copy()
    df = df.dropna(subset=["home_goals_ft", "away_goals_ft", "home_team_id", "away_team_id", "utc_date"])

    if df.empty:
        # Safe fallback if no history
        return StrengthModel(
            mu=math.log(1.35),
            home_adv=0.0,
            teams=[],
            attack={},
            defence={},
            cfg=cfg,
        )

    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)

    teams = sorted(set(df["home_team_id"]).union(set(df["away_team_id"])))
    t_index = {tid: i for i, tid in enumerate(teams)}
    n = len(teams)

    # theta = [mu, home_adv, attack[n], defence[n]]
    p = 2 + 2 * n
    theta = np.zeros(p, dtype=float)

    mean_goals = float(pd.concat([df["home_goals_ft"], df["away_goals_ft"]]).mean())
    if (not math.isfinite(mean_goals)) or (mean_goals < 0.2):
        raise ValueError(
            f"Training data looks broken: mean_goals={mean_goals}. "
            "Check goals columns and FINISHED filtering."
        )
    theta[0] = math.log(max(mean_goals, 1e-6))
    theta[1] = 0.1

    w = _decay_weights(df["utc_date"], cutoff_utc, cfg.half_life_days)

    h_ids = df["home_team_id"].map(t_index).to_numpy(dtype=int)
    a_ids = df["away_team_id"].map(t_index).to_numpy(dtype=int)
    y_h = df["home_goals_ft"].to_numpy(dtype=float)
    y_a = df["away_goals_ft"].to_numpy(dtype=float)

    def nll_and_grad(th: np.ndarray) -> Tuple[float, np.ndarray]:
        mu = th[0]
        ha = th[1]
        attack = th[2 : 2 + n]
        defence = th[2 + n : 2 + 2 * n]

        eta_h = mu + ha + attack[h_ids] - defence[a_ids]
        eta_a = mu + attack[a_ids] - defence[h_ids]

        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)

        ll = (w * (y_h * eta_h - lam_h)).sum() + (w * (y_a * eta_a - lam_a)).sum()
        nll = -ll

        l2 = float(cfg.l2)
        nll += 0.5 * l2 * (ha * ha + (attack * attack).sum() + (defence * defence).sum())

        r_h = w * (y_h - lam_h)
        r_a = w * (y_a - lam_a)

        grad = np.zeros_like(th)
        grad[0] = -(r_h.sum() + r_a.sum())
        grad[1] = -(r_h.sum()) + l2 * ha

        g_attack = np.zeros(n, dtype=float)
        np.add.at(g_attack, h_ids, r_h)
        np.add.at(g_attack, a_ids, r_a)
        grad[2 : 2 + n] = -(g_attack) + l2 * attack

        g_def = np.zeros(n, dtype=float)
        np.add.at(g_def, a_ids, -r_h)
        np.add.at(g_def, h_ids, -r_a)
        grad[2 + n : 2 + 2 * n] = -(g_def) + l2 * defence

        return nll, grad

    prev = float("inf")
    for _ in range(int(cfg.max_iter)):
        nll, grad = nll_and_grad(theta)
        if abs(prev - nll) < cfg.tol:
            break
        prev = nll
        
        # Gradient clipping to prevent numerical blow-ups
        grad_norm = float(np.sqrt((grad * grad).sum()))
        if grad_norm > cfg.grad_clip_norm:
            grad = grad * (cfg.grad_clip_norm / (grad_norm + 1e-12))

        theta = theta - float(cfg.lr) * grad

        # identifiability constraints: mean attack/defence = 0
        a = theta[2 : 2 + n]
        d = theta[2 + n : 2 + 2 * n]
        theta[2 : 2 + n] = a - a.mean()
        theta[2 + n : 2 + 2 * n] = d - d.mean()

        # Parameter clipping (keeps exp(.) in a sane range)
        theta[0] = float(np.clip(theta[0], *cfg.mu_clamp))
        theta[1] = float(np.clip(theta[1], *cfg.home_adv_clamp))
        theta[2:] = np.clip(theta[2:], *cfg.team_param_clamp)


    mu = float(theta[0])
    ha = float(theta[1])
    attack = theta[2 : 2 + n]
    defence = theta[2 + n : 2 + 2 * n]

    return StrengthModel(
        mu=mu,
        home_adv=ha,
        teams=teams,
        attack={tid: float(attack[i]) for tid, i in t_index.items()},
        defence={tid: float(defence[i]) for tid, i in t_index.items()},
        cfg=cfg,
    )
