# src/metrics.py
# Shared evaluation metrics, calibration, and plotting utilities.

from __future__ import annotations

from pathlib import Path
from typing import Dict

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def brier_score_1x2(p_home: float, p_draw: float, p_away: float, actual: str) -> float:
    o_home = 1.0 if actual == "H" else 0.0
    o_draw = 1.0 if actual == "D" else 0.0
    o_away = 1.0 if actual == "A" else 0.0
    return ((p_home - o_home) ** 2 + (p_draw - o_draw) ** 2 + (p_away - o_away) ** 2) / 3.0


def log_loss_1x2(p_home: float, p_draw: float, p_away: float, actual: str, eps: float = 1e-12) -> float:
    p_home = min(max(p_home, eps), 1.0 - eps)
    p_draw = min(max(p_draw, eps), 1.0 - eps)
    p_away = min(max(p_away, eps), 1.0 - eps)

    if actual == "H":
        return -math.log(p_home)
    if actual == "D":
        return -math.log(p_draw)
    return -math.log(p_away)


def calibration_table(df_finished: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability for 'predicted outcome happens' using confidence = max(p_home,p_draw,p_away).
    Each row in df_finished must include:
      - p_home, p_draw, p_away
      - pred_outcome ("H"/"D"/"A")
      - actual_outcome ("H"/"D"/"A")
    """
    df = df_finished.copy()
    df["confidence"] = df[["p_home", "p_draw", "p_away"]].max(axis=1)
    df["hit"] = (df["pred_outcome"] == df["actual_outcome"]).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)

    grouped = df.groupby("bin", dropna=False).agg(
        n=("hit", "count"),
        avg_conf=("confidence", "mean"),
        win_rate=("hit", "mean"),
    ).reset_index()

    grouped["avg_conf"] = grouped["avg_conf"].astype(float)
    grouped["win_rate"] = grouped["win_rate"].astype(float)
    return grouped


def plot_calibration(calib: pd.DataFrame, out_path: Path, title: str) -> None:
    c = calib[calib["n"] > 0].copy()
    if c.empty:
        return

    x = c["avg_conf"].to_numpy()
    y = c["win_rate"].to_numpy()
    sizes = c["n"].to_numpy()

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.scatter(x, y, s=20 + 10 * np.sqrt(sizes))
    plt.xlabel("Predicted confidence (avg in bin)")
    plt.ylabel("Observed accuracy (win rate)")
    plt.title(title)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid(True, linewidth=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_rolling(df_roll: pd.DataFrame, out_path: Path, title: str) -> None:
    if df_roll.empty:
        return

    plt.figure()
    plt.plot(df_roll["kickoff_utc"], df_roll["rolling_brier"])
    plt.xlabel("Match date")
    plt.ylabel("Rolling Brier (lower better)")
    plt.title(title)
    plt.grid(True, linewidth=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
