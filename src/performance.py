# src/performance.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PerfConfig:
    eval_dir: Path = Path("data/evaluation")
    reports_dir: Path = Path("reports")


def ensure_dirs(cfg: PerfConfig) -> None:
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)


def load_ledger(cfg: PerfConfig) -> pd.DataFrame:
    path = cfg.eval_dir / "all_matches.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["kickoff_utc"], low_memory=False)


def compute_cumulative_summary(df: pd.DataFrame) -> Dict[str, float]:
    finished = df[df["status"] == "FINISHED"].copy()
    if finished.empty:
        return {"matches_scored": 0.0}

    out = {
        "matches_scored": float(len(finished)),
        "accuracy_outcome": float(finished["correct_outcome"].mean()),
        "brier_1x2_mean": float(finished["brier_1x2"].mean()),
        "logloss_1x2_mean": float(finished["logloss_1x2"].mean()),
        "total_goals_mae": float(finished["total_goals_abs_err"].mean()),
    }
    return out


def rolling_metrics(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Rolling averages over finished matches, ordered by kickoff_utc.
    """
    finished = df[df["status"] == "FINISHED"].copy()
    if finished.empty:
        return pd.DataFrame()

    finished = finished.sort_values("kickoff_utc").reset_index(drop=True)

    finished["rolling_brier"] = finished["brier_1x2"].rolling(window, min_periods=10).mean()
    finished["rolling_logloss"] = finished["logloss_1x2"].rolling(window, min_periods=10).mean()
    finished["rolling_accuracy"] = finished["correct_outcome"].rolling(window, min_periods=10).mean()
    finished["rolling_goal_mae"] = finished["total_goals_abs_err"].rolling(window, min_periods=10).mean()

    return finished


def calibration_table(df_finished: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability for 'predicted outcome happens' using confidence = max(p_home,p_draw,p_away).
    """
    df = df_finished.copy()
    df["confidence"] = df[["p_home", "p_draw", "p_away"]].max(axis=1)
    df["hit"] = (df["pred_outcome"] == df["actual_outcome"]).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)

    calib = df.groupby("bin", dropna=False).agg(
        n=("hit", "count"),
        avg_conf=("confidence", "mean"),
        win_rate=("hit", "mean"),
    ).reset_index()

    calib["avg_conf"] = calib["avg_conf"].astype(float)
    calib["win_rate"] = calib["win_rate"].astype(float)
    return calib


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


def write_performance_report(df: pd.DataFrame, season: int | None, cfg: PerfConfig) -> Path:
    ensure_dirs(cfg)

    finished = df[df["status"] == "FINISHED"].copy()
    summary = compute_cumulative_summary(df)

    season_label = f"Season {season}" if season is not None else "All seasons"
    out_md = cfg.reports_dir / f"model_performance_{'season_'+str(season) if season else 'all'}.md"

    lines = []
    lines.append(f"# Model Performance — {season_label}")
    lines.append("")
    lines.append("## Cumulative summary")
    for k, v in summary.items():
        if isinstance(v, float) and v.is_integer():
            lines.append(f"- {k}: {int(v)}")
        else:
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("")

    if not finished.empty:
        # Latest rolling snapshot (50 match window)
        df_roll = rolling_metrics(df, window=50)
        if not df_roll.empty:
            last = df_roll.iloc[-1]
            lines.append("## Rolling (last ~50 matches)")
            lines.append(f"- rolling_brier: {float(last['rolling_brier']):.4f}" if pd.notna(last["rolling_brier"]) else "- rolling_brier: n/a")
            lines.append(f"- rolling_accuracy: {float(last['rolling_accuracy']):.4f}" if pd.notna(last["rolling_accuracy"]) else "- rolling_accuracy: n/a")
            lines.append(f"- rolling_goal_mae: {float(last['rolling_goal_mae']):.4f}" if pd.notna(last["rolling_goal_mae"]) else "- rolling_goal_mae: n/a")
            lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def refresh_artifacts(season: int | None = None, cfg: PerfConfig = PerfConfig()) -> Dict[str, Path]:
    """
    Build cumulative report + charts from the ledger.
    If season is provided, filter to that season.
    """
    ensure_dirs(cfg)
    df = load_ledger(cfg)
    if df.empty:
        return {}

    # Backward-compat: if ledger was created before we added season/gameweek columns
    if "season" not in df.columns:
        # try to recover from kickoff_utc year as a fallback, otherwise keep unfiltered
        df["season"] = pd.NA

    if season is not None and "season" in df.columns:
        df = df[df["season"] == season].copy()


    finished = df[df["status"] == "FINISHED"].copy()
    artifacts: Dict[str, Path] = {}

    # performance report
    artifacts["report"] = write_performance_report(df, season=season, cfg=cfg)

    # calibration
    if not finished.empty:
        calib = calibration_table(finished, n_bins=10)
        calib_csv = cfg.reports_dir / f"reliability_table_{'season_'+str(season) if season else 'all'}.csv"
        calib.to_csv(calib_csv, index=False)
        artifacts["reliability_table"] = calib_csv

        calib_png = cfg.reports_dir / f"calibration_{'season_'+str(season) if season else 'all'}.png"
        plot_calibration(calib, calib_png, title=f"Reliability — {'Season '+str(season) if season else 'All'}")
        artifacts["calibration_plot"] = calib_png

        # rolling chart
        df_roll = rolling_metrics(df, window=50)
        roll_png = cfg.reports_dir / f"rolling_brier_{'season_'+str(season) if season else 'all'}.png"
        plot_rolling(df_roll, roll_png, title=f"Rolling Brier (50-match) — {'Season '+str(season) if season else 'All'}")
        artifacts["rolling_brier_plot"] = roll_png

    return artifacts
