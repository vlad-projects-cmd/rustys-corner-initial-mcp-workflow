# src/performance.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.metrics import calibration_table, plot_calibration, plot_rolling


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

    return {
        "matches_scored": float(len(finished)),
        "accuracy_outcome": float(finished["correct_outcome"].mean()),
        "brier_1x2_mean": float(finished["brier_1x2"].mean()),
        "logloss_1x2_mean": float(finished["logloss_1x2"].mean()),
        "total_goals_mae": float(finished["total_goals_abs_err"].mean()),
    }


def rolling_metrics(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    finished = df[df["status"] == "FINISHED"].copy()
    if finished.empty:
        return pd.DataFrame()

    finished = finished.sort_values("kickoff_utc").reset_index(drop=True)

    finished["rolling_brier"] = finished["brier_1x2"].rolling(window, min_periods=10).mean()
    finished["rolling_logloss"] = finished["logloss_1x2"].rolling(window, min_periods=10).mean()
    finished["rolling_accuracy"] = finished["correct_outcome"].rolling(window, min_periods=10).mean()
    finished["rolling_goal_mae"] = finished["total_goals_abs_err"].rolling(window, min_periods=10).mean()

    return finished


def available_models(df: pd.DataFrame) -> list[str]:
    if "model_id" not in df.columns:
        return []
    return sorted(df["model_id"].dropna().astype(str).unique().tolist())


def write_performance_report(df: pd.DataFrame, season: int | None, cfg: PerfConfig) -> Path:
    ensure_dirs(cfg)

    summary = compute_cumulative_summary(df)

    season_label = f"Season {season}" if season is not None else "All seasons"
    out_md = cfg.reports_dir / f"model_performance_{'season_' + str(season) if season else 'all'}.md"

    lines = []
    lines.append(f"# Model Performance -- {season_label}")
    lines.append("")
    lines.append("## Cumulative summary")
    for k, v in summary.items():
        if isinstance(v, float) and v.is_integer():
            lines.append(f"- {k}: {int(v)}")
        else:
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("")

    finished = df[df["status"] == "FINISHED"].copy()
    if not finished.empty:
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
    ensure_dirs(cfg)
    df = load_ledger(cfg)
    if df.empty:
        return {}

    if "season" not in df.columns:
        df["season"] = pd.NA

    if season is not None and "season" in df.columns:
        df = df[df["season"] == season].copy()

    artifacts: Dict[str, Path] = {}

    def run_one(df_one: pd.DataFrame, label: str) -> Dict[str, Path]:
        finished_one = df_one[df_one["status"] == "FINISHED"].copy()
        out: Dict[str, Path] = {}

        out["report"] = write_performance_report(df_one, season=season, cfg=cfg)

        if not finished_one.empty:
            calib = calibration_table(finished_one, n_bins=10)
            calib_csv = cfg.reports_dir / f"reliability_table_{label}.csv"
            calib.to_csv(calib_csv, index=False)
            out["reliability_table"] = calib_csv

            calib_png = cfg.reports_dir / f"calibration_{label}.png"
            plot_calibration(calib, calib_png, title=f"Reliability -- {label}")
            out["calibration_plot"] = calib_png

            df_roll = rolling_metrics(df_one, window=50)
            roll_png = cfg.reports_dir / f"rolling_brier_{label}.png"
            plot_rolling(df_roll, roll_png, title=f"Rolling Brier (50-match) -- {label}")
            out["rolling_brier_plot"] = roll_png

        return out

    if "model_id" not in df.columns:
        label = f"season_{season}" if season else "all"
        artifacts.update(run_one(df, label))
        return artifacts

    for mid in available_models(df):
        dfm = df[df["model_id"] == mid].copy()
        safe_mid = mid.replace("/", "_").replace(" ", "_")
        label = f"{safe_mid}_{'season_' + str(season) if season else 'all'}"

        arts = run_one(dfm, label)
        for k, v in arts.items():
            artifacts[f"{mid}:{k}"] = v

    return artifacts
