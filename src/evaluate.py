# src/evaluate.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class EvalConfig:
    competition_id: int = 2021
    predictions_dir: Path = Path("data/predictions")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")


def load_matches_csv(
    season: int,
    competition_id: int,
    processed_dir: Path,
    curated_dir: Path = Path("data/curated"),
) -> pd.DataFrame:
    """
    Prefer curated merged file; fallback to processed per-season.
    """
    curated_candidates = sorted(curated_dir.glob(f"matches_comp_{competition_id}_seasons_*.csv"))
    if curated_candidates:
        df_all = pd.read_csv(curated_candidates[-1], parse_dates=["utc_date"])
        df = df_all[df_all["season"] == season].copy()
        if not df.empty:
            return df.sort_values("utc_date").reset_index(drop=True)

    # fallback: processed per-season
    csv_path = processed_dir / f"matches_comp_{competition_id}_season_{season}.csv"
    df = pd.read_csv(csv_path, parse_dates=["utc_date"])
    return df.sort_values("utc_date").reset_index(drop=True)

LEDGER_COLUMNS = [
  "season","gameweek","match_id",
  "home_team_name","away_team_name","kickoff_utc",
  "status",
  "p_home_win","p_draw","p_away_win",
  "lambda_home","lambda_away",
  "result_home_goals","result_away_goals",
  "result_outcome",   # H/D/A (optional but consistent)
  "model_id",
]

def load_predictions_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def brier_score_1x2(p_home: float, p_draw: float, p_away: float, actual: str) -> float:
    """
    Multi-class Brier score for 3 outcomes (lower is better).
    Range is [0, 2] but typically ~0.2-0.8 in practice.
    """
    o_home = 1.0 if actual == "H" else 0.0
    o_draw = 1.0 if actual == "D" else 0.0
    o_away = 1.0 if actual == "A" else 0.0
    return ((p_home - o_home) ** 2 + (p_draw - o_draw) ** 2 + (p_away - o_away) ** 2) / 3.0


def log_loss_1x2(p_home: float, p_draw: float, p_away: float, actual: str, eps: float = 1e-12) -> float:
    """
    Multi-class log loss (cross entropy). Lower is better.
    """
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

    # Bin by confidence
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)

    grouped = df.groupby("bin", dropna=False).agg(
        n=("hit", "count"),
        avg_conf=("confidence", "mean"),
        win_rate=("hit", "mean"),
    ).reset_index()

    # Some bins may be empty; keep them for plotting clarity
    grouped["avg_conf"] = grouped["avg_conf"].astype(float)
    grouped["win_rate"] = grouped["win_rate"].astype(float)

    return grouped


def plot_calibration(calib: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    Saves a reliability diagram:
      x = avg_confidence, y = observed win_rate
      plus y=x reference.
    """
    # Filter empty bins (n==0) to avoid NaNs on plot
    c = calib[calib["n"] > 0].copy()
    if c.empty:
        return

    x = c["avg_conf"].to_numpy()
    y = c["win_rate"].to_numpy()
    sizes = c["n"].to_numpy()

    plt.figure()
    plt.plot([0, 1], [0, 1])  # reference line
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


def evaluate_gameweek(
    season: int,
    gameweek: int,
    cfg: EvalConfig = EvalConfig(),
    append: bool = False,
    refresh_cumulative: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float], Path]:
    """
    Loads predictions for a given gameweek + actual results, computes per-match metrics and summary.

    Returns:
      - per_match_df
      - summary dict
      - markdown report path
    """
    preds_path = cfg.predictions_dir / f"season_{season}" / f"gameweek_{gameweek}.json"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {preds_path}. "
            f"Generate outlook with saving enabled first."
        )

    preds = load_predictions_json(preds_path)
    model_id = preds.get("model_id")

    # Backward compat: derive from preds["model"] if needed
    if not model_id:
        m = preds.get("model", {}) or {}
        if m.get("type", "").startswith("strength"):
            model_id = f"strength_hl{int(round(m.get('half_life_days', 60)))}_l2{float(m.get('l2', 1.0)):.2f}_g{int(m.get('max_goals_grid', 5))}"
        else:
            model_id = f"rolling_w{int(m.get('window', 5))}_g{int(m.get('max_goals_grid', 5))}"
    
    items: List[Dict[str, Any]] = preds.get("predictions", [])
    if not items:
        raise ValueError(f"No predictions found in {preds_path}")

    matches = load_matches_csv(season, cfg.competition_id, cfg.processed_dir)

    # Build lookup for actual results by match_id
    finished = matches[(matches["matchday"] == gameweek) & (matches["status"] == "FINISHED")].copy()
    finished = finished.dropna(subset=["home_goals_ft", "away_goals_ft"])

    actual_by_id = {
        int(r["match_id"]): (int(r["home_goals_ft"]), int(r["away_goals_ft"]))
        for _, r in finished.iterrows()
    }

    rows: List[Dict[str, Any]] = []
    missing_actual = 0

    for it in items:
        match_id = int(it["match_id"])
        home = it["home_team"]
        away = it["away_team"]

        p_home = float(it["p_home_win"])
        p_draw = float(it["p_draw"])
        p_away = float(it["p_away_win"])

        lam_home = float(it["lambda_home"])
        lam_away = float(it["lambda_away"])

        # predicted "winner" = argmax
        pred_outcome = max(
            [("H", p_home), ("D", p_draw), ("A", p_away)],
            key=lambda x: x[1],
        )[0]

        if match_id not in actual_by_id:
            missing_actual += 1
            rows.append(
                {
                    "season": season,
                    "gameweek": gameweek,
                    "match_id": match_id,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_utc": it.get("kickoff_utc"),
                    "status": "NO_RESULT",
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                    "pred_outcome": pred_outcome,
                    "lambda_home": lam_home,
                    "lambda_away": lam_away,
                    "actual_home_goals": None,
                    "actual_away_goals": None,
                    "actual_outcome": None,
                    "correct_outcome": None,
                    "brier_1x2": None,
                    "logloss_1x2": None,
                    "home_goals_abs_err": None,
                    "away_goals_abs_err": None,
                    "total_goals_abs_err": None,
                    "model_id": model_id
                }
            )
            continue

        ah, aa = actual_by_id[match_id]
        actual_outcome = _outcome_from_score(ah, aa)

        brier = brier_score_1x2(p_home, p_draw, p_away, actual_outcome)
        ll = log_loss_1x2(p_home, p_draw, p_away, actual_outcome)

        rows.append(
            {
                "season": season,
                "gameweek": gameweek,
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "kickoff_utc": it.get("kickoff_utc"),
                "status": "FINISHED",
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "pred_outcome": pred_outcome,
                "lambda_home": lam_home,
                "lambda_away": lam_away,
                "actual_home_goals": ah,
                "actual_away_goals": aa,
                "actual_outcome": actual_outcome,
                "correct_outcome": 1 if pred_outcome == actual_outcome else 0,
                "brier_1x2": brier,
                "logloss_1x2": ll,
                "home_goals_abs_err": abs(lam_home - ah),
                "away_goals_abs_err": abs(lam_away - aa),
                "total_goals_abs_err": abs((lam_home + lam_away) - (ah + aa)),
                "model_id": model_id
            }
        )

    df = pd.DataFrame(rows).sort_values(["status", "kickoff_utc", "match_id"], ascending=[True, True, True])

    finished_df = df[df["status"] == "FINISHED"].copy()

    summary: Dict[str, float] = {}
    if not finished_df.empty:
        summary = {
            "matches_predicted": float(len(df)),
            "matches_scored": float(len(finished_df)),
            "missing_results": float(missing_actual),
            "accuracy_outcome": float(finished_df["correct_outcome"].mean()),
            "brier_1x2_mean": float(finished_df["brier_1x2"].mean()),
            "logloss_1x2_mean": float(finished_df["logloss_1x2"].mean()),
            "home_goals_mae": float(finished_df["home_goals_abs_err"].mean()),
            "away_goals_mae": float(finished_df["away_goals_abs_err"].mean()),
            "total_goals_mae": float(finished_df["total_goals_abs_err"].mean()),
        }
    else:
        summary = {
            "matches_predicted": float(len(df)),
            "matches_scored": 0.0,
            "missing_results": float(missing_actual),
        }

    # Write a simple Markdown report
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out_md = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}_review.md"
    out_md.write_text(_render_markdown_review(season, gameweek, preds_path, summary, df), encoding="utf-8")
    
        # Calibration outputs (only if we have finished matches)
    if not finished_df.empty:
        calib = calibration_table(finished_df, n_bins=10)
        calib_csv = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}_reliability_table.csv"
        calib.to_csv(calib_csv, index=False)

        calib_png = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}_calibration.png"
        plot_calibration(
            calib,
            calib_png,
            title=f"Reliability (Gameweek {gameweek}, Season {season})",
        )

    if append:
        ledger = Path("data/evaluation/all_matches.csv")
        append_to_ledger(df, ledger)

    if refresh_cumulative:
        from src.performance import refresh_artifacts, PerfConfig
        refresh_artifacts(season=season, cfg=PerfConfig())

    
    return df, summary, out_md


def _render_markdown_review(
    season: int,
    gameweek: int,
    preds_path: Path,
    summary: Dict[str, float],
    df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append(f"# Premier League — Gameweek {gameweek} Review (Season {season})")
    lines.append("")
    lines.append(f"Predictions source: `{preds_path}`")
    lines.append("")

    lines.append("## Summary")
    for k, v in summary.items():
        if isinstance(v, float) and v.is_integer():
            lines.append(f"- {k}: {int(v)}")
        else:
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("")

    lines.append("## Match-by-match")
    lines.append("")
    # compact table
    table_cols = [
        "home_team",
        "away_team",
        "actual_home_goals",
        "actual_away_goals",
        "p_home",
        "p_draw",
        "p_away",
        "pred_outcome",
        "actual_outcome",
        "brier_1x2",
        "total_goals_abs_err",
        "status",
    ]

    display = df[table_cols].copy()
    # Round numeric fields for readability
    for c in ["p_home", "p_draw", "p_away", "brier_1x2", "total_goals_abs_err"]:
        display[c] = display[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

    display["score"] = display.apply(
        lambda r: "" if pd.isna(r["actual_home_goals"]) else f"{int(r['actual_home_goals'])}-{int(r['actual_away_goals'])}",
        axis=1,
    )

    # Build markdown table manually
    md_cols = ["home_team", "away_team", "score", "p_home", "p_draw", "p_away", "pred_outcome", "actual_outcome", "brier_1x2", "total_goals_abs_err", "status"]
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(md_cols)) + "|")
    for _, r in display.iterrows():
        row = [
            str(r["home_team"]),
            str(r["away_team"]),
            str(r["score"]),
            str(r["p_home"]),
            str(r["p_draw"]),
            str(r["p_away"]),
            str(r["pred_outcome"]) if r["pred_outcome"] else "",
            str(r["actual_outcome"]) if r["actual_outcome"] else "",
            str(r["brier_1x2"]),
            str(r["total_goals_abs_err"]),
            str(r["status"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Disclaimer: Analytics for entertainment/education. Not betting advice.")
    lines.append("")
    return "\n".join(lines)

def append_to_ledger(per_match_df: pd.DataFrame, ledger_path: Path) -> Path:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure deterministic column order: existing ledger columns first, then new ones
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path, nrows=0)
        cols_existing = list(existing.columns)
        cols_new = [c for c in per_match_df.columns if c not in cols_existing]
        ordered = cols_existing + cols_new
        out_df = per_match_df.reindex(columns=ordered)
        out_df.to_csv(ledger_path, mode="a", index=False, header=False)
    else:
        per_match_df.to_csv(ledger_path, index=False)

    return ledger_path



if __name__ == "__main__":
    # Example run (requires that predictions JSON exists)
    df, summary, out_md = evaluate_gameweek(season=2025, gameweek=3)
    print(summary)
    print(f"Wrote: {out_md}")
