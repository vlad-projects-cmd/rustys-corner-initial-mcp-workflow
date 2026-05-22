# src/evaluate.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import pandas as pd

from src.data_loader import load_matches_for_season
from src.metrics import brier_score_1x2, log_loss_1x2, calibration_table, plot_calibration


@dataclass(frozen=True)
class EvalConfig:
    competition_id: int = 2021
    predictions_dir: Path = Path("data/predictions")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")


LEDGER_COLUMNS = [
    "season", "gameweek", "match_id",
    "home_team_name", "away_team_name", "kickoff_utc",
    "status",
    "p_home_win", "p_draw", "p_away_win",
    "lambda_home", "lambda_away",
    "result_home_goals", "result_away_goals",
    "result_outcome",
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


def _find_prediction_file(cfg: EvalConfig, season: int, gameweek: int, model_id: str | None) -> Path:
    """Locate prediction JSON file: model-specific first, then legacy fallback."""
    season_dir = cfg.predictions_dir / f"season_{season}"

    if model_id:
        path = season_dir / f"gameweek_{gameweek}_{model_id}.json"
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Missing predictions file for model '{model_id}': {path}. "
            f"Generate outlook with --save-predictions first."
        )

    # No model_id specified: look for legacy path first, then list available models
    legacy = season_dir / f"gameweek_{gameweek}.json"
    if legacy.exists():
        return legacy

    # Try to find any model-specific file for this gameweek
    pattern = f"gameweek_{gameweek}_*.json"
    found = sorted(season_dir.glob(pattern)) if season_dir.exists() else []
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        names = [f.stem.replace(f"gameweek_{gameweek}_", "") for f in found]
        raise FileNotFoundError(
            f"Multiple prediction files found for gameweek {gameweek}. "
            f"Specify --model-id to pick one: {names}"
        )

    raise FileNotFoundError(
        f"No predictions file found in {season_dir} for gameweek {gameweek}. "
        f"Generate outlook with --save-predictions first."
    )


def list_saved_models(season: int, gameweek: int, cfg: EvalConfig = EvalConfig()) -> List[str]:
    """Return model_ids that have saved predictions for this gameweek."""
    season_dir = cfg.predictions_dir / f"season_{season}"
    if not season_dir.exists():
        return []
    pattern = f"gameweek_{gameweek}_*.json"
    found = sorted(season_dir.glob(pattern))
    prefix = f"gameweek_{gameweek}_"
    return [f.stem[len(prefix):] for f in found]


def evaluate_gameweek(
    season: int,
    gameweek: int,
    cfg: EvalConfig = EvalConfig(),
    append: bool = False,
    refresh_cumulative: bool = False,
    model_id: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Path]:
    """
    Loads predictions for a given gameweek + actual results, computes per-match metrics and summary.
    If model_id is provided, evaluates that specific model's predictions.
    """
    preds_path = _find_prediction_file(cfg, season, gameweek, model_id)
    if not preds_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {preds_path}. "
            f"Generate outlook with saving enabled first."
        )

    preds = load_predictions_json(preds_path)
    model_id = preds.get("model_id")

    if not model_id:
        m = preds.get("model", {}) or {}
        if m.get("type", "").startswith("strength"):
            model_id = f"strength_hl{int(round(m.get('half_life_days', 60)))}_l2{float(m.get('l2', 1.0)):.2f}_g{int(m.get('max_goals_grid', 5))}"
        else:
            model_id = f"rolling_w{int(m.get('window', 5))}_g{int(m.get('max_goals_grid', 5))}"

    items: List[Dict[str, Any]] = preds.get("predictions", [])
    if not items:
        raise ValueError(f"No predictions found in {preds_path}")

    matches = load_matches_for_season(season, cfg.competition_id)

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

        pred_outcome = max(
            [("H", p_home), ("D", p_draw), ("A", p_away)],
            key=lambda x: x[1],
        )[0]

        if match_id not in actual_by_id:
            missing_actual += 1
            rows.append({
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
                "model_id": model_id,
            })
            continue

        ah, aa = actual_by_id[match_id]
        actual_outcome = _outcome_from_score(ah, aa)

        brier = brier_score_1x2(p_home, p_draw, p_away, actual_outcome)
        ll = log_loss_1x2(p_home, p_draw, p_away, actual_outcome)

        rows.append({
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
            "model_id": model_id,
        })

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

    # Write Markdown report (include model_id in filename to avoid overwrites)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    review_suffix = f"_{model_id}" if model_id else ""
    out_md = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}{review_suffix}_review.md"
    out_md.write_text(_render_markdown_review(season, gameweek, preds_path, summary, df), encoding="utf-8")

    # Calibration outputs
    if not finished_df.empty:
        calib = calibration_table(finished_df, n_bins=10)
        calib_csv = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}_reliability_table.csv"
        calib.to_csv(calib_csv, index=False)

        calib_png = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}_calibration.png"
        plot_calibration(calib, calib_png, title=f"Reliability (Gameweek {gameweek}, Season {season})")

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
    lines.append(f"# Premier League -- Gameweek {gameweek} Review (Season {season})")
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

    table_cols = [
        "home_team", "away_team", "actual_home_goals", "actual_away_goals",
        "p_home", "p_draw", "p_away", "pred_outcome", "actual_outcome",
        "brier_1x2", "total_goals_abs_err", "status",
    ]

    display = df[table_cols].copy()
    for c in ["p_home", "p_draw", "p_away", "brier_1x2", "total_goals_abs_err"]:
        display[c] = display[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

    display["score"] = display.apply(
        lambda r: "" if pd.isna(r["actual_home_goals"]) else f"{int(r['actual_home_goals'])}-{int(r['actual_away_goals'])}",
        axis=1,
    )

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
    df, summary, out_md = evaluate_gameweek(season=2025, gameweek=3)
    print(summary)
    print(f"Wrote: {out_md}")
