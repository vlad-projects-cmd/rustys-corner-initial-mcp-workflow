# src/render.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.features import load_matches, build_team_match_history, compute_rolling_averages, get_fixture_features
from src.model_poisson import compute_league_goal_rates, predict_match_from_features, PoissonConfig

import json
from datetime import datetime, timezone


@dataclass(frozen=True)
class RenderConfig:
    window: int = 5
    top_scorelines: int = 5
    max_goals_grid: int = 5
    reports_dir: Path = Path("reports")
    predictions_dir: Path = Path("data/predictions")


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _fmt_dt(dt: pd.Timestamp) -> str:
    # dt is UTC-aware
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def render_match_block(match_row: pd.Series, pred: Dict[str, object], feats: Dict[str, float]) -> str:
    kickoff = _fmt_dt(match_row["utc_date"])
    lines = []
    lines.append(f"### {pred['home_team']} vs {pred['away_team']}")
    lines.append(f"- Kickoff: {kickoff}")
    lines.append("")
    lines.append(f"- Rolling (last 5) GF/GA:")
    lines.append(f"  - {pred['home_team']}: {feats['home_gf_avg']:.2f} / {feats['home_ga_avg']:.2f}")
    lines.append(f"  - {pred['away_team']}: {feats['away_gf_avg']:.2f} / {feats['away_ga_avg']:.2f}")
    lines.append("")
    lines.append(f"- Expected goals (λ): {pred['home_team']} {pred['lambda_home']:.2f} | {pred['away_team']} {pred['lambda_away']:.2f}")
    lines.append(f"- Outcome probabilities: Home {_pct(pred['p_home_win'])} | Draw {_pct(pred['p_draw'])} | Away {_pct(pred['p_away_win'])}")
    lines.append("")
    top = pred.get("top_scorelines", [])
    # top is like [("1-0", 0.21), ("2-0", 0.19), ...]
    top3 = top[:3] + [("", 0.0)] * (3 - len(top[:3]))  # pad to 3
    top_scoreline_1, top_scoreline_1_p = top3[0]
    top_scoreline_2, top_scoreline_2_p = top3[1]
    top_scoreline_3, top_scoreline_3_p = top3[2]
    top_str = ", ".join([f"{s} ({_pct(p)})" for s, p in top])
    lines.append(f"- Top scorelines: {top_str}")
    lines.append("")
    return "\n".join(lines)


def render_gameweek_outlook(
    season: int,
    gameweek: int,
    competition_id: int = 2021,
    cfg: RenderConfig = RenderConfig(),
    save_predictions: bool = False,
) -> Path:
    """
    Generate a Markdown report for all fixtures in a Premier League gameweek.
    """
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(f"data/processed/matches_comp_{competition_id}_season_{season}.csv")
    matches = load_matches(csv_path)

    # subset fixtures for this gameweek
    gw_matches = matches[matches["matchday"] == gameweek].copy()
    if gw_matches.empty:
        raise ValueError(f"No matches found for season={season}, gameweek={gameweek}")

    # build rolling team features from the full dataset
    team_history = build_team_match_history(matches)
    team_history = compute_rolling_averages(team_history, window=cfg.window)

    # league fallback rates (for early season NaNs)
    league_rates = compute_league_goal_rates(matches)
    league_avg_team_goals = league_rates["avg_team_goals"]

    poisson_cfg = PoissonConfig(max_goals=cfg.max_goals_grid)

    # render
    gw_matches = gw_matches.sort_values("utc_date").reset_index(drop=True)

    md: List[str] = []
    md.append(f"# Premier League — Gameweek {gameweek} Outlook (Season {season})")
    md.append("")
    md.append(f"Model: rolling GF/GA (N={cfg.window}) + Poisson scoreline grid (0..{cfg.max_goals_grid})")
    md.append(f"League avg goals/team (fallback): {league_avg_team_goals:.2f}")
    md.append("")
    md.append("Disclaimer: Sports analytics for entertainment/education. Not betting advice.")
    md.append("")
    md.append("---")
    md.append("")

    pred_items = []
    rank_rows = []
    for _, m in gw_matches.iterrows():
        match_id = int(m["match_id"])
        feats = get_fixture_features(match_id, team_history)
        pred = predict_match_from_features(feats, league_avg_team_goals, cfg=poisson_cfg, top_n_scorelines=cfg.top_scorelines)
        top = pred.get("top_scorelines", [])
        # Ensure we always have 3 for CSV columns
        top3 = top[:3] + [("", 0.0)] * (3 - len(top[:3]))
        top_scoreline_1, top_scoreline_1_p = top3[0]
        top_scoreline_2, top_scoreline_2_p = top3[1]
        top_scoreline_3, top_scoreline_3_p = top3[2]
        # --- END BLOCK ---

        # ... markdown rendering ...
        md.append(render_match_block(m, pred, feats))
        md.append("---")
        md.append("")
        # Predicted outcome label (H/D/A) based on max probability
        p_home = float(pred["p_home_win"])
        p_draw = float(pred["p_draw"])
        p_away = float(pred["p_away_win"])
        predicted_outcome = max([("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda x: x[1])[0]
        pred_items.append(
            {
                "match_id": int(m["match_id"]),
                "kickoff_utc": m["utc_date"].isoformat(),
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "lambda_home": float(pred["lambda_home"]),
                "lambda_away": float(pred["lambda_away"]),
                "p_home_win": float(pred["p_home_win"]),
                "p_draw": float(pred["p_draw"]),
                "p_away_win": float(pred["p_away_win"]),
                "predicted_outcome": predicted_outcome,
                "top_scorelines": [(s, float(p)) for s, p in top],  # JSON-friendly
                "top_scoreline_1": top_scoreline_1,
                "top_scoreline_1_p": float(top_scoreline_1_p),
                "top_scoreline_2": top_scoreline_2,
                "top_scoreline_2_p": float(top_scoreline_2_p),
                "top_scoreline_3": top_scoreline_3,
                "top_scoreline_3_p": float(top_scoreline_3_p),
            }
        )
        
        conf = max(p_home, p_draw, p_away)
        rank_rows.append({
            "home": pred["home_team"],
            "away": pred["away_team"],
            "predicted_outcome": predicted_outcome,
            "confidence": conf,
            "top_scoreline": top_scoreline_1,
            "top_scoreline_p": float(top_scoreline_1_p),
        })

    rank_rows = sorted(rank_rows, key=lambda r: r["confidence"], reverse=True)
    top3 = rank_rows[:3]

    summary = []
    summary.append("## 🔝 Model picks – Gameweek {}".format(gameweek))
    summary.append("")

    for i, r in enumerate(top3, 1):
        side = {"H": "HOME", "A": "AWAY", "D": "DRAW"}[r["predicted_outcome"]]
        conf_pct = int(round(100 * r["confidence"]))
        score = r["top_scoreline"]
        score_p = int(round(100 * r["top_scoreline_p"])) if r["top_scoreline_p"] else None

        line = f"{i}. {r['home']} vs {r['away']} — {side} ({conf_pct}%)"
        if score:
            line += f"\n   Most likely: {score}"
            if score_p:
                line += f" ({score_p}%)"

        summary.append(line)
        summary.append("")

    md = md[:md.index('---')] + summary + ["---", ""] + md[md.index('---')+2:]
    out_path = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    # Optionally save predictions for evaluation
    if save_predictions:
        season_dir = cfg.predictions_dir / f"season_{season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        preds_path = season_dir / f"gameweek_{gameweek}.json"
        # Also write a CSV for easy inspection
        preds_csv = season_dir / f"gameweek_{gameweek}.csv"
        pd.DataFrame(pred_items).to_csv(preds_csv, index=False)


        payload = {
            "season": season,
            "competition_id": competition_id,
            "gameweek": gameweek,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": {
                "window": cfg.window,
                # keep these as documentation; update values if you changed them
                "alpha": 0.7,
                "min_ga": 0.6,
                "max_goals_grid": cfg.max_goals_grid,
            },
            "predictions": pred_items,
        }

        preds_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    # Example:
    # python -m src.render
    path = render_gameweek_outlook(season=2025, gameweek=3)
    print(f"Wrote report: {path}")
