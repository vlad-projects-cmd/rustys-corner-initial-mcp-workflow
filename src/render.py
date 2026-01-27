# src/render.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.features import load_matches, build_team_match_history, compute_rolling_averages, get_fixture_features
from src.model_poisson import compute_league_goal_rates, predict_match_from_features, PoissonConfig


@dataclass(frozen=True)
class RenderConfig:
    window: int = 5
    top_scorelines: int = 5
    max_goals_grid: int = 5
    reports_dir: Path = Path("reports")


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
    top = pred["top_scorelines"]
    top_str = ", ".join([f"{s} ({_pct(p)})" for s, p in top])
    lines.append(f"- Top scorelines: {top_str}")
    lines.append("")
    return "\n".join(lines)


def render_gameweek_outlook(
    season: int,
    gameweek: int,
    competition_id: int = 2021,
    cfg: RenderConfig = RenderConfig(),
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

    for _, m in gw_matches.iterrows():
        match_id = int(m["match_id"])
        feats = get_fixture_features(match_id, team_history)
        pred = predict_match_from_features(feats, league_avg_team_goals, cfg=poisson_cfg, top_n_scorelines=cfg.top_scorelines)
        md.append(render_match_block(m, pred, feats))
        md.append("---")
        md.append("")

    out_path = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    # Example:
    # python -m src.render
    path = render_gameweek_outlook(season=2025, gameweek=3)
    print(f"Wrote report: {path}")
