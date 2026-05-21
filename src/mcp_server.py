from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import re

import pandas as pd

from mcp.server.fastmcp import FastMCP

from src.fetch import FootballDataConfig, fetch_season_matches
from src.render import RenderConfig, render_gameweek_outlook
from src.evaluate import EvalConfig, evaluate_gameweek
from src.performance import PerfConfig, refresh_artifacts
from src.data_loader import curate_seasons
from src.competitions import resolve_competition, list_competitions


mcp = FastMCP("Football Predictor", json_response=True)


def _read_text(path: Path, max_chars: int = 25_000) -> str:
    txt = path.read_text(encoding="utf-8")
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt


@mcp.tool()
def football_list_leagues() -> Dict[str, Any]:
    """List all supported leagues with their codes, IDs, and season patterns."""
    comps = list_competitions()
    return {
        "ok": True,
        "leagues": [
            {
                "code": c.code,
                "id": c.id,
                "name": c.name,
                "country": c.country,
                "season_pattern": c.season_pattern,
            }
            for c in comps
        ],
    }


@mcp.tool()
def football_fetch_season(
    season: int,
    league: Optional[str] = None,
    competition_id: Optional[int] = None,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Fetch season matches for any supported league and write normalized CSV. Use 'league' code (e.g. 'allsvenskan', 'pl') or numeric competition_id."""
    comp_id = resolve_competition(league, competition_id)
    cfg = FootballDataConfig(
        competition_id=comp_id,
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
    )
    out_csv = fetch_season_matches(season=season, cfg=cfg, force_refresh=force_refresh)
    return {"ok": True, "csv_path": str(out_csv), "competition_id": comp_id}


@mcp.tool()
def football_curate(
    seasons: list[int] = [2023, 2024, 2025],
    league: Optional[str] = None,
    competition_id: Optional[int] = None,
    processed_dir: str = "data/processed",
    curated_dir: str = "data/curated",
    out_format: str = "csv",
) -> Dict[str, Any]:
    """Merge processed season CSVs into a single curated dataset + manifest. Supports any league."""
    comp_id = resolve_competition(league, competition_id)
    out_path, manifest_path = curate_seasons(
        seasons=seasons,
        competition_id=comp_id,
        processed_dir=Path(processed_dir),
        curated_dir=Path(curated_dir),
        out_format=out_format,
    )
    return {"ok": True, "curated_path": str(out_path), "manifest_path": str(manifest_path)}


@mcp.tool()
def football_generate_outlook(
    season: int,
    gameweek: int,
    league: Optional[str] = None,
    competition_id: Optional[int] = None,
    window: int = 5,
    top_scorelines: int = 5,
    max_goals_grid: int = 5,
    reports_dir: str = "reports",
    predictions_dir: str = "data/predictions",
    save_predictions: bool = True,
    model: str = "rolling",
    half_life_days: float = 60.0,
    l2: float = 1.0,
    lr: float = 0.05,
    max_iter: int = 250,
    venue_weight: float = 0.5,
    elo_k: float = 30.0,
    elo_home_advantage: float = 65.0,
    elo_season_carryover: float = 0.6,
) -> Dict[str, Any]:
    """Generate the gameweek outlook markdown + predictions. Models: rolling, strength, elo, ensemble."""
    comp_id = resolve_competition(league, competition_id)
    cfg = RenderConfig(
        window=window,
        top_scorelines=top_scorelines,
        max_goals_grid=max_goals_grid,
        reports_dir=Path(reports_dir),
        predictions_dir=Path(predictions_dir),
        model=model,
        half_life_days=half_life_days,
        l2=l2,
        lr=lr,
        max_iter=max_iter,
        venue_weight=venue_weight,
        elo_k=elo_k,
        elo_home_advantage=elo_home_advantage,
        elo_season_carryover=elo_season_carryover,
    )

    md_path = render_gameweek_outlook(
        season=season,
        gameweek=gameweek,
        competition_id=comp_id,
        cfg=cfg,
        save_predictions=save_predictions,
    )

    payload: Dict[str, Any] = {"ok": True, "report_path": str(md_path)}
    if save_predictions:
        season_dir = Path(predictions_dir) / f"season_{season}"
        payload["predictions_json"] = str(season_dir / f"gameweek_{gameweek}.json")
        payload["predictions_csv"] = str(season_dir / f"gameweek_{gameweek}.csv")
    return payload


@mcp.tool()
def football_evaluate_gameweek(
    season: int,
    gameweek: int,
    league: Optional[str] = None,
    competition_id: Optional[int] = None,
    append: bool = True,
    refresh_cumulative: bool = True,
) -> Dict[str, Any]:
    """Evaluate saved predictions vs actual results, write review report. Supports any league."""
    comp_id = resolve_competition(league, competition_id)
    cfg = EvalConfig(competition_id=comp_id)
    df, summary, review_path = evaluate_gameweek(
        season=season,
        gameweek=gameweek,
        cfg=cfg,
        append=append,
        refresh_cumulative=refresh_cumulative,
    )
    return {
        "ok": True,
        "review_path": str(review_path),
        "summary": summary,
        "rows_scored": int(summary.get("matches_scored", 0)),
        "ledger_appended": bool(append),
        "cumulative_refreshed": bool(refresh_cumulative),
    }


@mcp.tool()
def football_get_performance(
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Regenerate and return cumulative performance artifacts (md + plots)."""
    artifacts = refresh_artifacts(season=season, cfg=PerfConfig())
    return {"ok": True, "artifacts": {k: str(v) for k, v in artifacts.items()}}


@mcp.tool()
def football_compose_x_post(
    season: int,
    gameweek: int,
    predictions_csv: str | None = None,
    include_disclaimer: bool = True,
) -> Dict[str, Any]:
    """
    Create an X-ready post from the predictions CSV:
    - 2 most confident outcomes (highest max 1X2 probability)
    - 1 'goals' pick (highest lambda_home+lambda_away)
    Writes the draft to reports/x_posts/.
    """
    if predictions_csv is None or not str(predictions_csv).strip():
        predictions_csv = f"data/predictions/season_{season}/gameweek_{gameweek}.csv"

    df = pd.read_csv(predictions_csv)
    required = {"home_team", "away_team", "lambda_home", "lambda_away", "p_home_win", "p_draw", "p_away_win"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    df["conf"] = df[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)

    def outcome_label(r) -> str:
        if r["p_home_win"] >= r["p_draw"] and r["p_home_win"] >= r["p_away_win"]:
            return "HOME"
        if r["p_away_win"] >= r["p_home_win"] and r["p_away_win"] >= r["p_draw"]:
            return "AWAY"
        return "DRAW"

    def best_scoreline(r) -> tuple[str, int]:
        s = str(r.get("top_scoreline_1", "")).strip()
        p = r.get("top_scoreline_1_p", None)

        if s and p is not None and str(p) != "nan":
            return s, int(round(100 * float(p)))

        h = int(round(float(r["lambda_home"])))
        a = int(round(float(r["lambda_away"])))
        h = max(0, min(5, h))
        a = max(0, min(5, a))
        return f"{h}-{a}", -1

    df["pick"] = df.apply(outcome_label, axis=1)
    df["pick_p"] = df[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)
    df["xg_total"] = df["lambda_home"].astype(float) + df["lambda_away"].astype(float)

    # Top 2 by confidence
    top_conf = df.sort_values(["conf"], ascending=False).head(2)

    # Top 1 by expected total goals
    top_goals = df.sort_values(["xg_total"], ascending=False).head(1)

    lines = []
    lines.append(f"PL GW{gameweek} (Season {season}) -- 3 quick calls:")

    for _, r in top_conf.iterrows():
        match = f"{r['home_team']} vs {r['away_team']}"
        pick = r["pick"]
        p = int(round(100 * float(r["pick_p"])))
        score, score_p = best_scoreline(r)
        score_part = f"{score}" if score_p < 0 else f"{score} ({score_p}%)"
        lines.append(f"- {match}: {score_part} ({pick}, {p}%)")

    r = top_goals.iloc[0]
    match = f"{r['home_team']} vs {r['away_team']}"
    lines.append(f"- Goals watch: {match} (xG~{float(r['xg_total']):.1f})")

    if include_disclaimer:
        lines.append("Not betting advice -- analytics/entertainment only.")

    x_post = "\n".join(lines)

    x_post = re.sub(r"[ \t]+", " ", x_post).strip()
    if len(x_post) > 280:
        x_post = "\n".join(lines[:-1]).strip()
    if len(x_post) > 280:
        x_post = x_post[:279] + "..."

    out_dir = Path("reports/x_posts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"x_outlook_gw{gameweek}_season{season}.txt"
    out_path.write_text(x_post, encoding="utf-8")

    return {"ok": True, "x_post": x_post, "saved_to": str(out_path)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
