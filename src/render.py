# src/render.py

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.data_loader import load_matches_for_season, load_training_matches
from src.features import build_team_match_history, compute_rolling_averages, get_fixture_features
from src.model_poisson import (
    PoissonConfig,
    compute_league_goal_rates,
    outcome_probs,
    predict_match_from_features,
    scoreline_grid,
    scoreline_grid_dc,
    top_scorelines,
)
from src.model_strength import StrengthConfig, fit_strength_model
from src.model_elo import EloConfig, build_elo_ratings, predict_match_elo
from src.model_ensemble import EnsembleConfig, ensemble_with_lambdas


@dataclass(frozen=True)
class RenderConfig:
    window: int = 5
    top_scorelines: int = 5
    max_goals_grid: int = 5
    reports_dir: Path = Path("reports")
    predictions_dir: Path = Path("data/predictions")
    dc_rho: float | None = None
    include_prev_seasons: int = 0

    # Model selection: "rolling", "strength", "elo", "ensemble"
    model: str = "rolling"

    # Strength model params
    half_life_days: float = 60.0
    l2: float = 1.0
    lr: float = 0.05
    max_iter: int = 250

    # Elo params
    elo_k: float = 30.0
    elo_home_advantage: float = 65.0
    elo_season_carryover: float = 0.6

    # Ensemble weights
    ensemble_weight_rolling: float = 0.35
    ensemble_weight_strength: float = 0.35
    ensemble_weight_elo: float = 0.30

    # Venue split weight for rolling model (0=overall only, 1=venue only)
    venue_weight: float = 0.5


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.0f}%"


def _fmt_dt(dt: pd.Timestamp) -> str:
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    return str(dt)


def make_model_id(cfg: RenderConfig) -> str:
    if cfg.model == "strength":
        return f"strength_hl{int(round(cfg.half_life_days))}_l2{cfg.l2:.2f}_lr{cfg.lr:.3f}_it{int(cfg.max_iter)}_g{cfg.max_goals_grid}"
    if cfg.model == "elo":
        return f"elo_k{int(cfg.elo_k)}_ha{int(cfg.elo_home_advantage)}_co{cfg.elo_season_carryover:.1f}"
    if cfg.model == "ensemble":
        return f"ensemble_r{cfg.ensemble_weight_rolling:.0%}_s{cfg.ensemble_weight_strength:.0%}_e{cfg.ensemble_weight_elo:.0%}_g{cfg.max_goals_grid}"
    return f"rolling_w{int(cfg.window)}_vw{cfg.venue_weight:.1f}_g{cfg.max_goals_grid}"


def predict_from_lambdas(
    home_team: str,
    away_team: str,
    lambda_home: float,
    lambda_away: float,
    max_goals: int,
    top_n: int,
    dc_rho: float | None = None,
) -> Dict[str, Any]:
    if dc_rho is None:
        grid = scoreline_grid(lambda_home, lambda_away, max_goals)
    else:
        grid = scoreline_grid_dc(lambda_home, lambda_away, max_goals, rho=dc_rho)
    probs = outcome_probs(grid)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        **probs,
        "top_scorelines": top_scorelines(grid, top_n=top_n),
    }


# --- Prediction logic (separated from rendering) ---

def predict_gameweek(
    season: int,
    gameweek: int,
    competition_id: int = 2021,
    cfg: RenderConfig = RenderConfig(),
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generate predictions for all fixtures in a gameweek.
    Supports models: rolling, strength, elo, ensemble.
    Returns (gw_matches DataFrame, list of prediction dicts).
    """
    matches = load_matches_for_season(season=season, competition_id=competition_id)
    train_matches = load_training_matches(
        season=season,
        competition_id=competition_id,
        include_prev_seasons=cfg.include_prev_seasons,
    )

    gw_matches = matches[matches["matchday"] == gameweek].copy()
    if gw_matches.empty:
        raise ValueError(f"No matches found for season={season}, gameweek={gameweek}")
    gw_matches = gw_matches.sort_values("utc_date").reset_index(drop=True)

    poisson_cfg = PoissonConfig(
        max_goals=cfg.max_goals_grid,
        dc_rho=cfg.dc_rho,
        venue_weight=cfg.venue_weight,
    )
    model_id = make_model_id(cfg)

    needs_rolling = cfg.model in ("rolling", "ensemble")
    needs_strength = cfg.model in ("strength", "ensemble")
    needs_elo = cfg.model in ("elo", "ensemble")

    # --- Setup: Strength model ---
    strength_by_kickoff: Dict[pd.Timestamp, Any] = {}
    if needs_strength:
        s_cfg = StrengthConfig(
            half_life_days=cfg.half_life_days,
            l2=cfg.l2,
            lr=cfg.lr,
            max_iter=cfg.max_iter,
        )
        kickoffs = sorted(gw_matches["utc_date"].dropna().unique())
        for ko in kickoffs:
            ko_ts = pd.Timestamp(ko)
            if ko_ts.tzinfo is None:
                ko_ts = ko_ts.tz_localize("UTC")
            strength_by_kickoff[ko_ts] = fit_strength_model(train_matches, cutoff_utc=ko_ts, cfg=s_cfg)

    # --- Setup: League averages (always needed for fallbacks) ---
    league_rates = compute_league_goal_rates(train_matches)
    league_avg_team_goals = float(league_rates["avg_team_goals"])

    # --- Setup: Rolling model ---
    team_history = None
    if needs_rolling:
        team_history = build_team_match_history(train_matches)
        team_history = compute_rolling_averages(team_history, window=cfg.window)

    # --- Setup: Elo model ---
    elo_state = None
    if needs_elo:
        elo_cfg = EloConfig(
            k=cfg.elo_k,
            home_advantage=cfg.elo_home_advantage,
            season_carryover=cfg.elo_season_carryover,
        )
        # Use earliest kickoff as cutoff for Elo state
        earliest_ko = pd.Timestamp(gw_matches["utc_date"].min())
        if earliest_ko.tzinfo is None:
            earliest_ko = earliest_ko.tz_localize("UTC")
        elo_state = build_elo_ratings(train_matches, cutoff_utc=earliest_ko, cfg=elo_cfg)

    pred_items: List[Dict[str, Any]] = []

    for _, m in gw_matches.iterrows():
        home_team_id = int(m["home_team_id"])
        away_team_id = int(m["away_team_id"])
        home_team_name = m["home_team_name"]
        away_team_name = m["away_team_name"]

        # --- Rolling prediction ---
        pred_rolling = None
        feats = {
            "home_gf_avg": float("nan"),
            "home_ga_avg": float("nan"),
            "away_gf_avg": float("nan"),
            "away_ga_avg": float("nan"),
        }
        if needs_rolling:
            feats = get_fixture_features(int(m["match_id"]), team_history)
            pred_rolling = predict_match_from_features(
                feats,
                league_avg_team_goals,
                cfg=poisson_cfg,
                top_n_scorelines=cfg.top_scorelines,
            )

        # --- Strength prediction ---
        pred_strength = None
        if needs_strength:
            ko_ts = pd.Timestamp(m["utc_date"])
            if ko_ts.tzinfo is None:
                ko_ts = ko_ts.tz_localize("UTC")
            strength = strength_by_kickoff[ko_ts]
            lam_home, lam_away = strength.expected_goals(home_team_id, away_team_id)
            pred_strength = predict_from_lambdas(
                home_team=home_team_name,
                away_team=away_team_name,
                lambda_home=lam_home,
                lambda_away=lam_away,
                max_goals=poisson_cfg.max_goals,
                top_n=cfg.top_scorelines,
                dc_rho=cfg.dc_rho,
            )

        # --- Elo prediction ---
        pred_elo = None
        if needs_elo:
            elo_probs = predict_match_elo(home_team_id, away_team_id, elo_state)
            pred_elo = {
                "p_home_win": elo_probs["p_home_win"],
                "p_draw": elo_probs["p_draw"],
                "p_away_win": elo_probs["p_away_win"],
                # Elo doesn't produce lambdas natively; skip
            }

        # --- Select or ensemble final prediction ---
        if cfg.model == "ensemble":
            combined = ensemble_with_lambdas(
                predictions=[pred_rolling, pred_strength, pred_elo],
                weights=[cfg.ensemble_weight_rolling, cfg.ensemble_weight_strength, cfg.ensemble_weight_elo],
            )
            # Generate scorelines from ensembled lambdas
            pred = predict_from_lambdas(
                home_team=home_team_name,
                away_team=away_team_name,
                lambda_home=combined["lambda_home"],
                lambda_away=combined["lambda_away"],
                max_goals=poisson_cfg.max_goals,
                top_n=cfg.top_scorelines,
                dc_rho=cfg.dc_rho,
            )
            # Override probs with ensemble-averaged probs (more accurate than re-deriving from averaged lambdas)
            pred["p_home_win"] = combined["p_home_win"]
            pred["p_draw"] = combined["p_draw"]
            pred["p_away_win"] = combined["p_away_win"]
        elif cfg.model == "elo":
            # Elo doesn't produce scorelines; derive from rating-implied lambdas
            # Use a rough lambda estimate based on probabilities
            p_h = pred_elo["p_home_win"]
            p_a = pred_elo["p_away_win"]
            # Approximate: higher win prob -> higher lambda
            avg_goals = league_avg_team_goals if league_avg_team_goals else 1.35
            lam_h = avg_goals * (1.0 + 0.8 * (p_h - 0.33))
            lam_a = avg_goals * (1.0 + 0.8 * (p_a - 0.33))
            lam_h = max(0.3, min(3.5, lam_h))
            lam_a = max(0.3, min(3.5, lam_a))
            pred = predict_from_lambdas(
                home_team=home_team_name,
                away_team=away_team_name,
                lambda_home=lam_h,
                lambda_away=lam_a,
                max_goals=poisson_cfg.max_goals,
                top_n=cfg.top_scorelines,
                dc_rho=cfg.dc_rho,
            )
            # Override probs with Elo's direct predictions (more calibrated)
            pred["p_home_win"] = pred_elo["p_home_win"]
            pred["p_draw"] = pred_elo["p_draw"]
            pred["p_away_win"] = pred_elo["p_away_win"]
        elif cfg.model == "strength":
            pred = pred_strength
        else:
            # rolling
            pred = pred_rolling

        top = pred.get("top_scorelines", [])
        top3 = top[:3] + [("", 0.0)] * (3 - len(top[:3]))

        p_home = float(pred["p_home_win"])
        p_draw = float(pred["p_draw"])
        p_away = float(pred["p_away_win"])
        predicted_outcome = max([("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda x: x[1])[0]

        pred_items.append({
            "match_id": int(m["match_id"]),
            "kickoff_utc": pd.Timestamp(m["utc_date"]).isoformat(),
            "home_team": pred["home_team"],
            "away_team": pred["away_team"],
            "lambda_home": float(pred["lambda_home"]),
            "lambda_away": float(pred["lambda_away"]),
            "p_home_win": p_home,
            "p_draw": p_draw,
            "p_away_win": p_away,
            "predicted_outcome": predicted_outcome,
            "top_scorelines": [(s, float(p)) for s, p in top],
            "top_scoreline_1": top3[0][0],
            "top_scoreline_1_p": float(top3[0][1]),
            "top_scoreline_2": top3[1][0],
            "top_scoreline_2_p": float(top3[1][1]),
            "top_scoreline_3": top3[2][0],
            "top_scoreline_3_p": float(top3[2][1]),
            "model_id": model_id,
            # carry features for rendering
            "_feats": feats,
        })

    return gw_matches, pred_items


# --- Markdown rendering (separated from prediction) ---

def _render_match_block(match_row: pd.Series, pred: Dict[str, Any], window: int) -> str:
    feats = pred.get("_feats", {})
    kickoff = _fmt_dt(match_row["utc_date"])
    lines: List[str] = []
    lines.append(f"### {pred['home_team']} vs {pred['away_team']}")
    lines.append(f"- Kickoff: {kickoff}")
    lines.append("")

    if pd.notna(feats.get("home_gf_avg", float("nan"))):
        lines.append(f"- Rolling (last {window}) GF/GA:")
        lines.append(f"  - {pred['home_team']}: {feats['home_gf_avg']:.2f} / {feats['home_ga_avg']:.2f}")
        lines.append(f"  - {pred['away_team']}: {feats['away_gf_avg']:.2f} / {feats['away_ga_avg']:.2f}")
        lines.append("")

    lines.append(
        f"- Expected goals (lambda): {pred['home_team']} {pred['lambda_home']:.2f} | {pred['away_team']} {pred['lambda_away']:.2f}"
    )
    lines.append(
        f"- Outcome probabilities: Home {_pct(pred['p_home_win'])} | Draw {_pct(pred['p_draw'])} | Away {_pct(pred['p_away_win'])}"
    )
    lines.append("")

    top = pred.get("top_scorelines", [])
    top_str = ", ".join([f"{s} ({_pct(p)})" for s, p in top]) if top else "(n/a)"
    lines.append(f"- Top scorelines: {top_str}")
    lines.append("")
    return "\n".join(lines)


def render_outlook_markdown(
    season: int,
    gameweek: int,
    gw_matches: pd.DataFrame,
    pred_items: List[Dict[str, Any]],
    cfg: RenderConfig,
    league_avg_team_goals: float | None = None,
) -> str:
    """Render predictions into a Markdown string."""
    md: List[str] = []
    md.append(f"# Premier League -- Gameweek {gameweek} Outlook (Season {season})")
    md.append("")
    if cfg.model == "strength":
        md.append(
            f"Model: strength (attack/defence + home adv, half-life={cfg.half_life_days}d, l2={cfg.l2}) + "
            f"Poisson scoreline grid (0..{cfg.max_goals_grid})"
        )
    else:
        md.append(f"Model: rolling GF/GA (N={cfg.window}) + Poisson scoreline grid (0..{cfg.max_goals_grid})")
        if league_avg_team_goals is not None:
            md.append(f"League avg goals/team (fallback): {league_avg_team_goals:.2f}")
    md.append("")
    md.append("Disclaimer: Sports analytics for entertainment/education. Not betting advice.")
    md.append("")
    md.append("---")
    md.append("")

    rank_rows: List[Dict[str, Any]] = []

    for i, pred in enumerate(pred_items):
        m = gw_matches.iloc[i]
        md.append(_render_match_block(m, pred, window=cfg.window))
        md.append("---")
        md.append("")

        conf = max(pred["p_home_win"], pred["p_draw"], pred["p_away_win"])
        rank_rows.append({
            "home": pred["home_team"],
            "away": pred["away_team"],
            "predicted_outcome": pred["predicted_outcome"],
            "confidence": float(conf),
            "top_scoreline": pred["top_scoreline_1"],
            "top_scoreline_p": pred["top_scoreline_1_p"],
        })

    # Top 3 most confident picks
    rank_rows = sorted(rank_rows, key=lambda r: r["confidence"], reverse=True)
    top3_picks = rank_rows[:3]

    summary: List[str] = []
    summary.append(f"## Top Model picks -- Gameweek {gameweek}")
    summary.append("")
    for i, r in enumerate(top3_picks, 1):
        side = {"H": "HOME", "A": "AWAY", "D": "DRAW"}[r["predicted_outcome"]]
        conf_pct = int(round(100 * r["confidence"]))
        score = r["top_scoreline"]
        score_p = int(round(100 * r["top_scoreline_p"])) if r["top_scoreline_p"] else None

        line = f"{i}. {r['home']} vs {r['away']} -- {side} ({conf_pct}%)"
        if score:
            line += f"\n   Most likely: {score}"
            if score_p:
                line += f" ({score_p}%)"
        summary.append(line)
        summary.append("")

    # Insert summary after the first ---
    try:
        sep_idx = md.index("---")
        md = md[:sep_idx] + [""] + summary + ["---", ""] + md[sep_idx + 2:]
    except ValueError:
        md = summary + [""] + md

    return "\n".join(md)


def write_predictions(
    pred_items: List[Dict[str, Any]],
    season: int,
    gameweek: int,
    competition_id: int,
    cfg: RenderConfig,
) -> tuple[Path, Path]:
    """Save predictions as JSON + CSV. Returns (json_path, csv_path).

    Files are stored per-model so different models don't overwrite each other:
        data/predictions/season_{season}/gameweek_{gameweek}_{model_id}.json
        data/predictions/season_{season}/gameweek_{gameweek}_{model_id}.csv
    """
    model_id = make_model_id(cfg)
    season_dir = cfg.predictions_dir / f"season_{season}"
    season_dir.mkdir(parents=True, exist_ok=True)

    preds_json = season_dir / f"gameweek_{gameweek}_{model_id}.json"
    preds_csv = season_dir / f"gameweek_{gameweek}_{model_id}.csv"

    # Strip internal _feats before saving
    clean_items = [{k: v for k, v in it.items() if not k.startswith("_")} for it in pred_items]

    pd.DataFrame(clean_items).to_csv(preds_csv, index=False)

    model_types = {
        "rolling": "rolling_gf_ga_venue_split_poisson",
        "strength": "strength_attack_defence_poisson",
        "elo": "elo_rating_system",
        "ensemble": "ensemble_rolling_strength_elo",
    }
    model_type = model_types.get(cfg.model, cfg.model)
    model_meta: Dict[str, Any] = {
        "type": model_type,
        "max_goals_grid": cfg.max_goals_grid,
        "top_scorelines": cfg.top_scorelines,
    }
    if cfg.model == "strength":
        model_meta.update({
            "half_life_days": cfg.half_life_days,
            "l2": cfg.l2,
            "lr": cfg.lr,
            "max_iter": cfg.max_iter,
        })
    elif cfg.model == "elo":
        model_meta.update({
            "k": cfg.elo_k,
            "home_advantage": cfg.elo_home_advantage,
            "season_carryover": cfg.elo_season_carryover,
        })
    elif cfg.model == "ensemble":
        model_meta.update({
            "weight_rolling": cfg.ensemble_weight_rolling,
            "weight_strength": cfg.ensemble_weight_strength,
            "weight_elo": cfg.ensemble_weight_elo,
        })
    else:
        model_meta["window"] = cfg.window
        model_meta["venue_weight"] = cfg.venue_weight

    model_id = make_model_id(cfg)
    payload = {
        "season": season,
        "competition_id": competition_id,
        "gameweek": gameweek,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model": model_meta,
        "predictions": clean_items,
    }
    preds_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return preds_json, preds_csv


# --- High-level entrypoint (orchestrates predict + render + save) ---

def render_gameweek_outlook(
    season: int,
    gameweek: int,
    competition_id: int = 2021,
    cfg: RenderConfig = RenderConfig(),
    save_predictions: bool = False,
) -> Path:
    """
    Generate a Markdown report for all fixtures in a Premier League gameweek.
    Optionally save predictions JSON + CSV for evaluation.
    """
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    gw_matches, pred_items = predict_gameweek(season, gameweek, competition_id, cfg)

    # Compute league avg for display
    train_matches = load_training_matches(season=season, competition_id=competition_id, include_prev_seasons=cfg.include_prev_seasons)
    league_rates = compute_league_goal_rates(train_matches)
    league_avg = float(league_rates["avg_team_goals"])

    md_text = render_outlook_markdown(season, gameweek, gw_matches, pred_items, cfg, league_avg)

    out_path = cfg.reports_dir / f"gameweek_{gameweek}_season_{season}.md"
    out_path.write_text(md_text, encoding="utf-8")

    if save_predictions:
        write_predictions(pred_items, season, gameweek, competition_id, cfg)

    return out_path


if __name__ == "__main__":
    path = render_gameweek_outlook(season=2025, gameweek=3, save_predictions=True)
    print(f"Wrote report: {path}")
