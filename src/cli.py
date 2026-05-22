# src/cli.py

from __future__ import annotations

import argparse
from pathlib import Path

from src.fetch import fetch_season_matches, FootballDataConfig
from src.render import RenderConfig, render_gameweek_outlook
from src.evaluate import evaluate_gameweek, list_saved_models, EvalConfig
from src.performance import PerfConfig, refresh_artifacts
from src.data_loader import curate_seasons
from src.competitions import resolve_competition, list_competitions


def _add_league_args(parser: argparse.ArgumentParser) -> None:
    """Add --league and --competition-id to a subparser (mutually informative, not exclusive)."""
    parser.add_argument("--league", type=str, default=None,
                        help="League short code (e.g. pl, allsvenskan, laliga). Use 'leagues' command to list all.")
    parser.add_argument("--competition-id", type=int, default=None,
                        help="football-data.org numeric competition id (alternative to --league)")


def _get_competition_id(args: argparse.Namespace) -> int:
    """Resolve competition ID from --league or --competition-id flags."""
    league = getattr(args, "league", None)
    comp_id = getattr(args, "competition_id", None)
    return resolve_competition(league, comp_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="football-predictor",
        description="Football match predictions (rolling stats + Poisson). Supports multiple leagues.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # leagues
    sub.add_parser("leagues", help="List all supported leagues and their codes.")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch season matches and write normalized CSV.")
    g = p_fetch.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", type=int, help="Single season year (e.g. 2025)")
    g.add_argument("--seasons", type=int, nargs="+", help="One or more seasons (e.g. --seasons 2021 2022 2023)")
    g.add_argument("--season-range", type=int, nargs=2, metavar=("START", "END"),
                   help="Inclusive season range (e.g. --season-range 2018 2025)")
    _add_league_args(p_fetch)
    p_fetch.add_argument("--force-refresh", action="store_true", help="Re-download even if cached")
    p_fetch.add_argument("--raw-dir", type=str, default="data/raw", help="Raw cache directory")
    p_fetch.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")

    # outlook
    p_out = sub.add_parser("outlook", help="Generate a Markdown outlook report for a gameweek.")
    p_out.add_argument("--season", type=int, required=True, help="Season year (e.g. 2025 or 2026 for calendar leagues)")
    p_out.add_argument("--gameweek", type=int, required=True, help="Matchday (gameweek)")
    _add_league_args(p_out)
    p_out.add_argument("--window", type=int, default=5, help="Rolling window size (default: 5)")
    p_out.add_argument("--top-scorelines", type=int, default=5, help="Top scorelines to show (default: 5)")
    p_out.add_argument("--max-goals-grid", type=int, default=5, help="Poisson grid max goals per team (default: 5)")
    p_out.add_argument("--reports-dir", type=str, default="reports", help="Output reports directory")
    p_out.add_argument("--save-predictions", action="store_true", help="Save predictions JSON for evaluation")
    p_out.add_argument("--predictions-dir", type=str, default="data/predictions", help="Predictions output directory")
    p_out.add_argument("--half-life-days", type=float, default=60.0)
    p_out.add_argument("--l2", type=float, default=1.0)
    p_out.add_argument("--model", choices=["rolling", "strength", "elo", "ensemble"], default="rolling")
    p_out.add_argument("--lr", type=float, default=0.05)
    p_out.add_argument("--max-iter", type=int, default=250)
    p_out.add_argument("--dc-rho", type=float, default=None, help="Dixon-Coles rho (e.g. -0.10).")
    p_out.add_argument("--include-prev-seasons", type=int, default=0,
                       help="How many previous seasons to include in training (0 = current season only).")
    p_out.add_argument("--venue-weight", type=float, default=0.5,
                       help="Home/away venue split weight for rolling model (0=overall only, 1=venue only, default: 0.5)")
    p_out.add_argument("--elo-k", type=float, default=30.0, help="Elo K-factor (default: 30)")
    p_out.add_argument("--elo-home-advantage", type=float, default=65.0, help="Elo home advantage in points (default: 65)")
    p_out.add_argument("--elo-season-carryover", type=float, default=0.6, help="Elo season carryover (default: 0.6)")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate predictions vs results for a gameweek.")
    p_eval.add_argument("--season", type=int, required=True, help="Season year (e.g. 2025)")
    p_eval.add_argument("--gameweek", type=int, required=True, help="Matchday (gameweek)")
    _add_league_args(p_eval)
    p_eval.add_argument("--model-id", type=str, default=None,
                        help="Evaluate a specific model's predictions (e.g. 'rolling_w5_vw0.5_g5'). "
                             "If omitted and multiple exist, lists available models.")
    p_eval.add_argument("--append", action="store_true", help="Append per-match evaluation rows to ledger")
    p_eval.add_argument("--refresh-cumulative", action="store_true", help="Regenerate cumulative performance artifacts")

    # curate
    p_cur = sub.add_parser("curate", help="Merge processed season CSVs into a single curated dataset.")
    p_cur.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
                       help="Seasons to include (default: 2023 2024 2025)")
    _add_league_args(p_cur)
    p_cur.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")
    p_cur.add_argument("--curated-dir", type=str, default="data/curated", help="Curated output directory")
    p_cur.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv", help="Output format (default: csv)")

    # performance
    p_perf = sub.add_parser("performance", help="Generate cumulative performance artifacts from the evaluation ledger.")
    p_perf.add_argument("--season", type=int, default=None, help="Filter to a single season (optional)")

    # models (list saved prediction models for a gameweek)
    p_models = sub.add_parser("models", help="List saved prediction models for a gameweek.")
    p_models.add_argument("--season", type=int, required=True, help="Season year")
    p_models.add_argument("--gameweek", type=int, required=True, help="Matchday (gameweek)")
    _add_league_args(p_models)

    return p


def _resolve_seasons(args: argparse.Namespace) -> list[int]:
    if getattr(args, "season", None) is not None:
        return [args.season]
    if getattr(args, "seasons", None):
        return sorted(set(args.seasons))
    if getattr(args, "season_range", None):
        start, end = args.season_range
        lo, hi = (start, end) if start <= end else (end, start)
        return list(range(lo, hi + 1))
    raise RuntimeError("No season selection provided")


def cmd_leagues(args: argparse.Namespace) -> int:
    comps = list_competitions()
    print(f"{'Code':<15} {'ID':<6} {'League':<25} {'Country':<15} {'Season'}")
    print("-" * 75)
    for c in comps:
        pattern = "calendar (Jan-Dec)" if c.season_pattern == "calendar" else "split (Aug-May)"
        print(f"{c.code:<15} {c.id:<6} {c.name:<25} {c.country:<15} {pattern}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    competition_id = _get_competition_id(args)
    cfg = FootballDataConfig(
        competition_id=competition_id,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
    )

    seasons = _resolve_seasons(args)
    wrote = []
    failed = []

    for season in seasons:
        try:
            out_csv = fetch_season_matches(season=season, cfg=cfg, force_refresh=args.force_refresh)
            print(f"OK: season {season} -> {out_csv}")
            wrote.append((season, out_csv))
        except Exception as e:
            print(f"ERROR: season {season} failed: {e}")
            failed.append(season)

    if failed:
        print(f"Done with failures. Succeeded: {len(wrote)}. Failed seasons: {failed}")
        return 2

    print(f"Done. Fetched {len(wrote)} season(s).")
    return 0


def cmd_outlook(args: argparse.Namespace) -> int:
    competition_id = _get_competition_id(args)
    cfg = RenderConfig(
        window=args.window,
        top_scorelines=args.top_scorelines,
        max_goals_grid=args.max_goals_grid,
        reports_dir=Path(args.reports_dir),
        predictions_dir=Path(args.predictions_dir),
        model=args.model,
        half_life_days=args.half_life_days,
        l2=args.l2,
        lr=args.lr,
        max_iter=args.max_iter,
        dc_rho=args.dc_rho,
        include_prev_seasons=args.include_prev_seasons,
        venue_weight=args.venue_weight,
        elo_k=args.elo_k,
        elo_home_advantage=args.elo_home_advantage,
        elo_season_carryover=args.elo_season_carryover,
    )

    out_path = render_gameweek_outlook(
        season=args.season,
        gameweek=args.gameweek,
        competition_id=competition_id,
        cfg=cfg,
        save_predictions=args.save_predictions,
    )

    print(f"OK: wrote {out_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    competition_id = _get_competition_id(args)
    cfg = EvalConfig(competition_id=competition_id)
    df, summary, out_md = evaluate_gameweek(
        season=args.season,
        gameweek=args.gameweek,
        cfg=cfg,
        append=args.append,
        refresh_cumulative=args.refresh_cumulative,
        model_id=args.model_id,
    )
    print(f"OK: wrote {out_md}")
    if args.append:
        print("OK: appended to data/evaluation/all_matches.csv")
    if args.refresh_cumulative:
        print("OK: refreshed cumulative performance artifacts in reports/")
    return 0


def cmd_curate(args: argparse.Namespace) -> int:
    competition_id = _get_competition_id(args)
    out_path, manifest_path = curate_seasons(
        seasons=sorted(set(args.seasons)),
        competition_id=competition_id,
        processed_dir=Path(args.processed_dir),
        curated_dir=Path(args.curated_dir),
        out_format=args.format,
    )
    print(f"OK: wrote {out_path}")
    print(f"OK: wrote {manifest_path}")
    return 0


def cmd_performance(args: argparse.Namespace) -> int:
    artifacts = refresh_artifacts(season=args.season, cfg=PerfConfig())
    if not artifacts:
        print("No evaluation ledger found (data/evaluation/all_matches.csv). Run evaluate first.")
        return 1

    print("OK: refreshed performance artifacts:")
    for k, v in artifacts.items():
        print(f"- {k}: {v}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    competition_id = _get_competition_id(args)
    cfg = EvalConfig(competition_id=competition_id)
    models = list_saved_models(season=args.season, gameweek=args.gameweek, cfg=cfg)
    if not models:
        print(f"No saved predictions found for season {args.season}, gameweek {args.gameweek}.")
        return 1
    print(f"Saved models for season {args.season}, gameweek {args.gameweek}:")
    for m in models:
        print(f"  - {m}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "leagues":
        return cmd_leagues(args)
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "outlook":
        return cmd_outlook(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "curate":
        return cmd_curate(args)
    if args.command == "performance":
        return cmd_performance(args)
    if args.command == "models":
        return cmd_models(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
