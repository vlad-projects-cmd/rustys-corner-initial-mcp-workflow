# src/cli.py

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.fetch import fetch_season_matches, FootballDataConfig
from src.render import render_gameweek_outlook, RenderConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pl-predictor",
        description="Premier League baseline predictions (rolling stats + Poisson).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch season matches and write normalized CSV.")
    p_fetch.add_argument("--season", type=int, required=True, help="Season year (e.g. 2025)")
    p_fetch.add_argument("--competition-id", type=int, default=2021, help="football-data.org competition id (default: 2021 PL)")
    p_fetch.add_argument("--force-refresh", action="store_true", help="Re-download even if cached")
    p_fetch.add_argument("--raw-dir", type=str, default="data/raw", help="Raw cache directory")
    p_fetch.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")

    # outlook
    p_out = sub.add_parser("outlook", help="Generate a Markdown outlook report for a gameweek.")
    p_out.add_argument("--season", type=int, required=True, help="Season year (e.g. 2025)")
    p_out.add_argument("--gameweek", type=int, required=True, help="Premier League matchday (gameweek)")
    p_out.add_argument("--competition-id", type=int, default=2021, help="competition id (default: 2021 PL)")
    p_out.add_argument("--window", type=int, default=5, help="Rolling window size (default: 5)")
    p_out.add_argument("--top-scorelines", type=int, default=5, help="Top scorelines to show (default: 5)")
    p_out.add_argument("--max-goals-grid", type=int, default=5, help="Poisson grid max goals per team (default: 5)")
    p_out.add_argument("--reports-dir", type=str, default="reports", help="Output reports directory")

    return p


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = FootballDataConfig(
        competition_id=args.competition_id,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
    )
    out_csv = fetch_season_matches(
        season=args.season,
        cfg=cfg,
        force_refresh=args.force_refresh,
    )
    print(f"OK: wrote {out_csv}")
    return 0


def cmd_outlook(args: argparse.Namespace) -> int:
    cfg = RenderConfig(
        window=args.window,
        top_scorelines=args.top_scorelines,
        max_goals_grid=args.max_goals_grid,
        reports_dir=Path(args.reports_dir),
    )
    out_path = render_gameweek_outlook(
        season=args.season,
        gameweek=args.gameweek,
        competition_id=args.competition_id,
        cfg=cfg,
    )
    print(f"OK: wrote {out_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "outlook":
        return cmd_outlook(args)

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
