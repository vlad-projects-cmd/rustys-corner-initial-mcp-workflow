# src/cli.py

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd


from src.fetch import fetch_season_matches, FootballDataConfig
from src.render import render_gameweek_outlook, RenderConfig
from src.evaluate import evaluate_gameweek
from src.performance import PerfConfig, refresh_artifacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pl-predictor",
        description="Premier League baseline predictions (rolling stats + Poisson).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch season matches and write normalized CSV.")
    # exactly one of: --season, --seasons, --season-range
    g = p_fetch.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", type=int, help="Single season year (e.g. 2025)")
    g.add_argument("--seasons", type=int, nargs="+", help="One or more seasons (e.g. --seasons 2021 2022 2023)")
    g.add_argument("--season-range", type=int, nargs=2, metavar=("START", "END"),
               help="Inclusive season range (e.g. --season-range 2018 2025)")
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
    p_out.add_argument("--save-predictions", action="store_true", help="Save predictions JSON for evaluation")
    p_out.add_argument("--predictions-dir", type=str, default="data/predictions", help="Predictions output directory")
    p_out.add_argument("--half-life-days", type=float, default=60.0)
    p_out.add_argument("--l2", type=float, default=1.0)
    p_out.add_argument("--model", choices=["rolling", "strength"], default="rolling")
    p_out.add_argument("--lr", type=float, default=0.05)
    p_out.add_argument("--max-iter", type=int, default=250)
    # Dixon–Coles + training history controls
    p_out.add_argument("--dc-rho", type=float, default=None, help="Dixon–Coles rho (e.g. -0.10).")
    p_out.add_argument(
        "--include-prev-seasons",
        type=int,
        default=0,
        help="How many previous seasons to include in training (0 = current season only).",
    )

    
    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate predictions vs results for a gameweek.")
    p_eval.add_argument("--season", type=int, required=True, help="Season year (e.g. 2025)")
    p_eval.add_argument("--gameweek", type=int, required=True, help="Premier League matchday (gameweek)")
    p_eval.add_argument("--append", action="store_true", help="Append per-match evaluation rows to ledger")
    p_eval.add_argument("--refresh-cumulative", action="store_true", help="Regenerate cumulative performance artifacts")
    
    # curate
    p_cur = sub.add_parser("curate", help="Merge processed season CSVs into a single curated dataset.")
    p_cur.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="Seasons to include (default: 2023 2024 2025)",
    )
    p_cur.add_argument("--competition-id", type=int, default=2021, help="competition id (default: 2021 PL)")
    p_cur.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")
    p_cur.add_argument("--curated-dir", type=str, default="data/curated", help="Curated output directory")
    p_cur.add_argument("--output", type=str, default=None, help="Override output filename (optional)")
    p_cur.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv", help="Output format (default: csv)")
    # performance
    p_perf = sub.add_parser("performance", help="Generate cumulative performance artifacts from the evaluation ledger.")
    p_perf.add_argument("--season", type=int, default=None, help="Filter to a single season (optional)")

    
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


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = FootballDataConfig(
        competition_id=args.competition_id,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
    )

    seasons = _resolve_seasons(args)

    wrote = []
    failed = []

    for season in seasons:
        try:
            out_csv = fetch_season_matches(
                season=season,
                cfg=cfg,
                force_refresh=args.force_refresh,
            )
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

    )

    out_path = render_gameweek_outlook(
        season=args.season,
        gameweek=args.gameweek,
        competition_id=args.competition_id,
        cfg=cfg,
        save_predictions=args.save_predictions,
    )

    print(f"OK: wrote {out_path}")
    return 0

def cmd_evaluate(args: argparse.Namespace) -> int:
    df, summary, out_md = evaluate_gameweek(
        season=args.season,
        gameweek=args.gameweek,
        append=args.append,
        refresh_cumulative=args.refresh_cumulative,
    )
    print(f"OK: wrote {out_md}")
    if args.append:
        print("OK: appended to data/evaluation/all_matches.csv")
    if args.refresh_cumulative:
        print("OK: refreshed cumulative performance artifacts in reports/")
    return 0

def cmd_curate(args: argparse.Namespace) -> int:
    processed_dir = Path(args.processed_dir)
    curated_dir = Path(args.curated_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)

    seasons = sorted(set(args.seasons))

    # Input file naming contract comes from fetch.py
    # data/processed/matches_comp_{competition_id}_season_{season}.csv
    in_paths = []
    for s in seasons:
        p = processed_dir / f"matches_comp_{args.competition_id}_season_{s}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file for season {s}: {p}")
        in_paths.append(p)

    dfs = []
    for p in in_paths:
        df = pd.read_csv(p, parse_dates=["utc_date"])
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # De-dupe and sort deterministically
    if "match_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["match_id"], keep="last")
    sort_cols = [c for c in ["utc_date", "match_id"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    # Output name
    if args.output:
        out_name = args.output
    else:
        out_name = f"matches_comp_{args.competition_id}_seasons_{seasons[0]}_{seasons[-1]}.{args.format}"

    out_path = curated_dir / out_name

    if args.format == "csv":
        merged.to_csv(out_path, index=False)
    else:
        # Parquet is great, but requires pyarrow or fastparquet.
        try:
            merged.to_parquet(out_path, index=False)
        except Exception as e:
            raise RuntimeError(
                "Failed to write parquet. Install pyarrow (recommended) or use --format csv."
            ) from e

    # Write a lightweight manifest for traceability
    manifest = {
        "competition_id": args.competition_id,
        "seasons": seasons,
        "input_files": [str(p) for p in in_paths],
        "output_file": str(out_path),
        "row_count": int(len(merged)),
        "column_count": int(len(merged.columns)),
        "dedupe_key": "match_id" if "match_id" in merged.columns else None,
    }
    manifest_path = curated_dir / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
