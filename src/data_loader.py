# src/data_loader.py
# Canonical data loading utilities shared across the project.

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_matches_for_season(
    season: int,
    competition_id: int = 2021,
    processed_dir: Path = Path("data/processed"),
    curated_dir: Path = Path("data/curated"),
) -> pd.DataFrame:
    """
    Load matches for a single season.
    Prefer curated merged dataset; fallback to per-season processed CSV.
    """
    curated_candidates = sorted(curated_dir.glob(f"matches_comp_{competition_id}_seasons_*.csv"))
    if curated_candidates:
        df_all = pd.read_csv(curated_candidates[-1], parse_dates=["utc_date"])
        df = df_all[df_all["season"] == season].copy()
        if not df.empty:
            return df.sort_values(["utc_date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    csv_path = processed_dir / f"matches_comp_{competition_id}_season_{season}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No data found for season {season} in {processed_dir} or {curated_dir}")
    df = pd.read_csv(csv_path, parse_dates=["utc_date"])
    return df.sort_values(["utc_date", "match_id"], ascending=[True, True]).reset_index(drop=True)


def load_training_matches(
    season: int,
    competition_id: int = 2021,
    include_prev_seasons: int = 0,
    processed_dir: Path = Path("data/processed"),
    curated_dir: Path = Path("data/curated"),
) -> pd.DataFrame:
    """
    Load training data: current season + optionally N previous seasons.
    """
    seasons = [season - i for i in range(include_prev_seasons, -1, -1)]
    dfs = []
    for s in seasons:
        try:
            dfs.append(load_matches_for_season(s, competition_id, processed_dir, curated_dir))
        except FileNotFoundError:
            continue
    if not dfs:
        return load_matches_for_season(season, competition_id, processed_dir, curated_dir)
    df = pd.concat(dfs, ignore_index=True)
    sort_cols = [c for c in ["utc_date", "match_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    return df


def curate_seasons(
    seasons: list[int],
    competition_id: int = 2021,
    processed_dir: Path = Path("data/processed"),
    curated_dir: Path = Path("data/curated"),
    out_format: str = "csv",
) -> tuple[Path, Path]:
    """
    Merge processed season CSVs into a single curated dataset + manifest.
    Returns (curated_path, manifest_path).
    """
    import json

    curated_dir.mkdir(parents=True, exist_ok=True)
    seasons = sorted(set(seasons))

    in_paths = []
    for s in seasons:
        p = processed_dir / f"matches_comp_{competition_id}_season_{s}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file for season {s}: {p}")
        in_paths.append(p)

    dfs = [pd.read_csv(p, parse_dates=["utc_date"]) for p in in_paths]
    merged = pd.concat(dfs, ignore_index=True)

    if "match_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["match_id"], keep="last")

    sort_cols = [c for c in ["utc_date", "match_id"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out_name = f"matches_comp_{competition_id}_seasons_{seasons[0]}_{seasons[-1]}.{out_format}"
    out_path = curated_dir / out_name

    if out_format == "csv":
        merged.to_csv(out_path, index=False)
    elif out_format == "parquet":
        merged.to_parquet(out_path, index=False)
    else:
        raise ValueError("out_format must be 'csv' or 'parquet'")

    manifest = {
        "competition_id": competition_id,
        "seasons": seasons,
        "input_files": [str(p) for p in in_paths],
        "output_file": str(out_path),
        "row_count": int(len(merged)),
        "column_count": int(len(merged.columns)),
        "dedupe_key": "match_id" if "match_id" in merged.columns else None,
    }
    manifest_path = curated_dir / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return out_path, manifest_path
