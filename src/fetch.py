# src/fetch.py

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests
import pandas as pd


@dataclass(frozen=True)
class FootballDataConfig:
    api_base: str = "https://api.football-data.org/v4"
    token_env_var: str = "FOOTBALL_DATA_TOKEN"
    competition_id: int = 2021  # Premier League on football-data.org (commonly 2021)
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    timeout_s: int = 30
    max_retries: int = 4
    backoff_s: float = 1.5


class FootballDataAPIError(RuntimeError):
    pass


def _ensure_dirs(cfg: FootballDataConfig) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def _get_token(cfg: FootballDataConfig) -> str:
    token = os.getenv(cfg.token_env_var)
    if not token:
        raise ValueError(
            f"Missing API token. Set environment variable {cfg.token_env_var}."
        )
    return token


def _request_with_retries(
    url: str,
    headers: Dict[str, str],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)

            # Handle rate limiting and transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = FootballDataAPIError(
                    f"HTTP {resp.status_code} for {url} (attempt {attempt})"
                )
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else (backoff_s ** attempt)
                time.sleep(sleep_s)
                continue

            if resp.status_code != 200:
                raise FootballDataAPIError(
                    f"HTTP {resp.status_code} for {url}: {resp.text[:300]}"
                )

            return resp.json()

        except (requests.RequestException, FootballDataAPIError) as e:
            last_err = e
            time.sleep(backoff_s ** attempt)

    raise FootballDataAPIError(f"Failed after {max_retries} retries: {last_err}")


def normalize_matches(payload: Dict[str, Any], season: int, competition_id: int) -> pd.DataFrame:
    """
    Normalize football-data.org /competitions/{id}/matches payload into a flat table.

    Output columns (stable contract):
    - match_id
    - season
    - competition_id
    - matchday
    - utc_date
    - status
    - home_team_id, home_team_name
    - away_team_id, away_team_name
    - home_goals_ft, away_goals_ft
    """
    matches: List[Dict[str, Any]] = payload.get("matches", [])
    rows: List[Dict[str, Any]] = []

    for m in matches:
        score = m.get("score") or {}
        ft = score.get("fullTime") or {}
        home_team = m.get("homeTeam") or {}
        away_team = m.get("awayTeam") or {}

        rows.append(
            {
                "match_id": m.get("id"),
                "season": season,
                "competition_id": competition_id,
                "matchday": m.get("matchday"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "home_team_id": home_team.get("id"),
                "home_team_name": home_team.get("name"),
                "away_team_id": away_team.get("id"),
                "away_team_name": away_team.get("name"),
                "home_goals_ft": ft.get("home"),
                "away_goals_ft": ft.get("away"),
            }
        )

    df = pd.DataFrame(rows)

    # Basic sanity / types
    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
        # matchday sometimes null for weird entries; keep as Int64
        df["matchday"] = df["matchday"].astype("Int64")
        df["match_id"] = df["match_id"].astype("Int64")
        df["home_team_id"] = df["home_team_id"].astype("Int64")
        df["away_team_id"] = df["away_team_id"].astype("Int64")
        df["home_goals_ft"] = df["home_goals_ft"].astype("Int64")
        df["away_goals_ft"] = df["away_goals_ft"].astype("Int64")

        # Sort for deterministic outputs
        df = df.sort_values(["utc_date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    return df


def fetch_season_matches(
    season: int,
    cfg: FootballDataConfig = FootballDataConfig(),
    force_refresh: bool = False,
) -> Path:
    """
    Fetch all Premier League matches for a season, cache raw JSON, and write normalized CSV.

    Returns: path to processed CSV file.
    """
    _ensure_dirs(cfg)
    token = _get_token(cfg)

    raw_path = cfg.raw_dir / f"football_data_comp_{cfg.competition_id}_season_{season}.json"
    out_csv = cfg.processed_dir / f"matches_comp_{cfg.competition_id}_season_{season}.csv"

    # If cached and not forcing refresh, reuse
    if raw_path.exists() and out_csv.exists() and not force_refresh:
        return out_csv

    url = f"{cfg.api_base}/competitions/{cfg.competition_id}/matches"
    headers = {"X-Auth-Token": token}

    payload = _request_with_retries(
        url=url,
        headers=headers,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        backoff_s=cfg.backoff_s,
        params={"season": season},
    )

    # Cache raw response for reproducibility
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    df = normalize_matches(payload, season=season, competition_id=cfg.competition_id)

    # Write CSV (deterministic)
    df.to_csv(out_csv, index=False)

    return out_csv

def fetch_multiple_seasons(
    seasons: list[int],
    cfg: FootballDataConfig = FootballDataConfig(),
    force_refresh: bool = False,
) -> list[Path]:
    outputs = []
    for season in seasons:
        try:
            out = fetch_season_matches(
                season=season,
                cfg=cfg,
                force_refresh=force_refresh,
            )
            outputs.append(out)
        except Exception as e:
            print(f"[WARN] Failed season {season}: {e}")
    return outputs

if __name__ == "__main__":
    # quick manual run:
    # FOOTBALL_DATA_TOKEN=... python src/fetch.py
    csv_path = fetch_season_matches(season=2025)
    # csv_path = fetch_multiple_seasons(list(range(2018, 2026)))
    print(f"Wrote: {csv_path}")
