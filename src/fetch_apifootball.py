# src/fetch_apifootball.py
# Fetcher for API-Football (api-sports.io) — free tier: 100 requests/day.
# Produces the same normalized CSV format as fetch.py so downstream code works unchanged.

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# API-Football league IDs for currently active (calendar-year) leagues
APIFOOTBALL_LEAGUES: Dict[str, int] = {
    "mls": 253,
    "brasileirao": 71,
    "brasileirao_b": 72,
    "j1_league": 98,
    "k_league": 292,
    "argentina": 128,
    "liga_mx": 262,
    "libertadores": 13,
    "sudamericana": 11,
    "a_league": 188,
}


@dataclass(frozen=True)
class APIFootballConfig:
    api_base: str = "https://v3.football.api-sports.io"
    token_env_var: str = "APIFOOTBALL_TOKEN"
    league_id: int = 71  # default: Brazilian Serie A
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    timeout_s: int = 30
    max_retries: int = 4
    backoff_s: float = 1.5


class APIFootballError(RuntimeError):
    pass


def _ensure_dirs(cfg: APIFootballConfig) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def _get_token(cfg: APIFootballConfig) -> str:
    token = os.getenv(cfg.token_env_var)
    if not token:
        raise ValueError(
            f"Missing API token. Set environment variable {cfg.token_env_var}. "
            f"Get a free key at https://www.api-football.com/"
        )
    return token


def _request_with_retries(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)

            if resp.status_code == 429:
                last_err = APIFootballError(f"Rate limited (attempt {attempt})")
                time.sleep(backoff_s ** attempt)
                continue

            if resp.status_code in (500, 502, 503, 504):
                last_err = APIFootballError(
                    f"HTTP {resp.status_code} for {url} (attempt {attempt})"
                )
                time.sleep(backoff_s ** attempt)
                continue

            if resp.status_code != 200:
                raise APIFootballError(
                    f"HTTP {resp.status_code} for {url}: {resp.text[:300]}"
                )

            data = resp.json()

            # API-Football wraps errors in the response body
            errors = data.get("errors")
            if errors:
                raise APIFootballError(f"API error: {errors}")

            return data

        except (requests.RequestException, APIFootballError) as e:
            last_err = e
            time.sleep(backoff_s ** attempt)

    raise APIFootballError(f"Failed after {max_retries} retries: {last_err}")


def normalize_fixtures(payload: Dict[str, Any], season: int, league_id: int) -> pd.DataFrame:
    """
    Normalize API-Football /fixtures response into the same flat table format
    used by the rest of the pipeline.

    Output columns match fetch.py normalize_matches():
    - match_id, season, competition_id, matchday, utc_date, status
    - home_team_id, home_team_name, away_team_id, away_team_name
    - home_goals_ft, away_goals_ft
    """
    fixtures: List[Dict[str, Any]] = payload.get("response", [])
    rows: List[Dict[str, Any]] = []

    # Map API-Football status codes to football-data.org style
    STATUS_MAP = {
        "NS": "SCHEDULED",
        "TBD": "SCHEDULED",
        "1H": "IN_PLAY",
        "HT": "IN_PLAY",
        "2H": "IN_PLAY",
        "ET": "IN_PLAY",
        "P": "IN_PLAY",
        "FT": "FINISHED",
        "AET": "FINISHED",
        "PEN": "FINISHED",
        "PST": "POSTPONED",
        "CANC": "CANCELLED",
        "ABD": "CANCELLED",
        "AWD": "FINISHED",
        "WO": "FINISHED",
        "SUSP": "SUSPENDED",
        "INT": "IN_PLAY",
        "LIVE": "IN_PLAY",
    }

    for fix in fixtures:
        fixture_info = fix.get("fixture", {})
        league_info = fix.get("league", {})
        teams = fix.get("teams", {})
        goals = fix.get("goals", {})

        status_short = (fixture_info.get("status") or {}).get("short", "")
        mapped_status = STATUS_MAP.get(status_short, status_short)

        home = teams.get("home", {})
        away = teams.get("away", {})

        rows.append(
            {
                "match_id": fixture_info.get("id"),
                "season": season,
                "competition_id": league_id,
                "matchday": league_info.get("round"),
                "utc_date": fixture_info.get("date"),
                "status": mapped_status,
                "home_team_id": home.get("id"),
                "home_team_name": home.get("name"),
                "away_team_id": away.get("id"),
                "away_team_name": away.get("name"),
                "home_goals_ft": goals.get("home"),
                "away_goals_ft": goals.get("away"),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
        df["match_id"] = df["match_id"].astype("Int64")
        df["home_team_id"] = df["home_team_id"].astype("Int64")
        df["away_team_id"] = df["away_team_id"].astype("Int64")
        df["home_goals_ft"] = df["home_goals_ft"].astype("Int64")
        df["away_goals_ft"] = df["away_goals_ft"].astype("Int64")

        # Extract numeric matchday from round string like "Regular Season - 17"
        df["matchday"] = df["matchday"].apply(_parse_matchday)
        df["matchday"] = df["matchday"].astype("Int64")

        df = df.sort_values(["utc_date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    return df


def _parse_matchday(round_str: Any) -> Optional[int]:
    """Extract numeric matchday from API-Football round string."""
    if round_str is None:
        return None
    if isinstance(round_str, int):
        return round_str
    s = str(round_str)
    # Common formats: "Regular Season - 17", "1st Round", "Group Stage - 5"
    parts = s.rsplit("-", 1)
    if len(parts) == 2:
        try:
            return int(parts[1].strip())
        except ValueError:
            pass
    # Try extracting any trailing number
    import re
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def fetch_season_fixtures(
    season: int,
    cfg: APIFootballConfig = APIFootballConfig(),
    force_refresh: bool = False,
) -> Path:
    """
    Fetch all fixtures for a league/season from API-Football.
    Caches raw JSON and writes normalized CSV.

    Returns: path to processed CSV file.
    """
    _ensure_dirs(cfg)
    token = _get_token(cfg)

    raw_path = cfg.raw_dir / f"apifootball_league_{cfg.league_id}_season_{season}.json"
    out_csv = cfg.processed_dir / f"matches_comp_{cfg.league_id}_season_{season}.csv"

    if raw_path.exists() and out_csv.exists() and not force_refresh:
        return out_csv

    url = f"{cfg.api_base}/fixtures"
    headers = {"x-apisports-key": token}
    params = {"league": cfg.league_id, "season": season}

    payload = _request_with_retries(
        url=url,
        headers=headers,
        params=params,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        backoff_s=cfg.backoff_s,
    )

    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    df = normalize_fixtures(payload, season=season, league_id=cfg.league_id)
    df.to_csv(out_csv, index=False)

    return out_csv


def fetch_multiple_seasons(
    seasons: list[int],
    cfg: APIFootballConfig = APIFootballConfig(),
    force_refresh: bool = False,
) -> list[Path]:
    outputs = []
    for season in seasons:
        try:
            out = fetch_season_fixtures(season=season, cfg=cfg, force_refresh=force_refresh)
            outputs.append(out)
        except Exception as e:
            print(f"[WARN] Failed season {season}: {e}")
    return outputs


if __name__ == "__main__":
    # quick manual run:
    # APIFOOTBALL_TOKEN=... python -m src.fetch_apifootball
    csv_path = fetch_season_fixtures(season=2026)
    print(f"Wrote: {csv_path}")
