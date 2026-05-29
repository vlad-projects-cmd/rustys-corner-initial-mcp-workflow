# src/fetch_thesportsdb.py
# Fetcher for TheSportsDB free API.
# No registration required. Free public API key: "3".
# Covers calendar-year leagues: Swedish, Norwegian, Finnish, Japanese, Korean, MLS.
# Source: https://www.thesportsdb.com/free_sports_api

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


@dataclass(frozen=True)
class TheSportsDBLeague:
    id: int         # TheSportsDB league ID
    code: str       # CLI code for --league
    name: str
    country: str
    season_format: str  # "calendar" -> "2025", "split" -> "2024-2025"
    total_rounds: int   # expected number of rounds in a season


THESPORTSDB_LEAGUES: Dict[str, TheSportsDBLeague] = {}

_LEAGUES_DEF = [
    (4347, "se-allsvenskan", "Allsvenskan", "Sweden", "calendar", 30),
    (4358, "no-eliteserien", "Eliteserien", "Norway", "calendar", 30),
    (4636, "fi-veikkausliiga", "Veikkausliiga", "Finland", "calendar", 26),
    (4633, "jp-j1league", "J1 League", "Japan", "calendar", 34),
    (4689, "kr-kleague", "K League 1", "South Korea", "calendar", 33),
    (4346, "us-mls", "MLS", "USA", "calendar", 34),
]

for _id, _code, _name, _country, _fmt, _rounds in _LEAGUES_DEF:
    THESPORTSDB_LEAGUES[_code] = TheSportsDBLeague(
        id=_id, code=_code, name=_name, country=_country,
        season_format=_fmt, total_rounds=_rounds,
    )


@dataclass(frozen=True)
class TheSportsDBConfig:
    api_base: str = "https://www.thesportsdb.com/api/v1/json/3"
    league_code: str = "se-allsvenskan"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    timeout_s: int = 30
    max_retries: int = 3
    backoff_s: float = 1.0
    request_delay_s: float = 0.5  # be polite to free API
    verify_ssl: bool = True       # set False for corporate proxy/cert issues


class TheSportsDBError(RuntimeError):
    pass


def _ensure_dirs(cfg: TheSportsDBConfig) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def _get_league(cfg: TheSportsDBConfig) -> TheSportsDBLeague:
    league = THESPORTSDB_LEAGUES.get(cfg.league_code)
    if league is None:
        available = ", ".join(sorted(THESPORTSDB_LEAGUES.keys()))
        raise ValueError(
            f"Unknown TheSportsDB league code '{cfg.league_code}'. Available: {available}"
        )
    return league


def _request_with_retries(
    url: str,
    params: Dict[str, Any],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    verify_ssl: bool = True,
) -> Optional[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s, verify=verify_ssl)

            if resp.status_code == 429:
                last_err = TheSportsDBError(f"Rate limited (attempt {attempt})")
                wait = max(30, backoff_s * attempt * 10)
                print(f"  [rate limited] waiting {wait:.0f}s before retry...")
                time.sleep(wait)
                continue

            if resp.status_code in (500, 502, 503, 504):
                last_err = TheSportsDBError(f"HTTP {resp.status_code} (attempt {attempt})")
                time.sleep(backoff_s * attempt)
                continue

            if resp.status_code != 200:
                raise TheSportsDBError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            return resp.json()

        except (requests.RequestException, TheSportsDBError) as e:
            last_err = e
            time.sleep(backoff_s * attempt)

    raise TheSportsDBError(f"Failed after {max_retries} retries: {last_err}")


def _season_string(season: int, league: TheSportsDBLeague) -> str:
    """Convert season year to TheSportsDB season format."""
    if league.season_format == "split":
        return f"{season}-{season + 1}"
    return str(season)


def fetch_round_events(
    league: TheSportsDBLeague,
    season: int,
    round_num: int,
    cfg: TheSportsDBConfig,
) -> List[Dict[str, Any]]:
    """Fetch events for a single round."""
    url = f"{cfg.api_base}/eventsround.php"
    season_str = _season_string(season, league)
    params = {"id": league.id, "r": round_num, "s": season_str}

    data = _request_with_retries(
        url=url,
        params=params,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        backoff_s=cfg.backoff_s,
        verify_ssl=cfg.verify_ssl,
    )

    if data is None:
        return []

    events = data.get("events") or []
    return events


def normalize_events(
    all_events: List[Dict[str, Any]],
    season: int,
    league: TheSportsDBLeague,
) -> pd.DataFrame:
    """
    Normalize TheSportsDB events into the standard pipeline CSV format.

    Output columns match the project convention:
    - match_id, season, competition_id, matchday, utc_date, status
    - home_team_id, home_team_name, away_team_id, away_team_name
    - home_goals_ft, away_goals_ft
    """
    rows: List[Dict[str, Any]] = []

    STATUS_MAP = {
        "FT": "FINISHED",
        "AET": "FINISHED",
        "PEN": "FINISHED",
        "AP": "FINISHED",
        "Match Finished": "FINISHED",
        "NS": "SCHEDULED",
        "Not Started": "SCHEDULED",
        "PST": "POSTPONED",
        "Postponed": "POSTPONED",
        "CANC": "CANCELLED",
        "Cancelled": "CANCELLED",
        "1H": "IN_PLAY",
        "2H": "IN_PLAY",
        "HT": "IN_PLAY",
    }

    for ev in all_events:
        status_raw = ev.get("strStatus") or ""
        mapped_status = STATUS_MAP.get(status_raw, status_raw)
        # If status field is empty but we have scores, treat as finished
        if not mapped_status and ev.get("intHomeScore") is not None:
            mapped_status = "FINISHED"
        elif not mapped_status:
            mapped_status = "SCHEDULED"

        home_score = ev.get("intHomeScore")
        away_score = ev.get("intAwayScore")

        # Parse round number
        round_val = ev.get("intRound")
        try:
            matchday = int(round_val) if round_val is not None else None
        except (ValueError, TypeError):
            matchday = None

        # Parse team IDs
        try:
            home_team_id = int(ev.get("idHomeTeam") or 0)
        except (ValueError, TypeError):
            home_team_id = 0
        try:
            away_team_id = int(ev.get("idAwayTeam") or 0)
        except (ValueError, TypeError):
            away_team_id = 0

        # Parse scores
        try:
            home_goals = int(home_score) if home_score is not None else None
        except (ValueError, TypeError):
            home_goals = None
        try:
            away_goals = int(away_score) if away_score is not None else None
        except (ValueError, TypeError):
            away_goals = None

        # Build date string
        date_str = ev.get("dateEvent")
        time_str = ev.get("strTime") or "00:00:00"
        if date_str:
            utc_date = f"{date_str}T{time_str}+00:00"
        else:
            utc_date = None

        rows.append(
            {
                "match_id": int(ev.get("idEvent") or 0),
                "season": season,
                "competition_id": league.id,
                "matchday": matchday,
                "utc_date": utc_date,
                "status": mapped_status,
                "home_team_id": home_team_id,
                "home_team_name": ev.get("strHomeTeam") or "",
                "away_team_id": away_team_id,
                "away_team_name": ev.get("strAwayTeam") or "",
                "home_goals_ft": home_goals,
                "away_goals_ft": away_goals,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
        df["match_id"] = df["match_id"].astype("Int64")
        df["matchday"] = df["matchday"].astype("Int64")
        df["home_team_id"] = df["home_team_id"].astype("Int64")
        df["away_team_id"] = df["away_team_id"].astype("Int64")
        df["home_goals_ft"] = df["home_goals_ft"].astype("Int64")
        df["away_goals_ft"] = df["away_goals_ft"].astype("Int64")
        df = df.sort_values(["utc_date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    return df


def fetch_season_matches(
    season: int,
    cfg: TheSportsDBConfig = TheSportsDBConfig(),
    force_refresh: bool = False,
) -> Path:
    """
    Fetch all matches for a league/season from TheSportsDB (round by round).
    Caches raw JSON and writes normalized CSV.

    Returns: path to processed CSV file.
    """
    _ensure_dirs(cfg)
    league = _get_league(cfg)

    season_str = _season_string(season, league)
    raw_path = cfg.raw_dir / f"thesportsdb_{league.code}_{season_str}.json"
    out_csv = cfg.processed_dir / f"matches_comp_{league.id}_{season_str}.csv"

    if raw_path.exists() and out_csv.exists() and not force_refresh:
        return out_csv

    all_events: List[Dict[str, Any]] = []
    empty_rounds = 0

    print(f"  Fetching {league.name} {season_str} round-by-round...")

    for round_num in range(1, league.total_rounds + 1):
        events = fetch_round_events(league, season, round_num, cfg)
        if events:
            all_events.extend(events)
            empty_rounds = 0
        else:
            empty_rounds += 1
            # Stop early if we hit 3 consecutive empty rounds (season not that far yet)
            if empty_rounds >= 3:
                break

        # Polite delay between requests
        time.sleep(cfg.request_delay_s)

    if not all_events:
        raise TheSportsDBError(
            f"No events found for {league.name} season {season_str}. "
            f"Season may not exist or hasn't started."
        )

    # Cache raw response
    raw_path.write_text(json.dumps(all_events, indent=2), encoding="utf-8")

    df = normalize_events(all_events, season=season, league=league)
    df.to_csv(out_csv, index=False)

    print(f"  -> {len(df)} matches ({df['status'].value_counts().to_dict()})")

    return out_csv


def fetch_multiple_seasons(
    seasons: list[int],
    cfg: TheSportsDBConfig = TheSportsDBConfig(),
    force_refresh: bool = False,
) -> list[Path]:
    outputs = []
    for season in seasons:
        try:
            out = fetch_season_matches(season=season, cfg=cfg, force_refresh=force_refresh)
            outputs.append(out)
        except TheSportsDBError as e:
            print(f"[WARN] {cfg.league_code} season {season}: {e}")
        except Exception as e:
            print(f"[WARN] {cfg.league_code} season {season} unexpected: {e}")
    return outputs


def list_available_leagues() -> list[TheSportsDBLeague]:
    return sorted(THESPORTSDB_LEAGUES.values(), key=lambda l: (l.country, l.code))


if __name__ == "__main__":
    # Quick test: fetch Swedish Allsvenskan 2025
    csv_path = fetch_season_matches(season=2025)
    print(f"Wrote: {csv_path}")
