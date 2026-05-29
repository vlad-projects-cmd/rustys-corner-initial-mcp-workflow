# src/fetch_openfootball.py
# Fetcher for openfootball/football.json GitHub data.
# No API key required. Public domain (CC0). Historical data from 2010-11 onwards.
# Source: https://github.com/openfootball/football.json

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# League file codes used in the openfootball repo
# URL pattern: https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json
OPENFOOTBALL_LEAGUES: Dict[str, "OpenFootballLeague"] = {}


@dataclass(frozen=True)
class OpenFootballLeague:
    code: str          # CLI code for --league
    file_code: str     # filename in repo (e.g. "en.1")
    name: str
    country: str
    season_pattern: str  # "split" (2021-22) or "calendar" (2025)

    def season_str(self, season: int) -> str:
        """Convert numeric season year to the folder name used in the repo."""
        if self.season_pattern == "split":
            # e.g. 2021 -> "2021-22"
            next_yr = (season + 1) % 100
            return f"{season}-{next_yr:02d}"
        else:
            # Calendar year leagues use just the year
            return str(season)


# Register all known leagues
_LEAGUES_DEF = [
    # England
    ("en-pl", "en.1", "Premier League", "England", "split"),
    ("en-championship", "en.2", "Championship", "England", "split"),
    ("en-league1", "en.3", "League One", "England", "split"),
    ("en-league2", "en.4", "League Two", "England", "split"),
    # Germany
    ("de-bundesliga", "de.1", "Bundesliga", "Germany", "split"),
    ("de-2bundesliga", "de.2", "2. Bundesliga", "Germany", "split"),
    # Spain
    ("es-laliga", "es.1", "La Liga", "Spain", "split"),
    ("es-segunda", "es.2", "Segunda Division", "Spain", "split"),
    # Italy
    ("it-seriea", "it.1", "Serie A", "Italy", "split"),
    ("it-serieb", "it.2", "Serie B", "Italy", "split"),
    # France
    ("fr-ligue1", "fr.1", "Ligue 1", "France", "split"),
    ("fr-ligue2", "fr.2", "Ligue 2", "France", "split"),
]

for code, file_code, name, country, pattern in _LEAGUES_DEF:
    OPENFOOTBALL_LEAGUES[code] = OpenFootballLeague(
        code=code, file_code=file_code, name=name,
        country=country, season_pattern=pattern,
    )


RAW_BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"


@dataclass(frozen=True)
class OpenFootballConfig:
    league_code: str = "en-pl"  # default: Premier League
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    timeout_s: int = 30
    max_retries: int = 3
    backoff_s: float = 1.0


class OpenFootballError(RuntimeError):
    pass


def _ensure_dirs(cfg: OpenFootballConfig) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def _get_league(cfg: OpenFootballConfig) -> OpenFootballLeague:
    league = OPENFOOTBALL_LEAGUES.get(cfg.league_code)
    if league is None:
        available = ", ".join(sorted(OPENFOOTBALL_LEAGUES.keys()))
        raise ValueError(
            f"Unknown openfootball league code '{cfg.league_code}'. Available: {available}"
        )
    return league


def _fetch_json(url: str, timeout_s: int, max_retries: int, backoff_s: float) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code == 404:
                raise OpenFootballError(f"Not found: {url} (season may not exist in repo)")
            if resp.status_code in (429, 500, 502, 503):
                last_err = OpenFootballError(f"HTTP {resp.status_code} (attempt {attempt})")
                time.sleep(backoff_s * attempt)
                continue
            if resp.status_code != 200:
                raise OpenFootballError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(backoff_s * attempt)
    raise OpenFootballError(f"Failed after {max_retries} retries: {last_err}")


def _parse_round_number(round_str: Optional[str]) -> Optional[int]:
    """Extract numeric matchday from round string like 'Matchday 1' or 'Round 23'."""
    if not round_str:
        return None
    m = re.search(r"(\d+)", round_str)
    return int(m.group(1)) if m else None


def _generate_match_id(date: str, team1: str, team2: str) -> int:
    """Generate a stable numeric match ID from date + teams (no IDs in source data)."""
    key = f"{date}|{team1}|{team2}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def normalize_openfootball(
    payload: Dict[str, Any],
    season: int,
    league: OpenFootballLeague,
) -> pd.DataFrame:
    """
    Normalize openfootball JSON into the standard pipeline CSV format.

    Output columns match fetch.py normalize_matches():
    - match_id, season, competition_id, matchday, utc_date, status
    - home_team_id, home_team_name, away_team_id, away_team_name
    - home_goals_ft, away_goals_ft
    """
    matches: List[Dict[str, Any]] = payload.get("matches", [])
    rows: List[Dict[str, Any]] = []

    # Build stable team IDs from names
    team_names: set[str] = set()
    for m in matches:
        team_names.add(m.get("team1", ""))
        team_names.add(m.get("team2", ""))
    team_id_map = {name: idx for idx, name in enumerate(sorted(team_names), start=1)}

    current_round: Optional[str] = None

    for m in matches:
        # The "round" field appears at match level
        if "round" in m:
            current_round = m["round"]

        team1 = m.get("team1", "")
        team2 = m.get("team2", "")
        date_str = m.get("date")
        score = m.get("score")

        # score can be: dict {"ft": [h, a], "ht": [h, a]}, list [h, a], or None
        if isinstance(score, dict):
            ft = score.get("ft") or [None, None]
        elif isinstance(score, list):
            ft = score
        else:
            ft = [None, None]

        home_goals = ft[0] if len(ft) >= 2 else None
        away_goals = ft[1] if len(ft) >= 2 else None

        # Determine status
        if home_goals is not None and away_goals is not None:
            status = "FINISHED"
        else:
            status = "SCHEDULED"

        rows.append(
            {
                "match_id": _generate_match_id(date_str or "", team1, team2),
                "season": season,
                "competition_id": league.file_code,
                "matchday": _parse_round_number(current_round),
                "utc_date": date_str,
                "status": status,
                "home_team_id": team_id_map.get(team1, 0),
                "home_team_name": team1,
                "away_team_id": team_id_map.get(team2, 0),
                "away_team_name": team2,
                "home_goals_ft": home_goals,
                "away_goals_ft": away_goals,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
        df["matchday"] = df["matchday"].astype("Int64")
        df["match_id"] = df["match_id"].astype("Int64")
        df["home_team_id"] = df["home_team_id"].astype("Int64")
        df["away_team_id"] = df["away_team_id"].astype("Int64")
        df["home_goals_ft"] = df["home_goals_ft"].astype("Int64")
        df["away_goals_ft"] = df["away_goals_ft"].astype("Int64")
        df = df.sort_values(["utc_date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    return df


def fetch_season_matches(
    season: int,
    cfg: OpenFootballConfig = OpenFootballConfig(),
    force_refresh: bool = False,
) -> Path:
    """
    Fetch all matches for a league/season from openfootball/football.json on GitHub.
    Caches raw JSON and writes normalized CSV.

    Returns: path to processed CSV file.
    """
    _ensure_dirs(cfg)
    league = _get_league(cfg)

    season_str = league.season_str(season)
    raw_path = cfg.raw_dir / f"openfootball_{league.file_code}_{season_str}.json"
    out_csv = cfg.processed_dir / f"matches_comp_{league.file_code}_{season_str}.csv"

    if raw_path.exists() and out_csv.exists() and not force_refresh:
        return out_csv

    url = f"{RAW_BASE_URL}/{season_str}/{league.file_code}.json"

    payload = _fetch_json(
        url=url,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        backoff_s=cfg.backoff_s,
    )

    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    df = normalize_openfootball(payload, season=season, league=league)
    df.to_csv(out_csv, index=False)

    return out_csv


def fetch_multiple_seasons(
    seasons: list[int],
    cfg: OpenFootballConfig = OpenFootballConfig(),
    force_refresh: bool = False,
) -> list[Path]:
    outputs = []
    for season in seasons:
        try:
            out = fetch_season_matches(season=season, cfg=cfg, force_refresh=force_refresh)
            outputs.append(out)
            print(f"OK: {cfg.league_code} season {season} -> {out}")
        except OpenFootballError as e:
            print(f"[WARN] {cfg.league_code} season {season}: {e}")
        except Exception as e:
            print(f"[WARN] {cfg.league_code} season {season} unexpected: {e}")
    return outputs


def list_available_leagues() -> list[OpenFootballLeague]:
    return sorted(OPENFOOTBALL_LEAGUES.values(), key=lambda l: (l.country, l.code))


if __name__ == "__main__":
    # Quick test: fetch EPL 2023-24
    csv_path = fetch_season_matches(season=2023)
    print(f"Wrote: {csv_path}")
