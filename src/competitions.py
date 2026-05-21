# src/competitions.py
# Registry of known football-data.org competitions.
# Based on TIER_ONE (free plan) availability.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Competition:
    id: int
    code: str  # short alias for CLI usage
    name: str
    country: str
    type: str  # "LEAGUE" or "CUP"
    season_pattern: str  # "calendar" (Jan-Dec) or "split" (Aug-May)


# football-data.org v4 competition IDs — TIER_ONE (free plan)
# Source: GET /v4/competitions (with your token)
COMPETITIONS: list[Competition] = [
    # England
    Competition(id=2021, code="pl", name="Premier League", country="England", type="LEAGUE", season_pattern="split"),
    Competition(id=2016, code="championship", name="Championship", country="England", type="LEAGUE", season_pattern="split"),
    # Spain
    Competition(id=2014, code="laliga", name="Primera Division", country="Spain", type="LEAGUE", season_pattern="split"),
    # Germany
    Competition(id=2002, code="bundesliga", name="Bundesliga", country="Germany", type="LEAGUE", season_pattern="split"),
    # Italy
    Competition(id=2019, code="seriea", name="Serie A", country="Italy", type="LEAGUE", season_pattern="split"),
    # France
    Competition(id=2015, code="ligue1", name="Ligue 1", country="France", type="LEAGUE", season_pattern="split"),
    # Netherlands
    Competition(id=2003, code="eredivisie", name="Eredivisie", country="Netherlands", type="LEAGUE", season_pattern="split"),
    # Portugal
    Competition(id=2017, code="primeira", name="Primeira Liga", country="Portugal", type="LEAGUE", season_pattern="split"),
    # Brazil
    Competition(id=2013, code="brasileirao", name="Campeonato Brasileiro Serie A", country="Brazil", type="LEAGUE", season_pattern="calendar"),
    # International / Continental cups
    Competition(id=2001, code="ucl", name="UEFA Champions League", country="Europe", type="CUP", season_pattern="split"),
    Competition(id=2018, code="euro", name="European Championship", country="Europe", type="CUP", season_pattern="split"),
    Competition(id=2152, code="libertadores", name="Copa Libertadores", country="South America", type="CUP", season_pattern="calendar"),
    Competition(id=2000, code="worldcup", name="FIFA World Cup", country="World", type="CUP", season_pattern="split"),
]

# Lookup indexes
_BY_CODE: dict[str, Competition] = {c.code: c for c in COMPETITIONS}
_BY_ID: dict[int, Competition] = {c.id: c for c in COMPETITIONS}


def get_competition_by_code(code: str) -> Optional[Competition]:
    return _BY_CODE.get(code.lower().strip())


def get_competition_by_id(comp_id: int) -> Optional[Competition]:
    return _BY_ID.get(comp_id)


def resolve_competition(league: str | None, competition_id: int | None) -> int:
    """
    Resolve a competition ID from either a league code or numeric ID.
    --league takes precedence over --competition-id if both are provided.
    """
    if league:
        comp = get_competition_by_code(league)
        if comp is None:
            available = ", ".join(sorted(_BY_CODE.keys()))
            raise ValueError(f"Unknown league code '{league}'. Available: {available}")
        return comp.id
    if competition_id is not None:
        return competition_id
    return 2021  # default: Premier League


def list_competitions() -> list[Competition]:
    return list(COMPETITIONS)
