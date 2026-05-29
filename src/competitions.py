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
    source: str = "football-data"  # "football-data" or "apifootball"


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

# API-Football (api-sports.io) leagues — mostly calendar-year leagues currently active
APIFOOTBALL_COMPETITIONS: list[Competition] = [
    Competition(id=253, code="mls", name="MLS", country="USA", type="LEAGUE", season_pattern="calendar", source="apifootball"),
    Competition(id=71, code="brasileirao-af", name="Brasileirao Serie A", country="Brazil", type="LEAGUE", season_pattern="calendar", source="apifootball"),
    Competition(id=98, code="j1league", name="J1 League", country="Japan", type="LEAGUE", season_pattern="calendar", source="apifootball"),
    Competition(id=292, code="kleague", name="K League 1", country="South Korea", type="LEAGUE", season_pattern="calendar", source="apifootball"),
    Competition(id=128, code="argentina", name="Liga Profesional", country="Argentina", type="LEAGUE", season_pattern="calendar", source="apifootball"),
    Competition(id=262, code="ligamx", name="Liga MX", country="Mexico", type="LEAGUE", season_pattern="split", source="apifootball"),
    Competition(id=13, code="libertadores-af", name="Copa Libertadores", country="South America", type="CUP", season_pattern="calendar", source="apifootball"),
    Competition(id=188, code="aleague", name="A-League", country="Australia", type="LEAGUE", season_pattern="split", source="apifootball"),
]

# OpenFootball (GitHub, no API key, historical data 2010+)
OPENFOOTBALL_COMPETITIONS: list[Competition] = [
    # England
    Competition(id=0, code="en-pl", name="Premier League", country="England", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="en-championship", name="Championship", country="England", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="en-league1", name="League One", country="England", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="en-league2", name="League Two", country="England", type="LEAGUE", season_pattern="split", source="openfootball"),
    # Germany
    Competition(id=0, code="de-bundesliga", name="Bundesliga", country="Germany", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="de-2bundesliga", name="2. Bundesliga", country="Germany", type="LEAGUE", season_pattern="split", source="openfootball"),
    # Spain
    Competition(id=0, code="es-laliga", name="La Liga", country="Spain", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="es-segunda", name="Segunda Division", country="Spain", type="LEAGUE", season_pattern="split", source="openfootball"),
    # Italy
    Competition(id=0, code="it-seriea", name="Serie A", country="Italy", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="it-serieb", name="Serie B", country="Italy", type="LEAGUE", season_pattern="split", source="openfootball"),
    # France
    Competition(id=0, code="fr-ligue1", name="Ligue 1", country="France", type="LEAGUE", season_pattern="split", source="openfootball"),
    Competition(id=0, code="fr-ligue2", name="Ligue 2", country="France", type="LEAGUE", season_pattern="split", source="openfootball"),
]

# TheSportsDB (free API, no registration, calendar-year leagues)
THESPORTSDB_COMPETITIONS: list[Competition] = [
    Competition(id=4347, code="se-allsvenskan", name="Allsvenskan", country="Sweden", type="LEAGUE", season_pattern="calendar", source="thesportsdb"),
    Competition(id=4358, code="no-eliteserien", name="Eliteserien", country="Norway", type="LEAGUE", season_pattern="calendar", source="thesportsdb"),
    Competition(id=4636, code="fi-veikkausliiga", name="Veikkausliiga", country="Finland", type="LEAGUE", season_pattern="calendar", source="thesportsdb"),
    Competition(id=4633, code="jp-j1league", name="J1 League", country="Japan", type="LEAGUE", season_pattern="calendar", source="thesportsdb"),
    Competition(id=4689, code="kr-kleague", name="K League 1", country="South Korea", type="LEAGUE", season_pattern="calendar", source="thesportsdb"),
    Competition(id=4346, code="us-mls", name="MLS", country="USA", type="LEAGUE", season_pattern="calendar", source="thesportsdb"),
]

ALL_COMPETITIONS = COMPETITIONS + APIFOOTBALL_COMPETITIONS + OPENFOOTBALL_COMPETITIONS + THESPORTSDB_COMPETITIONS

# Lookup indexes (include both sources)
_BY_CODE: dict[str, Competition] = {c.code: c for c in ALL_COMPETITIONS}
_BY_ID: dict[int, Competition] = {c.id: c for c in ALL_COMPETITIONS}


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


def list_competitions(source: Optional[str] = None) -> list[Competition]:
    if source == "football-data":
        return list(COMPETITIONS)
    if source == "apifootball":
        return list(APIFOOTBALL_COMPETITIONS)
    if source == "openfootball":
        return list(OPENFOOTBALL_COMPETITIONS)
    if source == "thesportsdb":
        return list(THESPORTSDB_COMPETITIONS)
    return list(ALL_COMPETITIONS)
