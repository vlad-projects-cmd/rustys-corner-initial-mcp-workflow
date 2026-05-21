# tests/test_competitions.py

import pytest

from src.competitions import (
    resolve_competition,
    get_competition_by_code,
    get_competition_by_id,
    list_competitions,
)


class TestResolveCompetition:
    def test_default_is_premier_league(self):
        assert resolve_competition(None, None) == 2021

    def test_league_code_pl(self):
        assert resolve_competition("pl", None) == 2021

    def test_league_code_championship(self):
        assert resolve_competition("championship", None) == 2016

    def test_league_code_brasileirao(self):
        assert resolve_competition("brasileirao", None) == 2013

    def test_league_code_worldcup(self):
        assert resolve_competition("worldcup", None) == 2000

    def test_league_code_case_insensitive(self):
        assert resolve_competition("PL", None) == 2021
        assert resolve_competition("LaLiga", None) == 2014

    def test_numeric_id_passthrough(self):
        assert resolve_competition(None, 9999) == 9999

    def test_league_takes_precedence_over_id(self):
        assert resolve_competition("laliga", 2021) == 2014

    def test_unknown_league_raises(self):
        with pytest.raises(ValueError, match="Unknown league code"):
            resolve_competition("fake_league", None)


class TestGetCompetition:
    def test_by_code(self):
        c = get_competition_by_code("bundesliga")
        assert c is not None
        assert c.id == 2002
        assert c.country == "Germany"

    def test_by_id(self):
        c = get_competition_by_id(2019)
        assert c is not None
        assert c.code == "seriea"

    def test_unknown_returns_none(self):
        assert get_competition_by_code("xyz") is None
        assert get_competition_by_id(99999) is None


class TestListCompetitions:
    def test_returns_all_free_tier(self):
        comps = list_competitions()
        assert len(comps) == 13

    def test_all_major_leagues_present(self):
        codes = {c.code for c in list_competitions()}
        assert "pl" in codes
        assert "laliga" in codes
        assert "bundesliga" in codes
        assert "seriea" in codes
        assert "ligue1" in codes
        assert "eredivisie" in codes
        assert "primeira" in codes
        assert "championship" in codes
        assert "brasileirao" in codes

    def test_cups_present(self):
        codes = {c.code for c in list_competitions()}
        assert "ucl" in codes
        assert "euro" in codes
        assert "worldcup" in codes
        assert "libertadores" in codes

    def test_brasileirao_is_calendar(self):
        c = get_competition_by_code("brasileirao")
        assert c.season_pattern == "calendar"

    def test_pl_is_split(self):
        c = get_competition_by_code("pl")
        assert c.season_pattern == "split"
