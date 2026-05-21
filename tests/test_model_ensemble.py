# tests/test_model_ensemble.py

import math
import pytest

from src.model_ensemble import ensemble_probabilities, ensemble_with_lambdas


class TestEnsembleProbabilities:
    def test_equal_weights_averages(self):
        p1 = {"p_home_win": 0.6, "p_draw": 0.2, "p_away_win": 0.2}
        p2 = {"p_home_win": 0.4, "p_draw": 0.3, "p_away_win": 0.3}

        result = ensemble_probabilities([p1, p2], [0.5, 0.5])
        assert math.isclose(result["p_home_win"], 0.5, rel_tol=1e-6)
        assert math.isclose(result["p_draw"], 0.25, rel_tol=1e-6)
        assert math.isclose(result["p_away_win"], 0.25, rel_tol=1e-6)

    def test_sums_to_one(self):
        p1 = {"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2}
        p2 = {"p_home_win": 0.7, "p_draw": 0.15, "p_away_win": 0.15}
        p3 = {"p_home_win": 0.4, "p_draw": 0.35, "p_away_win": 0.25}

        result = ensemble_probabilities([p1, p2, p3], [0.35, 0.35, 0.30])
        total = result["p_home_win"] + result["p_draw"] + result["p_away_win"]
        assert math.isclose(total, 1.0, rel_tol=1e-6)

    def test_single_model_passthrough(self):
        p1 = {"p_home_win": 0.55, "p_draw": 0.25, "p_away_win": 0.20}
        result = ensemble_probabilities([p1], [1.0])
        assert math.isclose(result["p_home_win"], 0.55, rel_tol=1e-6)

    def test_skips_none_predictions(self):
        p1 = {"p_home_win": 0.6, "p_draw": 0.2, "p_away_win": 0.2}
        result = ensemble_probabilities([p1, None], [0.5, 0.5])
        # Only p1 is used, so should equal p1
        assert math.isclose(result["p_home_win"], 0.6, rel_tol=1e-6)

    def test_weighted_ensemble(self):
        p1 = {"p_home_win": 1.0, "p_draw": 0.0, "p_away_win": 0.0}
        p2 = {"p_home_win": 0.0, "p_draw": 1.0, "p_away_win": 0.0}

        result = ensemble_probabilities([p1, p2], [0.7, 0.3])
        assert math.isclose(result["p_home_win"], 0.7, rel_tol=1e-6)
        assert math.isclose(result["p_draw"], 0.3, rel_tol=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ensemble_probabilities([], [])

    def test_all_none_raises(self):
        with pytest.raises(ValueError):
            ensemble_probabilities([None, None], [0.5, 0.5])


class TestEnsembleWithLambdas:
    def test_averages_lambdas(self):
        p1 = {"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2, "lambda_home": 1.5, "lambda_away": 1.0}
        p2 = {"p_home_win": 0.6, "p_draw": 0.2, "p_away_win": 0.2, "lambda_home": 2.0, "lambda_away": 0.8}

        result = ensemble_with_lambdas([p1, p2], [0.5, 0.5])
        assert math.isclose(result["lambda_home"], 1.75, rel_tol=1e-6)
        assert math.isclose(result["lambda_away"], 0.9, rel_tol=1e-6)

    def test_elo_without_lambdas_excluded_from_lambda_avg(self):
        p_rolling = {"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2, "lambda_home": 1.5, "lambda_away": 1.0}
        p_elo = {"p_home_win": 0.6, "p_draw": 0.2, "p_away_win": 0.2}  # no lambdas

        result = ensemble_with_lambdas([p_rolling, p_elo], [0.5, 0.5])
        # Lambdas should come only from rolling
        assert math.isclose(result["lambda_home"], 1.5, rel_tol=1e-6)
        assert math.isclose(result["lambda_away"], 1.0, rel_tol=1e-6)
        # But probs are averaged
        assert math.isclose(result["p_home_win"], 0.55, rel_tol=1e-2)
