# src/model_ensemble.py
# Ensemble model that combines multiple prediction systems.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EnsembleConfig:
    # Weights for each model in the ensemble (must sum to 1)
    # rolling = rolling GF/GA + Poisson
    # strength = attack/defence strength model + Poisson
    # elo = Elo rating system
    weight_rolling: float = 0.35
    weight_strength: float = 0.35
    weight_elo: float = 0.30


def ensemble_probabilities(
    predictions: List[Dict[str, float]],
    weights: List[float],
) -> Dict[str, float]:
    """
    Combine 1X2 probability predictions from multiple models using weighted average.

    Each prediction dict must have: p_home_win, p_draw, p_away_win.
    Weights are normalized to sum to 1.
    """
    if not predictions:
        raise ValueError("No predictions to ensemble")

    # Filter out None/missing predictions and adjust weights
    valid = [(p, w) for p, w in zip(predictions, weights) if p is not None]
    if not valid:
        raise ValueError("All predictions are None")

    preds, wts = zip(*valid)
    wts = np.array(wts, dtype=float)
    wts = wts / wts.sum()  # normalize

    p_home = sum(w * p["p_home_win"] for p, w in zip(preds, wts))
    p_draw = sum(w * p["p_draw"] for p, w in zip(preds, wts))
    p_away = sum(w * p["p_away_win"] for p, w in zip(preds, wts))

    # Normalize to ensure they sum to 1 (floating point safety)
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return {
        "p_home_win": float(p_home),
        "p_draw": float(p_draw),
        "p_away_win": float(p_away),
    }


def ensemble_with_lambdas(
    predictions: List[Dict[str, Any]],
    weights: List[float],
) -> Dict[str, Any]:
    """
    Full ensemble: combine probabilities and average lambdas.
    Each prediction dict must have: p_home_win, p_draw, p_away_win, lambda_home, lambda_away.
    Missing lambdas (e.g. from Elo) are excluded from lambda averaging.
    """
    probs = ensemble_probabilities(predictions, weights)

    # Average lambdas from models that provide them
    lambda_preds = [(p, w) for p, w in zip(predictions, weights)
                    if p is not None and "lambda_home" in p]
    if lambda_preds:
        lp, lw = zip(*lambda_preds)
        lw = np.array(lw, dtype=float)
        lw = lw / lw.sum()
        lambda_home = float(sum(w * p["lambda_home"] for p, w in zip(lp, lw)))
        lambda_away = float(sum(w * p["lambda_away"] for p, w in zip(lp, lw)))
    else:
        # Fallback: estimate from probabilities (rough inverse)
        lambda_home = 1.3
        lambda_away = 1.1

    return {
        **probs,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
    }
