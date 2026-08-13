"""Proper scoring rules and calibration summaries for 1X2 forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ["p_home", "p_draw", "p_away"]


def outcome_index(frame: pd.DataFrame) -> np.ndarray:
    home = frame["home_goals"].to_numpy(dtype=float)
    away = frame["away_goals"].to_numpy(dtype=float)
    return np.where(home > away, 0, np.where(home == away, 1, 2)).astype(int)


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = set(PROBABILITY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions must be finite")
    if (probabilities < 0).any():
        raise ValueError("Predictions cannot be negative")
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.clip(probabilities, 1e-12, 1.0)


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, float]:
    """Evaluate predictions with log loss, Brier score, RPS, and accuracy."""
    probabilities = _probability_matrix(frame)
    actual = outcome_index(frame)
    one_hot = np.eye(3)[actual]
    log_loss = -np.log(probabilities[np.arange(len(actual)), actual]).mean()
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    predicted = probabilities.argmax(axis=1)
    cumulative_prediction = np.cumsum(probabilities, axis=1)[:, :2]
    cumulative_actual = np.cumsum(one_hot, axis=1)[:, :2]
    ranked_probability_score = np.square(
        cumulative_prediction - cumulative_actual
    ).sum(axis=1).mean() / 2
    return {
        "log_loss": float(log_loss),
        "brier_score": float(brier),
        "ranked_probability_score": float(ranked_probability_score),
        "accuracy": float((predicted == actual).mean()),
        "matches": float(len(frame)),
    }


def calibration_table(frame: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Reliability table for the most likely predicted outcome."""
    probabilities = _probability_matrix(frame)
    actual = outcome_index(frame)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == actual
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.digitize(confidence, edges, right=True) - 1, 0, bins - 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        selected = bin_id == index
        if not selected.any():
            continue
        rows.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "mean_confidence": float(confidence[selected].mean()),
                "observed_accuracy": float(correct[selected].mean()),
                "matches": float(selected.sum()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(frame: pd.DataFrame, *, bins: int = 10) -> float:
    table = calibration_table(frame, bins=bins)
    if table.empty:
        return 0.0
    weights = table["matches"] / table["matches"].sum()
    gaps = (table["mean_confidence"] - table["observed_accuracy"]).abs()
    return float((weights * gaps).sum())


def evaluate_market_baseline(frame: pd.DataFrame) -> dict[str, float] | None:
    market = ["market_home_prob", "market_draw_prob", "market_away_prob"]
    if not set(market).issubset(frame.columns):
        return None
    complete = frame[market].notna().all(axis=1)
    if not complete.any():
        return None
    renamed = frame.loc[complete].rename(
        columns=dict(zip(market, PROBABILITY_COLUMNS, strict=True))
    )
    return evaluate_predictions(renamed)
