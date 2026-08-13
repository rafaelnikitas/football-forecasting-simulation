"""Expanding-window backtesting that never trains on future matches."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pandas as pd

from football_lab.data.schema import completed_matches
from football_lab.evaluation.metrics import (
    evaluate_predictions,
    expected_calibration_error,
)


class ForecastModel(Protocol):
    model_name: str

    def fit(self, matches: pd.DataFrame) -> "ForecastModel": ...

    def predict_frame(self, matches: pd.DataFrame) -> pd.DataFrame: ...


def walk_forward_backtest(
    matches: pd.DataFrame,
    model_factory: Callable[[], ForecastModel],
    *,
    initial_train_matches: int = 100,
    refit_every: int = 40,
) -> pd.DataFrame:
    """Predict consecutive future blocks using only earlier observations."""
    frame = completed_matches(matches).sort_values("date").reset_index(drop=True)
    if initial_train_matches >= len(frame):
        raise ValueError("initial_train_matches must leave at least one test match")
    if initial_train_matches < 20:
        raise ValueError("At least 20 initial training matches are required")
    if refit_every < 1:
        raise ValueError("refit_every must be positive")

    predictions: list[pd.DataFrame] = []
    start = initial_train_matches
    while start < len(frame):
        end = min(start + refit_every, len(frame))
        training = frame.iloc[:start]
        testing = frame.iloc[start:end]
        model = model_factory().fit(training)
        predicted = model.predict_frame(testing)
        predicted["model"] = model.model_name
        predicted["training_matches"] = len(training)
        predicted["training_cutoff"] = training["date"].max()
        predictions.append(predicted)
        start = end
    result = pd.concat(predictions, ignore_index=True)
    if not (result["training_cutoff"] < result["date"]).all():
        raise RuntimeError("Temporal leakage detected in backtest")
    return result


def benchmark_models(
    matches: pd.DataFrame,
    model_factories: dict[str, Callable[[], ForecastModel]],
    *,
    initial_train_matches: int = 100,
    refit_every: int = 40,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run identical walk-forward windows for every model."""
    rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for name, factory in model_factories.items():
        result = walk_forward_backtest(
            matches,
            factory,
            initial_train_matches=initial_train_matches,
            refit_every=refit_every,
        )
        metrics = evaluate_predictions(result)
        metrics["expected_calibration_error"] = expected_calibration_error(result)
        rows.append({"model": name, **metrics})
        predictions[name] = result
    summary = pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)
    return summary, predictions
