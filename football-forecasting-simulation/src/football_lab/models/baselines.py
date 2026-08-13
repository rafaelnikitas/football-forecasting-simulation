"""Transparent forecasting baselines."""

from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd

from football_lab.data.schema import completed_matches, validate_matches


def poisson_score_matrix(home_lambda: float, away_lambda: float, max_goals: int) -> np.ndarray:
    home = np.array(
        [
            np.exp(-home_lambda) * home_lambda**goals / factorial(goals)
            for goals in range(max_goals + 1)
        ]
    )
    away = np.array(
        [
            np.exp(-away_lambda) * away_lambda**goals / factorial(goals)
            for goals in range(max_goals + 1)
        ]
    )
    matrix = np.outer(home, away)
    return matrix / matrix.sum()


def outcome_probabilities(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.tril(matrix, -1).sum(),
            np.trace(matrix),
            np.triu(matrix, 1).sum(),
        ]
    )


class LeagueRateBaseline:
    """Predict every match from the league-wide home and away scoring rates."""

    model_name = "league_rate_baseline"

    def __init__(self, *, max_goals: int = 8) -> None:
        self.max_goals = max_goals
        self.home_lambda_: float | None = None
        self.away_lambda_: float | None = None

    def fit(self, matches: pd.DataFrame) -> "LeagueRateBaseline":
        frame = completed_matches(matches)
        self.home_lambda_ = float(frame["home_goals"].mean())
        self.away_lambda_ = float(frame["away_goals"].mean())
        return self

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        del home_team, away_team
        if self.home_lambda_ is None or self.away_lambda_ is None:
            raise RuntimeError("Fit the model before predicting")
        return poisson_score_matrix(self.home_lambda_, self.away_lambda_, self.max_goals)

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        matrix = self.score_matrix(home_team, away_team)
        p_home, p_draw, p_away = outcome_probabilities(matrix)
        return {
            "p_home": float(p_home),
            "p_draw": float(p_draw),
            "p_away": float(p_away),
            "expected_home_goals": float(self.home_lambda_),
            "expected_away_goals": float(self.away_lambda_),
        }

    def predict_frame(self, matches: pd.DataFrame) -> pd.DataFrame:
        frame = validate_matches(matches)
        predictions = [
            self.predict_match(row.home_team, row.away_team)
            for row in frame.itertuples(index=False)
        ]
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
