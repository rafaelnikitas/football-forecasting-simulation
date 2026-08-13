"""Independent Poisson and time-decayed Dixon-Coles score models."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from football_lab.data.schema import completed_matches, validate_matches
from football_lab.models.baselines import outcome_probabilities


@dataclass(frozen=True)
class ModelFitSummary:
    converged: bool
    objective: float
    iterations: int
    matches: int


class DixonColesModel:
    """Maximum-likelihood score model with low-score correction and time decay."""

    model_name = "dixon_coles"

    def __init__(
        self,
        *,
        decay: float = 0.0018,
        max_goals: int = 8,
        l2: float = 0.01,
        use_correction: bool = True,
    ) -> None:
        self.decay = decay
        self.max_goals = max_goals
        self.l2 = l2
        self.use_correction = use_correction
        self.teams_: list[str] = []
        self.team_to_id_: dict[str, int] = {}
        self.params_: np.ndarray | None = None
        self.fit_summary_: ModelFitSummary | None = None
        self.fitted_at_: pd.Timestamp | None = None

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        n_teams = len(self.teams_)
        attack = params[:n_teams].copy()
        defence = params[n_teams : 2 * n_teams].copy()
        attack -= attack.mean()
        defence -= defence.mean()
        intercept = float(params[-3])
        home_advantage = float(params[-2])
        rho = 0.20 * np.tanh(float(params[-1])) if self.use_correction else 0.0
        return attack, defence, intercept, home_advantage, rho

    @staticmethod
    def _tau(
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        home_lambda: np.ndarray,
        away_lambda: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        tau = np.ones_like(home_lambda)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        tau[mask_00] = 1.0 - home_lambda[mask_00] * away_lambda[mask_00] * rho
        tau[mask_01] = 1.0 + home_lambda[mask_01] * rho
        tau[mask_10] = 1.0 + away_lambda[mask_10] * rho
        tau[mask_11] = 1.0 - rho
        return tau

    def fit(self, matches: pd.DataFrame, *, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        frame = completed_matches(matches)
        self.teams_ = sorted(set(frame["home_team"]).union(frame["away_team"]))
        self.team_to_id_ = {team: index for index, team in enumerate(self.teams_)}
        home_ids = frame["home_team"].map(self.team_to_id_).to_numpy(dtype=int)
        away_ids = frame["away_team"].map(self.team_to_id_).to_numpy(dtype=int)
        home_goals = frame["home_goals"].to_numpy(dtype=float)
        away_goals = frame["away_goals"].to_numpy(dtype=float)
        fitted_at = pd.Timestamp(as_of) if as_of is not None else frame["date"].max()
        if fitted_at.tzinfo is None:
            fitted_at = fitted_at.tz_localize("UTC")
        ages = (fitted_at - frame["date"]).dt.total_seconds().clip(lower=0) / 86_400
        weights = np.exp(-self.decay * ages.to_numpy(dtype=float))
        n_teams = len(self.teams_)

        mean_home = max(float(home_goals.mean()), 0.2)
        mean_away = max(float(away_goals.mean()), 0.2)
        initial = np.zeros(2 * n_teams + 3)
        initial[-3] = np.log(mean_away)
        initial[-2] = np.log(mean_home / mean_away)

        def objective(params: np.ndarray) -> float:
            attack, defence, intercept, home_advantage, rho = self._unpack(params)
            home_lambda = np.exp(
                np.clip(
                    intercept + home_advantage + attack[home_ids] - defence[away_ids],
                    -3.0,
                    3.0,
                )
            )
            away_lambda = np.exp(
                np.clip(intercept + attack[away_ids] - defence[home_ids], -3.0, 3.0)
            )
            tau = self._tau(home_goals, away_goals, home_lambda, away_lambda, rho)
            if np.any(tau <= 0):
                return 1e9
            log_probability = (
                -home_lambda
                + home_goals * np.log(home_lambda)
                - gammaln(home_goals + 1)
                - away_lambda
                + away_goals * np.log(away_lambda)
                - gammaln(away_goals + 1)
                + np.log(tau)
            )
            regularization = self.l2 * (
                np.square(attack).sum() + np.square(defence).sum() + params[-1] ** 2
            )
            return float(-(weights * log_probability).sum() + regularization)

        bounds = [(-2.0, 2.0)] * (2 * n_teams) + [(-2.0, 2.0), (-1.0, 1.0), (-2.5, 2.5)]
        if not self.use_correction:
            bounds[-1] = (0.0, 0.0)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        self.params_ = np.asarray(result.x, dtype=float)
        self.fitted_at_ = fitted_at
        self.fit_summary_ = ModelFitSummary(
            converged=bool(result.success),
            objective=float(result.fun),
            iterations=int(result.nit),
            matches=len(frame),
        )
        return self

    def _rates(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        if self.params_ is None:
            raise RuntimeError("Fit the model before predicting")
        attack, defence, intercept, home_advantage, rho = self._unpack(self.params_)
        home_id = self.team_to_id_.get(home_team)
        away_id = self.team_to_id_.get(away_team)
        home_attack = attack[home_id] if home_id is not None else 0.0
        home_defence = defence[home_id] if home_id is not None else 0.0
        away_attack = attack[away_id] if away_id is not None else 0.0
        away_defence = defence[away_id] if away_id is not None else 0.0
        home_lambda = np.exp(intercept + home_advantage + home_attack - away_defence)
        away_lambda = np.exp(intercept + away_attack - home_defence)
        return float(home_lambda), float(away_lambda), rho

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        home_lambda, away_lambda, rho = self._rates(home_team, away_team)
        home = np.array(
            [
                np.exp(-home_lambda) * home_lambda**goals / factorial(goals)
                for goals in range(self.max_goals + 1)
            ]
        )
        away = np.array(
            [
                np.exp(-away_lambda) * away_lambda**goals / factorial(goals)
                for goals in range(self.max_goals + 1)
            ]
        )
        matrix = np.outer(home, away)
        if self.use_correction:
            matrix[0, 0] *= 1.0 - home_lambda * away_lambda * rho
            matrix[0, 1] *= 1.0 + home_lambda * rho
            matrix[1, 0] *= 1.0 + away_lambda * rho
            matrix[1, 1] *= 1.0 - rho
        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float]:
        matrix = self.score_matrix(home_team, away_team)
        p_home, p_draw, p_away = outcome_probabilities(matrix)
        home_lambda, away_lambda, _ = self._rates(home_team, away_team)
        return {
            "p_home": float(p_home),
            "p_draw": float(p_draw),
            "p_away": float(p_away),
            "expected_home_goals": home_lambda,
            "expected_away_goals": away_lambda,
        }

    def predict_frame(self, matches: pd.DataFrame) -> pd.DataFrame:
        frame = validate_matches(matches)
        predictions = [
            self.predict_match(row.home_team, row.away_team)
            for row in frame.itertuples(index=False)
        ]
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)

    def strengths(self) -> pd.DataFrame:
        if self.params_ is None:
            raise RuntimeError("Fit the model before inspecting strengths")
        attack, defence, _, _, _ = self._unpack(self.params_)
        return pd.DataFrame(
            {
                "team": self.teams_,
                "attack_strength": attack,
                "defence_resistance": defence,
                "overall_strength": attack + defence,
            }
        ).sort_values("overall_strength", ascending=False, ignore_index=True)


class IndependentPoissonModel(DixonColesModel):
    """Same attack/defence structure without the Dixon-Coles correction."""

    model_name = "independent_poisson"

    def __init__(self, *, decay: float = 0.0018, max_goals: int = 8, l2: float = 0.01) -> None:
        super().__init__(
            decay=decay,
            max_goals=max_goals,
            l2=l2,
            use_correction=False,
        )
