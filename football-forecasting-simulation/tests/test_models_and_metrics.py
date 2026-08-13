import unittest

import numpy as np
import pandas as pd

from football_lab.data.synthetic import generate_synthetic_league
from football_lab.evaluation.metrics import evaluate_predictions
from football_lab.models import DixonColesModel, IndependentPoissonModel


class ModelAndMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches = generate_synthetic_league(n_teams=8, n_seasons=2, seed=7)

    def test_dixon_coles_probability_contract(self) -> None:
        model = DixonColesModel().fit(self.matches)
        prediction = model.predict_match("Northbridge FC", "Riverside Athletic")
        total = prediction["p_home"] + prediction["p_draw"] + prediction["p_away"]
        self.assertAlmostEqual(total, 1.0, places=8)
        self.assertGreater(prediction["expected_home_goals"], 0)
        self.assertTrue(model.fit_summary_.converged)

    def test_independent_poisson_score_matrix_is_normalized(self) -> None:
        model = IndependentPoissonModel().fit(self.matches)
        matrix = model.score_matrix("Northbridge FC", "Riverside Athletic")
        self.assertEqual(matrix.shape, (9, 9))
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=8)
        self.assertTrue((matrix >= 0).all())

    def test_proper_metrics_reward_confident_correct_predictions(self) -> None:
        frame = pd.DataFrame(
            {
                "home_goals": [2, 1, 0],
                "away_goals": [0, 1, 3],
                "p_home": [0.9, 0.05, 0.03],
                "p_draw": [0.05, 0.9, 0.02],
                "p_away": [0.05, 0.05, 0.95],
            }
        )
        good = evaluate_predictions(frame)
        uniform = frame.assign(p_home=1 / 3, p_draw=1 / 3, p_away=1 / 3)
        bad = evaluate_predictions(uniform)
        self.assertLess(good["log_loss"], bad["log_loss"])
        self.assertLess(good["brier_score"], bad["brier_score"])
        self.assertTrue(np.isfinite(good["ranked_probability_score"]))


if __name__ == "__main__":
    unittest.main()
