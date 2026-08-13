import unittest

from football_lab.data.synthetic import generate_synthetic_league
from football_lab.evaluation.backtest import walk_forward_backtest
from football_lab.evaluation.metrics import evaluate_predictions
from football_lab.models import LeagueRateBaseline


class BacktestTests(unittest.TestCase):
    def test_expanding_window_has_strict_temporal_cutoff(self) -> None:
        matches = generate_synthetic_league(n_teams=8, n_seasons=2, seed=3)
        predictions = walk_forward_backtest(
            matches,
            LeagueRateBaseline,
            initial_train_matches=60,
            refit_every=20,
        )
        self.assertTrue((predictions["training_cutoff"] < predictions["date"]).all())
        self.assertEqual(len(predictions), len(matches) - 60)
        self.assertGreater(evaluate_predictions(predictions)["log_loss"], 0)


if __name__ == "__main__":
    unittest.main()
