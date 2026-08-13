import unittest

import numpy as np

from football_lab.data.synthetic import (
    create_simulation_snapshot,
    generate_synthetic_league,
)
from football_lab.models import DixonColesModel
from football_lab.simulation.league import current_table, simulate_league


class SimulationTests(unittest.TestCase):
    def test_league_probabilities_are_coherent(self) -> None:
        matches = generate_synthetic_league(n_teams=8, n_seasons=2, seed=11)
        completed, fixtures = create_simulation_snapshot(matches)
        model = DixonColesModel().fit(completed)
        summary, positions = simulate_league(
            model,
            completed,
            fixtures,
            n_simulations=200,
            seed=11,
        )
        self.assertAlmostEqual(float(summary["title_probability"].sum()), 1.0, places=8)
        team_sums = positions.groupby("team")["probability"].sum().to_numpy()
        self.assertTrue(np.allclose(team_sums, 1.0))
        self.assertTrue(summary["expected_position"].between(1, 8).all())

    def test_current_table_points_are_consistent(self) -> None:
        matches = generate_synthetic_league(n_teams=6, n_seasons=1, seed=5)
        table = current_table(matches)
        expected_points = 3 * table["wins"] + table["draws"]
        self.assertTrue((table["points"] == expected_points).all())
        self.assertEqual(table["played"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
