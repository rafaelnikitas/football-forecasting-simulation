import unittest

import pandas as pd

from football_lab.data.synthetic import generate_synthetic_league


class SyntheticDataTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_synthetic_league(seed=19)
        second = generate_synthetic_league(seed=19)
        pd.testing.assert_frame_equal(first, second)

    def test_double_round_robin_match_count(self) -> None:
        matches = generate_synthetic_league(n_teams=10, n_seasons=3)
        self.assertEqual(len(matches), 10 * 9 * 3)
        self.assertTrue(matches["status"].eq("FINISHED").all())


if __name__ == "__main__":
    unittest.main()
