import io
import unittest

import numpy as np

from football_lab.data.football_data_org import parse_matches_payload
from football_lab.data.football_data_uk import build_url, parse_csv, season_code


class DataSourceTests(unittest.TestCase):
    def test_season_and_url_mapping(self) -> None:
        self.assertEqual(season_code("2024-25"), "2425")
        self.assertEqual(season_code("1999/2000"), "9900")
        self.assertEqual(
            build_url("la_liga", "2025-26"),
            "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
        )

    def test_football_data_uk_parser_and_margin_removal(self) -> None:
        csv = io.StringIO(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,AvgH,AvgD,AvgA\n"
            "16/08/2024,Alpha FC,Beta FC,2,1,2.00,3.50,4.00\n"
            "17/08/2024,Gamma FC,Delta FC,0,0,1.80,3.80,5.00\n"
        )
        frame = parse_csv(csv, league="premier_league", season="2024-25")
        self.assertEqual(len(frame), 2)
        probabilities = frame[
            ["market_home_prob", "market_draw_prob", "market_away_prob"]
        ].to_numpy()
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue(frame["status"].eq("FINISHED").all())

    def test_football_data_org_payload_parser(self) -> None:
        payload = {
            "matches": [
                {
                    "id": 11,
                    "utcDate": "2025-08-16T14:00:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"name": "Alpha FC"},
                    "awayTeam": {"name": "Beta FC"},
                    "score": {"fullTime": {"home": 3, "away": 1}},
                },
                {
                    "id": 12,
                    "utcDate": "2025-08-23T14:00:00Z",
                    "status": "TIMED",
                    "homeTeam": {"name": "Beta FC"},
                    "awayTeam": {"name": "Alpha FC"},
                    "score": {"fullTime": {"home": None, "away": None}},
                },
            ]
        }
        frame = parse_matches_payload(
            payload,
            league="premier_league",
            season=2025,
        )
        self.assertEqual(frame["status"].tolist(), ["FINISHED", "SCHEDULED"])
        self.assertEqual(frame.loc[0, "provider_match_id"], 11)


if __name__ == "__main__":
    unittest.main()
