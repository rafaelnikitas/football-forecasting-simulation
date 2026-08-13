"""Deterministic synthetic league used for tests and the zero-setup demo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from football_lab.data.schema import validate_matches

TEAM_NAMES = [
    "Northbridge FC",
    "Riverside Athletic",
    "Union City",
    "Harbour Rovers",
    "Royal Borough",
    "Olympic Vale",
    "Sporting Albion",
    "Metropolitan FC",
    "Redwood Town",
    "Coastal United",
    "Capital Wanderers",
    "Lakeside 1908",
]


def _double_round_robin(teams: list[str], rng: np.random.Generator) -> list[tuple[str, str]]:
    fixtures = [(home, away) for home in teams for away in teams if home != away]
    rng.shuffle(fixtures)
    return fixtures


def generate_synthetic_league(
    *,
    n_teams: int = 10,
    n_seasons: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate complete seasons from latent attack and defence strengths."""
    if not 6 <= n_teams <= len(TEAM_NAMES):
        raise ValueError(f"n_teams must be between 6 and {len(TEAM_NAMES)}")
    rng = np.random.default_rng(seed)
    teams = TEAM_NAMES[:n_teams]
    attack = rng.normal(0.0, 0.28, n_teams)
    defence = rng.normal(0.0, 0.22, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()
    records: list[dict[str, object]] = []

    for season_index in range(n_seasons):
        start_year = 2022 + season_index
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
        start_date = datetime(start_year, 8, 6, 14, tzinfo=UTC)
        fixtures = _double_round_robin(teams, rng)
        seasonal_attack = attack + rng.normal(0.0, 0.035, n_teams)
        seasonal_defence = defence + rng.normal(0.0, 0.035, n_teams)

        for match_index, (home_team, away_team) in enumerate(fixtures):
            home_id = teams.index(home_team)
            away_id = teams.index(away_team)
            home_lambda = np.exp(
                0.24 + seasonal_attack[home_id] - seasonal_defence[away_id]
            )
            away_lambda = np.exp(
                0.02 + seasonal_attack[away_id] - seasonal_defence[home_id]
            )
            records.append(
                {
                    "date": start_date + timedelta(days=match_index * 2),
                    "league": "synthetic_premier_division",
                    "season": season,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": int(rng.poisson(home_lambda)),
                    "away_goals": int(rng.poisson(away_lambda)),
                    "status": "FINISHED",
                }
            )
        attack += rng.normal(0.0, 0.025, n_teams)
        defence += rng.normal(0.0, 0.025, n_teams)
        attack -= attack.mean()
        defence -= defence.mean()

    return validate_matches(pd.DataFrame.from_records(records))


def create_simulation_snapshot(
    matches: pd.DataFrame,
    *,
    completed_fraction: float = 0.65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hide the final part of the latest season to create future fixtures."""
    validated = validate_matches(matches)
    latest_season = validated["season"].iloc[-1]
    prior = validated.loc[validated["season"] != latest_season]
    current = validated.loc[validated["season"] == latest_season].copy()
    cutoff = max(1, int(len(current) * completed_fraction))
    completed = pd.concat([prior, current.iloc[:cutoff]], ignore_index=True)
    fixtures = current.iloc[cutoff:].copy()
    fixtures[["home_goals", "away_goals"]] = np.nan
    fixtures["status"] = "SCHEDULED"
    return validate_matches(completed), validate_matches(fixtures)
