"""League-table calculation and Monte Carlo finish probabilities."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from football_lab.data.schema import completed_matches, validate_matches


class ScoreModel(Protocol):
    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray: ...


def current_table(matches: pd.DataFrame, *, teams: list[str] | None = None) -> pd.DataFrame:
    """Calculate a points table using points, goal difference, then goals for."""
    frame = completed_matches(matches)
    all_teams = teams or sorted(set(frame["home_team"]).union(frame["away_team"]))
    stats = {
        team: {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "points": 0}
        for team in all_teams
    }
    for row in frame.itertuples(index=False):
        home_goals, away_goals = int(row.home_goals), int(row.away_goals)
        home, away = stats[row.home_team], stats[row.away_team]
        home["played"] += 1
        away["played"] += 1
        home["gf"] += home_goals
        home["ga"] += away_goals
        away["gf"] += away_goals
        away["ga"] += home_goals
        if home_goals > away_goals:
            home["wins"] += 1
            away["losses"] += 1
            home["points"] += 3
        elif home_goals < away_goals:
            away["wins"] += 1
            home["losses"] += 1
            away["points"] += 3
        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1
    result = pd.DataFrame.from_dict(stats, orient="index").rename_axis("team").reset_index()
    result["goal_difference"] = result["gf"] - result["ga"]
    return result.sort_values(
        ["points", "goal_difference", "gf", "team"],
        ascending=[False, False, False, True],
        ignore_index=True,
    ).assign(position=lambda value: np.arange(1, len(value) + 1))


def simulate_league(
    model: ScoreModel,
    completed: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    n_simulations: int = 2_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate remaining scores and return team and position probabilities."""
    if n_simulations < 100:
        raise ValueError("Use at least 100 simulations")
    completed_frame = completed_matches(completed)
    fixture_frame = validate_matches(fixtures)
    if not fixture_frame["status"].eq("SCHEDULED").all():
        raise ValueError("Every simulated fixture must be SCHEDULED")
    fixture_seasons = fixture_frame["season"].unique()
    if len(fixture_seasons) != 1:
        raise ValueError("A league simulation must target exactly one season")
    target_season = fixture_seasons[0]
    standings_matches = completed_frame.loc[
        completed_frame["season"].eq(target_season)
    ].reset_index(drop=True)
    if standings_matches.empty:
        raise ValueError("No completed matches exist for the simulated season")
    teams = sorted(
        set(standings_matches["home_team"])
        .union(standings_matches["away_team"])
        .union(fixture_frame["home_team"])
        .union(fixture_frame["away_team"])
    )
    team_to_id = {team: index for index, team in enumerate(teams)}
    base = current_table(standings_matches, teams=teams).set_index("team").loc[teams]
    base_points = base["points"].to_numpy(dtype=int)
    base_gf = base["gf"].to_numpy(dtype=int)
    base_ga = base["ga"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    position_counts = np.zeros((len(teams), len(teams)), dtype=int)
    point_totals = np.zeros(len(teams), dtype=float)

    sampled_scores: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for row in fixture_frame.itertuples(index=False):
        matrix = model.score_matrix(row.home_team, row.away_team)
        choices = rng.choice(matrix.size, size=n_simulations, p=matrix.ravel())
        home_goals, away_goals = np.unravel_index(choices, matrix.shape)
        sampled_scores.append(
            (
                team_to_id[row.home_team],
                team_to_id[row.away_team],
                home_goals,
                away_goals,
            )
        )

    for simulation in range(n_simulations):
        points = base_points.copy()
        gf = base_gf.copy()
        ga = base_ga.copy()
        for home_id, away_id, home_scores, away_scores in sampled_scores:
            home_goals = int(home_scores[simulation])
            away_goals = int(away_scores[simulation])
            gf[home_id] += home_goals
            ga[home_id] += away_goals
            gf[away_id] += away_goals
            ga[away_id] += home_goals
            if home_goals > away_goals:
                points[home_id] += 3
            elif home_goals < away_goals:
                points[away_id] += 3
            else:
                points[home_id] += 1
                points[away_id] += 1
        goal_difference = gf - ga
        order = np.lexsort((-gf, -goal_difference, -points))
        for position, team_id in enumerate(order):
            position_counts[team_id, position] += 1
        point_totals += points

    probabilities = position_counts / n_simulations
    top_n = min(4, len(teams))
    relegation_n = min(3, len(teams))
    summary = pd.DataFrame(
        {
            "team": teams,
            "expected_points": point_totals / n_simulations,
            "title_probability": probabilities[:, 0],
            "top_four_probability": probabilities[:, :top_n].sum(axis=1),
            "relegation_probability": probabilities[:, -relegation_n:].sum(axis=1),
            "expected_position": (
                probabilities * np.arange(1, len(teams) + 1)
            ).sum(axis=1),
        }
    ).sort_values(
        ["title_probability", "expected_points"],
        ascending=[False, False],
        ignore_index=True,
    )
    position_rows = [
        {"team": team, "position": position + 1, "probability": probabilities[team_id, position]}
        for team_id, team in enumerate(teams)
        for position in range(len(teams))
    ]
    return summary, pd.DataFrame(position_rows)
