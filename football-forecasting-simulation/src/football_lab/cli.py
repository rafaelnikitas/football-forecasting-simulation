"""Command-line entry points for demos and provider ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_lab.data.football_data_org import fetch_matches
from football_lab.data.football_data_uk import fetch_many
from football_lab.data.schema import validate_matches
from football_lab.models import DixonColesModel
from football_lab.pipeline import run_demo, run_real_backtest
from football_lab.simulation.league import simulate_league


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Football Forecasting & Simulation Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the deterministic synthetic workflow")
    demo.add_argument("--output-dir", default="reports/generated")
    demo.add_argument("--simulations", type=int, default=2_000)
    demo.add_argument("--seed", type=int, default=42)

    historical = subparsers.add_parser(
        "fetch-historical",
        help="Fetch Football-Data.co.uk seasons without an API key",
    )
    historical.add_argument(
        "--leagues",
        nargs="+",
        choices=["premier_league", "la_liga"],
        required=True,
    )
    historical.add_argument("--seasons", nargs="+", required=True)
    historical.add_argument("--output-dir", default="data/raw/football_data_uk")
    historical.add_argument("--canonical-output", default="data/matches.csv")

    fixtures = subparsers.add_parser(
        "fetch-fixtures",
        help="Fetch football-data.org v4 matches using FOOTBALL_DATA_API_TOKEN",
    )
    fixtures.add_argument(
        "--league",
        choices=["premier_league", "la_liga"],
        required=True,
    )
    fixtures.add_argument("--season", type=int, required=True)
    fixtures.add_argument("--output", default="data/fixtures.csv")

    backtest = subparsers.add_parser(
        "backtest",
        help="Run per-league walk-forward benchmarks on a canonical CSV",
    )
    backtest.add_argument("--input", required=True)
    backtest.add_argument("--output-dir", default="reports/real_data")
    backtest.add_argument("--initial-train-matches", type=int, default=760)
    backtest.add_argument("--refit-every", type=int, default=100)

    simulation = subparsers.add_parser(
        "simulate",
        help="Fit on history and simulate one season's scheduled fixtures",
    )
    simulation.add_argument("--history", required=True)
    simulation.add_argument("--fixtures", required=True)
    simulation.add_argument("--league", required=True)
    simulation.add_argument("--season", required=True)
    simulation.add_argument("--simulations", type=int, default=10_000)
    simulation.add_argument("--output", default="reports/real_data/league_simulation.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "demo":
        summary = run_demo(args.output_dir, n_simulations=args.simulations, seed=args.seed)
        print(f"Synthetic demo complete: {summary}")
        return 0
    if args.command == "fetch-historical":
        frame = fetch_many(args.leagues, args.seasons, output_dir=args.output_dir)
        destination = Path(args.canonical_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Saved {len(frame)} canonical matches to {destination}")
        return 0
    if args.command == "fetch-fixtures":
        frame = fetch_matches(args.league, season=args.season)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Saved {len(frame)} API matches to {destination}")
        return 0
    if args.command == "backtest":
        frame = validate_matches(pd.read_csv(args.input))
        summary = run_real_backtest(
            frame,
            args.output_dir,
            initial_train_matches=args.initial_train_matches,
            refit_every=args.refit_every,
        )
        print(summary.to_string(index=False))
        return 0

    history = validate_matches(pd.read_csv(args.history))
    fixtures_frame = validate_matches(pd.read_csv(args.fixtures))
    historical_training = history.loc[history["league"].eq(args.league)]
    current_completed = fixtures_frame.loc[
        fixtures_frame["league"].eq(args.league)
        & fixtures_frame["season"].astype(str).eq(str(args.season))
        & fixtures_frame["status"].eq("FINISHED")
    ]
    training = validate_matches(
        pd.concat([historical_training, current_completed], ignore_index=True)
        .drop_duplicates(["date", "home_team", "away_team"], keep="last")
    )
    target = fixtures_frame.loc[
        fixtures_frame["league"].eq(args.league)
        & fixtures_frame["season"].astype(str).eq(str(args.season))
        & fixtures_frame["status"].eq("SCHEDULED")
    ].reset_index(drop=True)
    model = DixonColesModel().fit(training)
    summary, _ = simulate_league(
        model,
        training,
        target,
        n_simulations=args.simulations,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    print(f"Saved {args.simulations} simulation summary to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
