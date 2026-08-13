"""Reproducible end-to-end demo and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from football_lab.data.synthetic import (
    create_simulation_snapshot,
    generate_synthetic_league,
)
from football_lab.evaluation.backtest import benchmark_models
from football_lab.evaluation.metrics import calibration_table
from football_lab.evaluation.metrics import evaluate_market_baseline
from football_lab.models import (
    DixonColesModel,
    IndependentPoissonModel,
    LeagueRateBaseline,
)
from football_lab.simulation.league import simulate_league


def run_demo(
    output_dir: str | Path = "reports/generated",
    *,
    n_simulations: int = 2_000,
    seed: int = 42,
) -> dict[str, object]:
    """Run the full project on deterministic, explicitly synthetic data."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matches = generate_synthetic_league(seed=seed)
    factories = {
        "league_rate_baseline": lambda: LeagueRateBaseline(),
        "independent_poisson": lambda: IndependentPoissonModel(),
        "dixon_coles": lambda: DixonColesModel(),
    }
    metrics, predictions = benchmark_models(
        matches,
        factories,
        initial_train_matches=100,
        refit_every=50,
    )
    completed, fixtures = create_simulation_snapshot(matches)
    final_model = DixonColesModel().fit(completed)
    simulation, positions = simulate_league(
        final_model,
        completed,
        fixtures,
        n_simulations=n_simulations,
        seed=seed,
    )
    calibration = calibration_table(predictions["dixon_coles"])
    strengths = final_model.strengths()

    metrics.to_csv(destination / "benchmark_metrics.csv", index=False)
    calibration.to_csv(destination / "calibration.csv", index=False)
    simulation.to_csv(destination / "simulation_summary.csv", index=False)
    positions.to_csv(destination / "position_probabilities.csv", index=False)
    strengths.to_csv(destination / "team_strengths.csv", index=False)

    best = metrics.iloc[0]
    summary: dict[str, object] = {
        "data_scope": "deterministic synthetic demo",
        "models": list(factories),
        "backtest_predictions": int(best["matches"]),
        "best_model_by_log_loss": str(best["model"]),
        "best_log_loss": float(best["log_loss"]),
        "best_brier_score": float(best["brier_score"]),
        "simulations": n_simulations,
        "completed_matches": len(completed),
        "remaining_fixtures": len(fixtures),
        "model_converged": bool(final_model.fit_summary_.converged),
    }
    (destination / "demo_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_decision_report(destination / "model_report.md", metrics, simulation, summary)
    return summary


def run_real_backtest(
    matches: pd.DataFrame,
    output_dir: str | Path,
    *,
    initial_train_matches: int = 760,
    refit_every: int = 100,
) -> pd.DataFrame:
    """Benchmark each league independently and include market odds when available."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    for league, league_frame in matches.groupby("league", sort=True):
        league_frame = league_frame.sort_values("date").reset_index(drop=True)
        if len(league_frame) <= initial_train_matches:
            raise ValueError(
                f"{league} has {len(league_frame)} matches; "
                f"initial_train_matches={initial_train_matches} leaves no test set"
            )
        metrics, predictions = benchmark_models(
            league_frame,
            {
                "league_rate_baseline": lambda: LeagueRateBaseline(),
                "independent_poisson": lambda: IndependentPoissonModel(),
                "dixon_coles": lambda: DixonColesModel(),
            },
            initial_train_matches=initial_train_matches,
            refit_every=refit_every,
        )
        market = evaluate_market_baseline(predictions["dixon_coles"])
        if market is not None:
            market_row = pd.DataFrame([{"model": "market_consensus", **market}])
            metrics = pd.concat([metrics, market_row], ignore_index=True)
        metrics.insert(0, "league", league)
        metrics = metrics.sort_values("log_loss", ignore_index=True)
        metrics.to_csv(destination / f"{league}_benchmark_metrics.csv", index=False)
        predictions["dixon_coles"].to_csv(
            destination / f"{league}_dixon_coles_predictions.csv",
            index=False,
        )
        summaries.append(metrics)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(destination / "all_leagues_benchmark_metrics.csv", index=False)
    return combined


def _write_decision_report(
    path: Path,
    metrics: pd.DataFrame,
    simulation: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    metrics_table = _markdown_table(metrics.round(4))
    simulation_table = _markdown_table(simulation.head(5).round(4))
    path.write_text(
        f"""# Synthetic Demo Model Report

> This report validates the software pipeline. It is **not** a Premier League
> or La Liga performance claim.

## Backtest design

- Expanding-window evaluation with a strict training-cutoff assertion.
- `{summary['backtest_predictions']}` future matches predicted per model.
- Refit every 50 matches; no future result enters its own training window.
- Primary selection metric: multiclass log loss.

## Model comparison

{metrics_table}

## League simulation snapshot

The latest synthetic season is stopped part-way through and the remaining
fixtures are sampled `{summary['simulations']}` times from Dixon-Coles score
matrices.

{simulation_table}

## Production decision

**Engineering MVP: pass. Real-data benchmark: pending.**

The pipeline trains, forecasts, checks calibration, simulates a league and
generates reproducible artifacts. Before publishing claims, run the documented
Premier League and La Liga backtests, compare against margin-removed bookmaker
probabilities and report per-league confidence intervals.
""",
        encoding="utf-8",
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
