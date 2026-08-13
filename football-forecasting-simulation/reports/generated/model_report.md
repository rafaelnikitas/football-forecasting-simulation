# Synthetic Demo Model Report

> This report validates the software pipeline. It is **not** a Premier League
> or La Liga performance claim.

## Backtest design

- Expanding-window evaluation with a strict training-cutoff assertion.
- `170` future matches predicted per model.
- Refit every 50 matches; no future result enters its own training window.
- Primary selection metric: multiclass log loss.

## Model comparison

| model | log_loss | brier_score | ranked_probability_score | accuracy | matches | expected_calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| dixon_coles | 1.0491 | 0.6283 | 0.2274 | 0.4706 | 170.0 | 0.0717 |
| independent_poisson | 1.0532 | 0.629 | 0.2276 | 0.4765 | 170.0 | 0.0533 |
| league_rate_baseline | 1.079 | 0.6541 | 0.2373 | 0.3765 | 170.0 | 0.0005 |

## League simulation snapshot

The latest synthetic season is stopped part-way through and the remaining
fixtures are sampled `2000` times from Dixon-Coles score
matrices.

| team | expected_points | title_probability | top_four_probability | relegation_probability | expected_position |
| --- | --- | --- | --- | --- | --- |
| Harbour Rovers | 45.298 | 0.917 | 1.0 | 0.0 | 1.083 |
| Redwood Town | 38.5825 | 0.083 | 1.0 | 0.0 | 1.9175 |
| Metropolitan FC | 25.581 | 0.0 | 0.722 | 0.022 | 4.0105 |
| Olympic Vale | 22.6575 | 0.0 | 0.3315 | 0.171 | 5.542 |
| Union City | 21.7045 | 0.0 | 0.255 | 0.204 | 5.882 |

## Production decision

**Engineering MVP: pass. Real-data benchmark: pending.**

The pipeline trains, forecasts, checks calibration, simulates a league and
generates reproducible artifacts. Before publishing claims, run the documented
Premier League and La Liga backtests, compare against margin-removed bookmaker
probabilities and report per-league confidence intervals.
