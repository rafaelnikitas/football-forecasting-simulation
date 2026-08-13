# Model Card: Dixon-Coles Match Forecast

## Intended use

Estimate probabilistic football scorelines and derived match/season outcomes for
analysis, education, and portfolio demonstration.

## Not intended for

- guaranteed outcome claims;
- financial or betting advice;
- player health, employment, or disciplinary decisions;
- evaluation that mixes future observations into training.

## Inputs

Completed match dates, teams, and full-time scores. Optional market odds are
used only as an evaluation benchmark, not as training features in the MVP.

## Outputs

- home/draw/away probabilities;
- expected home and away goals;
- normalized exact-score matrix;
- simulated finish-position probabilities.

## Evaluation

Use league-specific expanding-window backtests and report log loss, Brier score,
ranked probability score, calibration, and market-baseline comparisons. Report
the date range, number of forecasts, refit cadence, and uncertainty intervals.

## Limitations

The current model does not directly represent injuries, lineups, managers,
travel, rest, promoted-team uncertainty, red-card states, or transfer activity.
Team strengths are assumed locally stable and decay smoothly through time.

## Monitoring

Track data freshness, missing fixtures, score/probability validity, rolling log
loss, calibration by league, and drift in expected goal rates.

## Current evidence

The checked-in metrics are from deterministic synthetic data and verify only
the software pipeline. Real EPL and La Liga evidence is explicitly pending.
