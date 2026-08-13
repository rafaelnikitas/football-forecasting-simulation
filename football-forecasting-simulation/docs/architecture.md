# Architecture and Design Decisions

## System boundaries

The system separates data acquisition, football modeling, evaluation, season
state, and presentation. Provider-specific fields never enter model code
directly.

| Boundary | Responsibility |
|---|---|
| Provider adapter | Download and map external records |
| Canonical schema | Enforce dates, teams, scores, status, and optional odds |
| Forecast model | Estimate score matrices and 1X2 probabilities |
| Backtester | Maintain strict train-before-test chronology |
| Simulator | Apply current-season state and sample remaining fixtures |
| Dashboard | Present probabilities, uncertainty, and diagnostics |

## Canonical match contract

Required fields:

- `date`, `league`, `season`;
- `home_team`, `away_team`;
- `home_goals`, `away_goals`;
- `status` (`FINISHED` or `SCHEDULED`).

Optional market fields are accepted only as a complete normalized triplet.

## Modeling

The independent Poisson and Dixon-Coles models estimate team attack, team
defence, an intercept, and home advantage. Attack and defence parameters are
zero-centered for identifiability. The Dixon-Coles likelihood adjusts four
low-score outcomes and exponentially downweights older matches.

The score matrix is the source of every derived prediction, including 1X2,
expected goals, exact scores, and season simulations. This prevents logically
inconsistent endpoints.

## Evaluation

Backtesting uses expanding windows. A model is fitted at a cutoff, predicts the
next fixed-size block, then advances. A mechanical assertion verifies that each
row's training cutoff is strictly earlier than the predicted match date.

Primary metrics:

- multiclass log loss;
- Brier score;
- ranked probability score;
- expected calibration error.

Accuracy remains a descriptive secondary metric.

## Simulation state

Historical results may span many seasons and are valid model training data.
League points, goals, and positions must include only the season containing the
remaining fixtures. The simulator enforces a single target season and maintains
this separation explicitly.

## Production evolution

A deployed version should add:

- immutable prediction snapshots before kickoff;
- scheduled and idempotent ingestion;
- provider retry/backoff and schema-drift alerts;
- entity-resolution tables rather than display-name equality;
- paired uncertainty intervals around model differences;
- monitoring for calibration, freshness, missing fixtures, and latency.
