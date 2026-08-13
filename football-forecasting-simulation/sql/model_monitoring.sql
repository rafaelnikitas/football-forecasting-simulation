-- Warehouse-agnostic monitoring contract for immutable pre-match predictions.
-- Adapt date functions and percentile syntax to the target database.

WITH scored AS (
    SELECT
        prediction_id,
        league,
        model_version,
        predicted_at,
        kickoff_at,
        CASE
            WHEN home_goals > away_goals THEN 'home'
            WHEN home_goals = away_goals THEN 'draw'
            ELSE 'away'
        END AS actual_outcome,
        p_home,
        p_draw,
        p_away
    FROM football_predictions
    WHERE predicted_at < kickoff_at
      AND home_goals IS NOT NULL
      AND away_goals IS NOT NULL
),
losses AS (
    SELECT
        *,
        -LN(
            CASE actual_outcome
                WHEN 'home' THEN NULLIF(p_home, 0)
                WHEN 'draw' THEN NULLIF(p_draw, 0)
                ELSE NULLIF(p_away, 0)
            END
        ) AS log_loss
    FROM scored
)
SELECT
    league,
    model_version,
    DATE_TRUNC('month', kickoff_at) AS kickoff_month,
    COUNT(*) AS scored_predictions,
    AVG(log_loss) AS mean_log_loss,
    AVG(p_home + p_draw + p_away) AS mean_probability_sum,
    MIN(predicted_at) AS first_prediction_at,
    MAX(predicted_at) AS last_prediction_at
FROM losses
GROUP BY 1, 2, 3
ORDER BY 3 DESC, 1, 2;
