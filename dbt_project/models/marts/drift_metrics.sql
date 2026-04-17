SELECT
    window_id,
    window_name,
    COUNT(*)                                      AS total_predictions,
    ROUND(AVG(correct) * 100, 2)                 AS accuracy_pct,
    ROUND(AVG(prediction_proba), 4)              AS avg_confidence,
    MIN(timestamp)                                AS window_start,
    MAX(timestamp)                                AS window_end
FROM {{ ref('stg_predictions') }}
GROUP BY window_id, window_name
ORDER BY window_id
