SELECT
    record_id,
    window_id,
    window_name,
    prediction_proba,
    prediction_label,
    actual_label,
    CASE WHEN prediction_label = actual_label
         THEN 1 ELSE 0 END            AS correct,
    timestamp
FROM predictions
WHERE prediction_proba IS NOT NULL
