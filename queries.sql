-- SQL analysis layer for the Vehicle Quality Analytics dataset.
-- Run against quality_data.db (build it first with `python build_database.py`).
-- Each query below reproduces one of the dashboard's KPIs or sections in
-- pure SQL, demonstrating the same analysis without pandas in the loop.

-- 1. Top-line KPIs (all-time)
SELECT
    SUM(defect_count)                                   AS total_defects,
    SUM(warranty_claims)                                AS total_warranty_claims,
    SUM(customer_complaints)                            AS total_customer_complaints,
    ROUND(AVG(defect_rate), 2)                          AS avg_defect_rate,
    ROUND(SUM(customer_complaints) * 100.0
          / SUM(production_volume), 2)                  AS pp100
FROM quality_data;

-- 2. Monthly defect & warranty trend
SELECT
    date,
    SUM(defect_count)     AS defects,
    SUM(warranty_claims)  AS warranty_claims,
    SUM(customer_complaints) AS customer_complaints
FROM quality_data
GROUP BY date
ORDER BY date;

-- 3. Defects by category, ranked
SELECT
    defect_category,
    SUM(defect_count)            AS total_defects,
    ROUND(AVG(defect_rate), 2)   AS avg_defect_rate
FROM quality_data
GROUP BY defect_category
ORDER BY total_defects DESC;

-- 4. PP100 (complaints per 100 units) by vehicle model — the industry
--    customer-satisfaction metric, broken out per model
SELECT
    vehicle_model,
    ROUND(SUM(customer_complaints) * 100.0
          / SUM(production_volume), 2) AS pp100
FROM quality_data
GROUP BY vehicle_model
ORDER BY pp100 DESC;

-- 5. Root-cause finder: month-over-month defect rate increase by
--    vehicle_model + defect_category, using a window function to pull the
--    prior month's rate for comparison (mirrors the dashboard's Root Cause
--    Analysis section)
WITH monthly AS (
    SELECT
        date,
        vehicle_model,
        defect_category,
        AVG(defect_rate) AS defect_rate
    FROM quality_data
    GROUP BY date, vehicle_model, defect_category
),
with_prior AS (
    SELECT
        *,
        LAG(defect_rate) OVER (
            PARTITION BY vehicle_model, defect_category
            ORDER BY date
        ) AS prior_defect_rate
    FROM monthly
)
SELECT
    date,
    vehicle_model,
    defect_category,
    ROUND(defect_rate, 2)        AS defect_rate,
    ROUND(prior_defect_rate, 2)  AS prior_defect_rate,
    ROUND(
        (defect_rate - prior_defect_rate) * 100.0 / prior_defect_rate, 1
    ) AS pct_change
FROM with_prior
WHERE prior_defect_rate IS NOT NULL
  AND date = (SELECT MAX(date) FROM quality_data)
ORDER BY pct_change DESC
LIMIT 3;

-- 6. Anomaly spotlight: Electric Van / Electrical defect rate by month,
--    flagging months more than 2x the trailing 12-month average
WITH ev_electrical AS (
    SELECT
        date,
        AVG(defect_rate) AS defect_rate
    FROM quality_data
    WHERE vehicle_model = 'Electric Van'
      AND defect_category = 'Electrical'
    GROUP BY date
),
with_baseline AS (
    SELECT
        date,
        defect_rate,
        AVG(defect_rate) OVER (
            ORDER BY date
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS trailing_avg
    FROM ev_electrical
)
SELECT
    date,
    ROUND(defect_rate, 2)     AS defect_rate,
    ROUND(trailing_avg, 2)    AS trailing_12mo_avg,
    CASE WHEN defect_rate > 2 * trailing_avg THEN 1 ELSE 0 END AS is_anomaly
FROM with_baseline
WHERE trailing_avg IS NOT NULL
ORDER BY date;

-- 7. Top 10 worst defect incidents (matches the dashboard's incidents table)
SELECT
    date,
    vehicle_model,
    defect_category,
    plant_line,
    defect_count,
    severity_score,
    defect_rate
FROM quality_data
ORDER BY severity_score DESC, defect_count DESC
LIMIT 10;
