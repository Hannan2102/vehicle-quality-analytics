"""
Generates a synthetic but realistic commercial vehicle manufacturing quality
dataset (quality_data.csv) covering 42 months (Jan 2023 - Jun 2026).

Safe to re-run any number of times: uses a fixed random seed, so it always
produces the same deterministic output and simply overwrites the CSV.
"""

import numpy as np
import pandas as pd

SEED = 42
OUTPUT_PATH = "quality_data.csv"

VEHICLE_MODELS = ["Class 8 Truck", "Electric Van", "Commercial Bus", "Heavy Hauler"]
DEFECT_CATEGORIES = [
    "Engine",
    "Transmission",
    "Brakes",
    "Electrical",
    "Body/Frame",
    "HVAC",
    "Suspension",
]
PLANT_LINES = ["Line A", "Line B", "Line C"]

# Baseline defect rate (%) per category - Engine and Transmission run highest,
# reflecting the complexity of those subsystems.
BASE_DEFECT_RATE = {
    "Engine": 3.2,
    "Transmission": 2.9,
    "Brakes": 1.8,
    "Electrical": 2.1,
    "Body/Frame": 1.4,
    "HVAC": 1.1,
    "Suspension": 1.6,
}

# Mild per-model multiplier so vehicle models aren't perfectly uniform.
MODEL_MULTIPLIER = {
    "Class 8 Truck": 1.05,
    "Electric Van": 0.95,
    "Commercial Bus": 1.10,
    "Heavy Hauler": 1.15,
}

# Customer complaints scale off the defect rate (not a fixed fraction of
# defect_count) so the relationship still tracks anomalies/improvement
# trends. The multiplier is chosen so the three customer-facing metrics
# form a clean, defensible funnel: defects_found >= customer_complaints >=
# warranty_claims (not every defect is noticed by the customer; not every
# complaint escalates to a formal claim) - rather than complaints
# outnumbering the defects they stem from, which would need an awkward
# "one defect, many reports" justification to defend.
COMPLAINT_RATE_MULTIPLIER = 0.9

ANOMALY_MODEL = "Electric Van"
ANOMALY_CATEGORY = "Electrical"
ANOMALY_START = pd.Timestamp("2024-05-01")
ANOMALY_END = pd.Timestamp("2024-08-01")

IMPROVEMENT_START = pd.Timestamp("2025-01-01")
IMPROVEMENT_TOTAL_REDUCTION = 0.15  # ~15% defect rate reduction by end of series


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    dates = pd.date_range("2023-01-01", "2026-06-01", freq="MS")
    total_months = len(dates)

    rows = []
    for date in dates:
        for model in VEHICLE_MODELS:
            for category in DEFECT_CATEGORIES:
                for line in PLANT_LINES:
                    production_volume = int(rng.normal(850, 90))
                    production_volume = max(production_volume, 400)

                    base_rate = BASE_DEFECT_RATE[category] * MODEL_MULTIPLIER[model]

                    # Gradual quality improvement trend from Jan 2025 onward.
                    if date >= IMPROVEMENT_START:
                        months_into_improvement = (
                            (date.year - IMPROVEMENT_START.year) * 12
                            + (date.month - IMPROVEMENT_START.month)
                        )
                        months_remaining = total_months - 1 - (
                            (IMPROVEMENT_START.year - dates[0].year) * 12
                            + (IMPROVEMENT_START.month - dates[0].month)
                        )
                        progress = min(months_into_improvement / max(months_remaining, 1), 1.0)
                        base_rate *= 1 - IMPROVEMENT_TOTAL_REDUCTION * progress

                    # Anomaly spike: bad component batch in Electric Van electrical
                    # systems, May-Aug 2024.
                    if (
                        model == ANOMALY_MODEL
                        and category == ANOMALY_CATEGORY
                        and ANOMALY_START <= date <= ANOMALY_END
                    ):
                        base_rate *= 2.8

                    noise = rng.normal(1.0, 0.12)
                    defect_rate_pct = max(base_rate * noise, 0.1)

                    defect_count = int(round(production_volume * defect_rate_pct / 100))
                    defect_count = max(defect_count, 0)

                    warranty_claims = int(
                        max(rng.normal(defect_count * 0.55, defect_count * 0.15 + 1), 0)
                    )

                    complaint_rate_pct = defect_rate_pct * COMPLAINT_RATE_MULTIPLIER
                    complaint_mean = production_volume * complaint_rate_pct / 100
                    customer_complaints = int(
                        max(rng.normal(complaint_mean, complaint_mean * 0.15 + 1), 0)
                    )

                    severity_score = rng.normal(
                        3.0 if category in ("Engine", "Transmission") else 2.3, 0.8
                    )
                    if (
                        model == ANOMALY_MODEL
                        and category == ANOMALY_CATEGORY
                        and ANOMALY_START <= date <= ANOMALY_END
                    ):
                        severity_score += 1.2
                    severity_score = float(np.clip(round(severity_score), 1, 5))

                    actual_defect_rate = (
                        (defect_count / production_volume) * 100 if production_volume else 0
                    )

                    rows.append(
                        {
                            "date": date,
                            "vehicle_model": model,
                            "defect_category": category,
                            "plant_line": line,
                            "production_volume": production_volume,
                            "defect_count": defect_count,
                            "warranty_claims": warranty_claims,
                            "customer_complaints": customer_complaints,
                            "severity_score": severity_score,
                            "defect_rate": round(actual_defect_rate, 3),
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} records across {df['date'].nunique()} months -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
