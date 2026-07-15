"""
FMEA-style (Failure Mode and Effects Analysis) risk prioritization, shared by
the dashboard (app.py) and the ML notebook so both show the same numbers.

Methodology, consistent with standard AIAG-VDA FMEA practice:
  - Occurrence (O) is derived from the actual defect_rate in the dataset,
    scaled 1-10 across categories (data-driven, not a guess).
  - Severity (S) is derived from the actual severity_score in the dataset,
    scaled from its 1-5 range to FMEA's standard 1-10 range.
  - Detection (D) is an engineering-judgment baseline per category (how hard
    a failure mode is to catch before it reaches the customer) — this is
    normal FMEA practice, since detection difficulty isn't something you can
    compute from defect counts alone.
  - Risk Priority Number = S x O x D, the standard FMEA prioritization score.
"""

import numpy as np
import pandas as pd

FAILURE_MODE_LIBRARY = {
    "Engine": {
        "failure_mode": "Turbocharger seal degradation / fuel injector fouling",
        "effect": "Loss of power, increased emissions, potential engine shutdown",
    },
    "Transmission": {
        "failure_mode": "Clutch pack wear / shift solenoid failure",
        "effect": "Harsh or delayed shifting, drivetrain slippage",
    },
    "Brakes": {
        "failure_mode": "Brake pad wear sensor failure",
        "effect": "Delayed wear warning, reduced stopping performance",
    },
    "Electrical": {
        "failure_mode": "Battery management system (BMS) cell imbalance / harness insulation breakdown",
        "effect": "Reduced range, thermal risk, unexpected power loss",
    },
    "Body/Frame": {
        "failure_mode": "Frame rail stress crack / weld fatigue",
        "effect": "Structural integrity risk, increased NVH",
    },
    "HVAC": {
        "failure_mode": "Cabin blower motor / compressor failure",
        "effect": "Loss of climate control, driver comfort and safety in extreme temperatures",
    },
    "Suspension": {
        "failure_mode": "Air suspension bag leak / bushing wear",
        "effect": "Ride height loss, uneven handling, accelerated component wear",
    },
}

# Detection difficulty baseline (1 = easily caught pre-shipment, 10 = often
# reaches the customer undetected). Engineering-judgment estimate per
# category, not derived from data.
BASE_DETECTION = {
    "Engine": 6,
    "Transmission": 6,
    "Brakes": 4,
    "Electrical": 8,
    "Body/Frame": 5,
    "HVAC": 4,
    "Suspension": 5,
}


def _recommended_action(rpn: int, category: str, failure_mode: str) -> str:
    primary_mode = failure_mode.split("/")[0].strip()
    if rpn >= 250:
        return f"High priority — add automated pre-shipment test targeting {primary_mode} in {category}."
    if rpn >= 120:
        return f"Moderate priority — increase inspection sampling and trend-monitor {category}."
    return f"Low priority — maintain current inspection cadence for {category}."


def compute_fmea_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Builds a Severity x Occurrence x Detection risk table, one row per
    defect category, ranked by Risk Priority Number (RPN) descending."""
    summary = (
        frame.groupby("defect_category")
        .agg(avg_severity=("severity_score", "mean"), avg_defect_rate=("defect_rate", "mean"))
        .reset_index()
    )

    min_rate, max_rate = summary["avg_defect_rate"].min(), summary["avg_defect_rate"].max()
    rate_span = max_rate - min_rate

    rows = []
    for _, row in summary.iterrows():
        category = row["defect_category"]
        lib = FAILURE_MODE_LIBRARY[category]

        severity = int(np.clip(round(row["avg_severity"] * 2), 1, 10))
        if rate_span > 0:
            occurrence = int(
                np.clip(round(1 + (row["avg_defect_rate"] - min_rate) / rate_span * 9), 1, 10)
            )
        else:
            occurrence = 5
        detection = BASE_DETECTION[category]
        rpn = severity * occurrence * detection

        rows.append(
            {
                "defect_category": category,
                "failure_mode": lib["failure_mode"],
                "effect": lib["effect"],
                "severity": severity,
                "occurrence": occurrence,
                "detection": detection,
                "rpn": rpn,
                "recommended_action": _recommended_action(rpn, category, lib["failure_mode"]),
            }
        )

    return pd.DataFrame(rows).sort_values("rpn", ascending=False).reset_index(drop=True)
