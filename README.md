# Commercial Vehicle Quality Intelligence Dashboard

An interactive quality analytics dashboard for commercial vehicle manufacturing — defect trend analysis, ML-based anomaly detection, root-cause surfacing, and short-term forecasting.

## Why this project

Built as a proof-of-work project for the **Data Analyst – Quality and Customer Satisfaction** role at **ALTEN Technology USA**'s Greensboro, NC engineering center, which delivers product development and engineering services for commercial vehicle, heavy truck, EV, rail, and energy clients. The dashboard is scoped and worded around that role specifically:

- Vehicle models (Class 8 Truck, Electric Van, Commercial Bus, Heavy Hauler) mirror Greensboro's commercial vehicle / heavy truck / EV focus; the header also nods to the rail and energy programs the same center supports.
- The **PP100 (complaints per 100 units)** KPI is the same problems-per-100-vehicles metric used industry-wide (e.g. J.D. Power's Initial Quality Study) for tracking customer-reported issues against production volume — a direct hit on the "Customer Satisfaction" half of the role.
- The anomaly detection, forecasting, and root-cause sections map to the JD's call for statistical techniques, early detection of emerging quality issues, and ML-based analysis.
- The footer notes this is a Python prototype today, with the same logic intended to carry over to Power BI + Azure Analytics for production reporting — the JD's listed tooling.

## Features

- **Interactive filters** — vehicle model, defect category, and plant line, all reactive
- **KPI cards** — total defects, warranty claims, customer complaints, average defect rate, and PP100 (complaints per 100 units), each with a period-over-period trend arrow (red = worse, green = better)
- **Defect & warranty trend** — monthly time series
- **Defects by category** — horizontal bar chart
- **Anomaly detection** — Isolation Forest (scikit-learn) flags anomalous months in the defect-rate trend
- **3-month forecast** — linear trend projection with the forecast window highlighted
- **Customer satisfaction impact by vehicle model** — grouped bar of warranty claims vs. complaints
- **Defect heatmap** — vehicle model × defect category
- **Root cause analysis** — automatically surfaces the top 3 model + defect-category combinations with the largest month-over-month defect rate increase, each with a plain-language inspection recommendation
- **Top 10 worst defect incidents** — sortable data table ranked by severity and defect count
- **Export** — download the currently filtered dataset as CSV, or print/export a one-page summary view

## Dataset

`generate_data.py` produces a synthetic but realistic dataset (`quality_data.csv`, 3,528 rows) spanning 42 months (Jan 2023 – Jun 2026) across 4 vehicle models, 7 defect categories, and 3 plant lines. It bakes in two intentional patterns used for the anomaly-detection and root-cause features:

- A defect-rate spike in **Electric Van / Electrical** components from May–Aug 2024, simulating a bad component batch that was caught through automated monitoring
- A gradual ~15% defect-rate improvement trend starting January 2025

The script is deterministic (fixed random seed) and safe to re-run any time — it simply regenerates and overwrites `quality_data.csv`.

## Stack

Python, Pandas, NumPy, Plotly Dash, scikit-learn (Isolation Forest)

## Running it

```bash
# Install dependencies
pip install pandas numpy plotly dash scikit-learn

# Generate the dataset (already included, but safe to re-run)
python generate_data.py

# Run the dashboard
python app.py
```

Then open http://localhost:8050

## Project structure

```
app.py              # Dash application: layout, callbacks, charts
generate_data.py     # Synthetic dataset generator
quality_data.csv     # Generated dataset (3,528 rows)
assets/style.css      # Dashboard styling
```
