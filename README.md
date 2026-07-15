# Commercial Vehicle Quality Intelligence Dashboard

An interactive quality analytics dashboard for commercial vehicle manufacturing — defect trend analysis, ML-based anomaly detection, classification, clustering, FMEA-style risk prioritization, and short-term forecasting.

## Why this project

Built as a proof-of-work project for the **Data Analyst – Quality and Customer Satisfaction** role at **ALTEN Technology USA**'s Greensboro, NC engineering center, which delivers product development and engineering services for commercial vehicle, heavy truck, EV, rail, and energy clients. The dashboard is scoped and worded around that role specifically:

- Vehicle models (Class 8 Truck, Electric Van, Commercial Bus, Heavy Hauler) mirror Greensboro's commercial vehicle / heavy truck / EV focus; the header also nods to the rail and energy programs the same center supports.
- The **PP100 (complaints per 100 units)** KPI is the same problems-per-100-vehicles metric used industry-wide (e.g. J.D. Power's Initial Quality Study) for tracking customer-reported issues against production volume — a direct hit on the "Customer Satisfaction" half of the role.
- The anomaly detection, forecasting, and root-cause sections map to the JD's call for statistical techniques, early detection of emerging quality issues, and ML-based analysis.
- `EDA.ipynb` and the SQL layer (`build_database.py`, `queries.sql`) cover the JD's Jupyter Notebook and SQL requirements directly, rather than just being mentioned in passing.
- `ML_Analysis.ipynb` adds the classification and clustering techniques named in the JD's machine-learning requirement, on top of the anomaly detection already in the dashboard.
- The FMEA-style risk table (`fmea.py`) is a standard automotive/commercial-vehicle quality engineering tool (AIAG-VDA Severity × Occurrence × Detection scoring), giving the project a data-driven bridge into the engineering side of quality work without overclaiming design-engineering skills the role doesn't require.
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
- **FMEA-style risk prioritization** — Severity × Occurrence × Detection scoring by defect category, ranked by Risk Priority Number, with a recommended action per category
- **Export** — download the currently filtered dataset as CSV, or print/export a one-page summary view

## Dataset

`generate_data.py` produces a synthetic but realistic dataset (`quality_data.csv`, 3,528 rows) spanning 42 months (Jan 2023 – Jun 2026) across 4 vehicle models, 7 defect categories, and 3 plant lines. It bakes in two intentional patterns used for the anomaly-detection and root-cause features:

- A defect-rate spike in **Electric Van / Electrical** components from May–Aug 2024, simulating a bad component batch that was caught through automated monitoring
- A gradual ~15% defect-rate improvement trend starting January 2025

The script is deterministic (fixed random seed) and safe to re-run any time — it simply regenerates and overwrites `quality_data.csv`.

## EDA notebook

`EDA.ipynb` is the exploratory analysis behind the dashboard: descriptive statistics, monthly trend and category/model breakdowns, a correlation matrix across quality signals, and two hypothesis tests (Welch's t-test) —

- confirming the 2023 → 2025 defect-rate improvement is statistically significant (not just a visual trend), and
- confirming the Electric Van / Electrical anomaly window is statistically distinct from baseline, not noise.

It closes with a cross-check that pulls the PP100 metric via raw SQL against the SQLite database and confirms it matches the pandas computation exactly.

## SQL layer

`build_database.py` loads `quality_data.csv` into a SQLite database (`quality_data.db`). `queries.sql` holds standalone SQL — KPI rollups, the PP100 metric by vehicle model, a window-function (LAG) root-cause query, and an anomaly-flagging query — that reproduces the dashboard's key numbers in SQL rather than pandas.

## ML notebook — classification & clustering

`ML_Analysis.ipynb` goes beyond the dashboard's unsupervised anomaly detection:

- **Classification** — a Random Forest / Logistic Regression comparison predicting whether an incident will be high-severity (severity ≥ 4, ~12% of records) from defect rate, defect count, warranty claims, customer complaints, and production volume. All five numeric signals outrank every categorical feature (vehicle model, defect category, plant line) in predictive importance — severity risk is driven more by how bad the signal already is than by where it happened.
- **Clustering** — KMeans segmentation of the 28 vehicle-model × defect-category combinations by quality profile (defect rate, severity, warranty claims, complaints), with the best `k` chosen by silhouette score. It separates a small chronic high-risk cluster (Engine and Transmission, across every vehicle model) from a larger stable cluster — and that split independently matches the FMEA table's top-2 risk ranking below.

## FMEA-style risk prioritization

`fmea.py` (shared by the dashboard and `ML_Analysis.ipynb`) computes a Severity × Occurrence × Detection Risk Priority Number per defect category — the standard AIAG-VDA FMEA framework used across automotive/commercial-vehicle quality engineering. Occurrence and Severity are derived from this dataset's actual defect rates and severity scores; Detection is an engineering-judgment estimate of how hard each failure mode is to catch pre-shipment, which is normal FMEA practice.

## Stack

Python, Pandas, NumPy, Plotly Dash, scikit-learn (Isolation Forest, Random Forest, Logistic Regression, KMeans), Jupyter, SQLite

## Running it

```bash
# Install dependencies
pip install pandas numpy plotly dash scikit-learn matplotlib seaborn scipy jupyter

# Generate the dataset (already included, but safe to re-run)
python generate_data.py

# Build the SQLite database for the SQL layer / notebook
python build_database.py

# Run the dashboard
python app.py
```

Then open http://localhost:8050

To explore the notebooks: `jupyter notebook EDA.ipynb` or `jupyter notebook ML_Analysis.ipynb`

## Project structure

```
app.py               # Dash application: layout, callbacks, charts
generate_data.py      # Synthetic dataset generator
quality_data.csv      # Generated dataset (3,528 rows)
build_database.py    # Loads quality_data.csv into quality_data.db (SQLite)
queries.sql           # Standalone SQL analysis queries
fmea.py               # Shared FMEA (Severity x Occurrence x Detection) risk table
EDA.ipynb             # Exploratory data analysis + hypothesis tests
ML_Analysis.ipynb     # Severity classifier + quality-profile clustering
assets/style.css       # Dashboard styling
```
