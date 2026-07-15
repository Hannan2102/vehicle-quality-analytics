"""
Commercial Vehicle Quality Intelligence Dashboard

Interactive Plotly Dash app for monitoring manufacturing quality across a
commercial vehicle production line: defect trends, ML-based anomaly
detection, short-term forecasting, and automated root-cause surfacing.

Run with: python app.py
Then open http://localhost:8050
"""

import os
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from sklearn.ensemble import IsolationForest

import fmea

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
NAVY = "#1F3864"
RED = "#E63946"
GREEN = "#2DC653"
AMBER = "#F4A261"
SLATE = "#5C6B7A"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F4F6F9"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = "quality_data.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

VEHICLE_MODELS = sorted(df["vehicle_model"].unique())
DEFECT_CATEGORIES = sorted(df["defect_category"].unique())
PLANT_LINES = sorted(df["plant_line"].unique())

# FMEA risk table is a standing engineering reference computed once from the
# full historical dataset — it isn't reactive to the dashboard filters,
# consistent with how FMEAs are periodically reviewed reference documents.
FMEA_TABLE = fmea.compute_fmea_table(df)


def kpi_card(card_id: str, label: str) -> html.Div:
    """A single KPI card with a value and a trend indicator, updated via callback."""
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(id=f"{card_id}-value", className="kpi-value"),
            html.Div(id=f"{card_id}-trend", className="kpi-trend"),
        ],
    )


def trend_badge(current: float, previous: float, higher_is_bad: bool = True):
    """Returns (text, color) for a trend badge comparing current vs prior period."""
    if previous == 0 or pd.isna(previous):
        return "n/a vs prior period", SLATE
    pct_change = (current - previous) / previous * 100
    arrow = "▲" if pct_change >= 0 else "▼"
    is_bad = (pct_change >= 0) if higher_is_bad else (pct_change < 0)
    color = RED if is_bad else GREEN
    return f"{arrow} {abs(pct_change):.1f}% vs prior period", color


def month_bounds(dates: pd.Series):
    """Latest month and the month before it, for period-over-period KPI trends."""
    unique_months = sorted(dates.unique())
    latest = unique_months[-1]
    prior = unique_months[-2] if len(unique_months) > 1 else None
    return latest, prior


def compute_root_causes(frame: pd.DataFrame, top_n: int = 3):
    """
    Surfaces the model + defect-category combinations whose defect rate rose
    the most versus the prior month, i.e. the top emerging quality issues.
    """
    monthly = (
        frame.groupby(["date", "vehicle_model", "defect_category"])["defect_rate"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    if monthly.empty:
        return []

    latest_month = monthly["date"].max()
    months = sorted(monthly["date"].unique())
    if len(months) < 2:
        return []
    prior_month = months[-2]

    latest = monthly[monthly["date"] == latest_month].set_index(
        ["vehicle_model", "defect_category"]
    )["defect_rate"]
    prior = monthly[monthly["date"] == prior_month].set_index(
        ["vehicle_model", "defect_category"]
    )["defect_rate"]

    joined = pd.DataFrame({"current": latest, "prior": prior}).dropna()
    joined = joined[joined["prior"] > 0]
    joined["pct_change"] = (joined["current"] - joined["prior"]) / joined["prior"] * 100
    worst = joined.sort_values("pct_change", ascending=False).head(top_n)

    issues = []
    for (model, category), row in worst.iterrows():
        issues.append(
            {
                "model": model,
                "category": category,
                "pct_change": row["pct_change"],
                "current_rate": row["current"],
                "recommendation": (
                    f"Recommend inspection of {model} {category} components — "
                    f"{row['pct_change']:.0f}% increase vs prior period"
                ),
            }
        )
    return issues


def detect_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    """Flags anomalous months in the overall defect-rate trend using Isolation Forest."""
    monthly = frame.groupby("date")["defect_rate"].mean().reset_index().sort_values("date")
    if len(monthly) < 8:
        monthly["anomaly"] = False
        return monthly

    model = IsolationForest(contamination=0.1, random_state=42)
    features = monthly[["defect_rate"]].values
    preds = model.fit_predict(features)
    monthly["anomaly"] = preds == -1
    return monthly


def forecast_next_periods(frame: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """Simple linear-trend projection of overall defect rate for the next N months."""
    monthly = frame.groupby("date")["defect_rate"].mean().reset_index().sort_values("date")
    if len(monthly) < 3:
        return pd.DataFrame(columns=["date", "defect_rate"])

    x = np.arange(len(monthly))
    y = monthly["defect_rate"].values
    slope, intercept = np.polyfit(x, y, 1)

    last_date = monthly["date"].max()
    future_dates = pd.date_range(last_date, periods=periods + 1, freq="MS")[1:]
    future_x = np.arange(len(monthly), len(monthly) + periods)
    future_y = slope * future_x + intercept

    return pd.DataFrame({"date": future_dates, "defect_rate": future_y})


CHART_LAYOUT_DEFAULTS = dict(
    font=dict(family="Segoe UI, Roboto, Helvetica, Arial, sans-serif", color="#2B2F36"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=30, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Commercial Vehicle Quality Intelligence Dashboard"
server = app.server


def build_filter_bar():
    return html.Div(
        className="filter-bar",
        children=[
            html.Div(
                [
                    html.Label("Vehicle Model"),
                    dcc.Dropdown(
                        id="filter-model",
                        options=[{"label": m, "value": m} for m in VEHICLE_MODELS],
                        multi=True,
                        placeholder="All models",
                    ),
                ],
                className="filter-item",
            ),
            html.Div(
                [
                    html.Label("Defect Category"),
                    dcc.Dropdown(
                        id="filter-category",
                        options=[{"label": c, "value": c} for c in DEFECT_CATEGORIES],
                        multi=True,
                        placeholder="All categories",
                    ),
                ],
                className="filter-item",
            ),
            html.Div(
                [
                    html.Label("Plant Line"),
                    dcc.Dropdown(
                        id="filter-line",
                        options=[{"label": p, "value": p} for p in PLANT_LINES],
                        multi=True,
                        placeholder="All lines",
                    ),
                ],
                className="filter-item",
            ),
            html.Div(
                [
                    html.Button("Download Filtered CSV", id="btn-download-csv", className="export-btn"),
                    html.Button("Print / Export Summary", id="btn-print", className="export-btn secondary"),
                    dcc.Download(id="download-csv"),
                ],
                className="filter-item export-actions",
            ),
        ],
    )


app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="header-banner",
            children=[
                html.Div(
                    [
                        html.H1("Commercial Vehicle Quality Intelligence Dashboard"),
                        html.P(
                            "Defect monitoring, ML-based anomaly detection, and forecasting "
                            "built around the quality and customer satisfaction data problems "
                            "commercial vehicle, heavy truck, EV, rail, and energy engineering "
                            "programs deal with daily."
                        ),
                        html.Div(
                            className="industry-badges",
                            children=[
                                html.Span(label, className="industry-badge")
                                for label in [
                                    "Commercial Vehicle",
                                    "Heavy Truck",
                                    "Electric Vehicle",
                                    "Rail",
                                    "Energy",
                                ]
                            ],
                        ),
                    ]
                ),
                html.Div(id="header-date-range", className="header-meta"),
            ],
        ),
        build_filter_bar(),
        html.Div(
            className="kpi-row",
            children=[
                kpi_card("kpi-defects", "Total Defects"),
                kpi_card("kpi-warranty", "Warranty Claims"),
                kpi_card("kpi-complaints", "Customer Complaints"),
                kpi_card("kpi-rate", "Avg Defect Rate"),
                kpi_card("kpi-pp100", "Complaints per 100 Units (PP100)"),
            ],
        ),
        html.Div(
            className="chart-grid two-col",
            children=[
                html.Div(dcc.Graph(id="chart-trend"), className="chart-card"),
                html.Div(dcc.Graph(id="chart-category"), className="chart-card"),
            ],
        ),
        html.Div(
            className="chart-grid two-col",
            children=[
                html.Div(dcc.Graph(id="chart-anomaly"), className="chart-card"),
                html.Div(dcc.Graph(id="chart-forecast"), className="chart-card"),
            ],
        ),
        html.Div(
            className="chart-grid two-col",
            children=[
                html.Div(dcc.Graph(id="chart-impact"), className="chart-card"),
                html.Div(dcc.Graph(id="chart-heatmap"), className="chart-card"),
            ],
        ),
        html.Div(
            className="section-card",
            id="print-section",
            children=[
                html.H2("Root Cause Analysis — Top 3 Quality Issues This Month"),
                html.Div(id="root-cause-list"),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                html.H2("Top 10 Worst Defect Incidents"),
                html.P(
                    "Ranked by severity score and defect count. Click a column header to sort.",
                    className="section-subtitle",
                ),
                dash_table.DataTable(
                    id="incidents-table",
                    columns=[
                        {"name": "Date", "id": "date"},
                        {"name": "Vehicle Model", "id": "vehicle_model"},
                        {"name": "Defect Category", "id": "defect_category"},
                        {"name": "Plant Line", "id": "plant_line"},
                        {"name": "Defect Count", "id": "defect_count"},
                        {"name": "Severity", "id": "severity_score"},
                        {"name": "Defect Rate (%)", "id": "defect_rate"},
                    ],
                    sort_action="native",
                    page_size=10,
                    style_as_list_view=True,
                    style_header={
                        "backgroundColor": NAVY,
                        "color": "white",
                        "fontWeight": "600",
                        "border": "none",
                    },
                    style_cell={
                        "padding": "10px 14px",
                        "fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": "1px solid #E7EAEE",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#FAFBFC",
                        }
                    ],
                ),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                html.H2("FMEA-Style Risk Prioritization"),
                html.P(
                    "Severity × Occurrence × Detection risk scoring by defect category, ranked "
                    "by Risk Priority Number (RPN). Occurrence is derived from this dataset's "
                    "actual defect rates; Severity is derived from actual severity scores; "
                    "Detection is an engineering-judgment estimate of how hard each failure "
                    "mode is to catch before it reaches the customer — standard FMEA practice.",
                    className="section-subtitle",
                ),
                dash_table.DataTable(
                    id="fmea-table",
                    data=FMEA_TABLE.to_dict("records"),
                    columns=[
                        {"name": "Defect Category", "id": "defect_category"},
                        {"name": "Failure Mode", "id": "failure_mode"},
                        {"name": "Effect", "id": "effect"},
                        {"name": "S", "id": "severity"},
                        {"name": "O", "id": "occurrence"},
                        {"name": "D", "id": "detection"},
                        {"name": "RPN", "id": "rpn"},
                        {"name": "Recommended Action", "id": "recommended_action"},
                    ],
                    sort_action="native",
                    page_size=10,
                    style_as_list_view=True,
                    style_header={
                        "backgroundColor": NAVY,
                        "color": "white",
                        "fontWeight": "600",
                        "border": "none",
                    },
                    style_cell={
                        "padding": "10px 14px",
                        "fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                        "fontSize": "13.5px",
                        "border": "none",
                        "borderBottom": "1px solid #E7EAEE",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#FAFBFC",
                        }
                    ],
                ),
            ],
        ),
        dcc.Store(id="filtered-data-store"),
        html.Div(
            className="footer",
            children=(
                "Built by Abdul Hannan — prepared for the Data Analyst, Quality & Customer "
                "Satisfaction role at ALTEN Technology USA's Greensboro, NC engineering center. "
                "Python prototype today; the same trend, anomaly-detection, and forecasting "
                "logic maps directly onto Power BI + Azure Analytics for production reporting."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def apply_filters(models, categories, lines):
    frame = df
    if models:
        frame = frame[frame["vehicle_model"].isin(models)]
    if categories:
        frame = frame[frame["defect_category"].isin(categories)]
    if lines:
        frame = frame[frame["plant_line"].isin(lines)]
    return frame


@app.callback(
    Output("kpi-defects-value", "children"),
    Output("kpi-defects-trend", "children"),
    Output("kpi-defects-trend", "style"),
    Output("kpi-warranty-value", "children"),
    Output("kpi-warranty-trend", "children"),
    Output("kpi-warranty-trend", "style"),
    Output("kpi-complaints-value", "children"),
    Output("kpi-complaints-trend", "children"),
    Output("kpi-complaints-trend", "style"),
    Output("kpi-rate-value", "children"),
    Output("kpi-rate-trend", "children"),
    Output("kpi-rate-trend", "style"),
    Output("kpi-pp100-value", "children"),
    Output("kpi-pp100-trend", "children"),
    Output("kpi-pp100-trend", "style"),
    Output("chart-trend", "figure"),
    Output("chart-category", "figure"),
    Output("chart-anomaly", "figure"),
    Output("chart-forecast", "figure"),
    Output("chart-impact", "figure"),
    Output("chart-heatmap", "figure"),
    Output("root-cause-list", "children"),
    Output("incidents-table", "data"),
    Output("header-date-range", "children"),
    Output("filtered-data-store", "data"),
    Input("filter-model", "value"),
    Input("filter-category", "value"),
    Input("filter-line", "value"),
)
def update_dashboard(models, categories, lines):
    frame = apply_filters(models, categories, lines)

    # --- KPIs with period-over-period trend ---
    latest_month, prior_month = month_bounds(frame["date"])
    latest_slice = frame[frame["date"] == latest_month]
    prior_slice = frame[frame["date"] == prior_month] if prior_month is not None else frame.iloc[0:0]

    total_defects = int(frame["defect_count"].sum())
    total_warranty = int(frame["warranty_claims"].sum())
    total_complaints = int(frame["customer_complaints"].sum())
    avg_rate = frame["defect_rate"].mean() if len(frame) else 0

    def pp100(slice_frame: pd.DataFrame) -> float:
        """Complaints per 100 units produced — the industry-standard quality
        metric (as used in J.D. Power's Initial Quality Study) for tracking
        customer-reported problems relative to production volume."""
        volume = slice_frame["production_volume"].sum()
        return (slice_frame["customer_complaints"].sum() / volume * 100) if volume else 0

    avg_pp100 = pp100(frame)

    defects_txt, defects_color = trend_badge(
        latest_slice["defect_count"].sum(), prior_slice["defect_count"].sum()
    )
    warranty_txt, warranty_color = trend_badge(
        latest_slice["warranty_claims"].sum(), prior_slice["warranty_claims"].sum()
    )
    complaints_txt, complaints_color = trend_badge(
        latest_slice["customer_complaints"].sum(), prior_slice["customer_complaints"].sum()
    )
    rate_txt, rate_color = trend_badge(
        latest_slice["defect_rate"].mean() if len(latest_slice) else 0,
        prior_slice["defect_rate"].mean() if len(prior_slice) else 0,
    )
    pp100_txt, pp100_color = trend_badge(pp100(latest_slice), pp100(prior_slice))

    # --- Defect & warranty trend ---
    trend_monthly = (
        frame.groupby("date")[["defect_count", "warranty_claims"]].sum().reset_index()
    )
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend_monthly["date"],
            y=trend_monthly["defect_count"],
            mode="lines+markers",
            name="Defects",
            line=dict(color=NAVY, width=3),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=trend_monthly["date"],
            y=trend_monthly["warranty_claims"],
            mode="lines+markers",
            name="Warranty Claims",
            line=dict(color=AMBER, width=3),
        )
    )
    fig_trend.update_layout(title="Defect & Warranty Trend", **CHART_LAYOUT_DEFAULTS)

    # --- Defects by category ---
    by_category = (
        frame.groupby("defect_category")["defect_count"].sum().sort_values().reset_index()
    )
    fig_category = go.Figure(
        go.Bar(
            x=by_category["defect_count"],
            y=by_category["defect_category"],
            orientation="h",
            marker_color=NAVY,
        )
    )
    fig_category.update_layout(title="Defects by Category", **CHART_LAYOUT_DEFAULTS)

    # --- Anomaly detection ---
    anomaly_df = detect_anomalies(frame)
    fig_anomaly = go.Figure()
    fig_anomaly.add_trace(
        go.Scatter(
            x=anomaly_df["date"],
            y=anomaly_df["defect_rate"],
            mode="lines+markers",
            name="Defect Rate",
            line=dict(color=NAVY, width=3),
        )
    )
    flagged = anomaly_df[anomaly_df["anomaly"]]
    fig_anomaly.add_trace(
        go.Scatter(
            x=flagged["date"],
            y=flagged["defect_rate"],
            mode="markers",
            name="Anomaly Flagged",
            marker=dict(color=RED, size=13, symbol="x"),
        )
    )
    fig_anomaly.update_layout(
        title="Anomaly Detection (Isolation Forest)", **CHART_LAYOUT_DEFAULTS
    )

    # --- Forecast ---
    history = frame.groupby("date")["defect_rate"].mean().reset_index().sort_values("date")
    forecast_df = forecast_next_periods(frame, periods=3)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["defect_rate"],
            mode="lines+markers",
            name="Historical",
            line=dict(color=NAVY, width=3),
        )
    )
    if not forecast_df.empty:
        bridge = pd.concat([history.tail(1), forecast_df])
        fig_forecast.add_trace(
            go.Scatter(
                x=bridge["date"],
                y=bridge["defect_rate"],
                mode="lines+markers",
                name="3-Month Forecast",
                line=dict(color=AMBER, width=3, dash="dash"),
            )
        )
        fig_forecast.add_vrect(
            x0=forecast_df["date"].min(),
            x1=forecast_df["date"].max(),
            fillcolor=AMBER,
            opacity=0.12,
            line_width=0,
        )
    fig_forecast.update_layout(title="3-Month Defect Rate Forecast", **CHART_LAYOUT_DEFAULTS)

    # --- Customer impact by vehicle model ---
    impact = (
        frame.groupby("vehicle_model")[["warranty_claims", "customer_complaints"]]
        .sum()
        .reset_index()
    )
    fig_impact = go.Figure()
    fig_impact.add_trace(
        go.Bar(
            x=impact["vehicle_model"],
            y=impact["warranty_claims"],
            name="Warranty Claims",
            marker_color=NAVY,
        )
    )
    fig_impact.add_trace(
        go.Bar(
            x=impact["vehicle_model"],
            y=impact["customer_complaints"],
            name="Customer Complaints",
            marker_color=RED,
        )
    )
    fig_impact.update_layout(
        title="Customer Satisfaction Impact by Vehicle Model", barmode="group", **CHART_LAYOUT_DEFAULTS
    )

    # --- Heatmap ---
    heat = frame.pivot_table(
        index="vehicle_model",
        columns="defect_category",
        values="defect_rate",
        aggfunc="mean",
    ).reindex(columns=DEFECT_CATEGORIES)
    fig_heatmap = go.Figure(
        go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale=[[0, "#EAF0FA"], [0.5, "#5C82B3"], [1, NAVY]],
            colorbar=dict(title="Defect Rate (%)"),
        )
    )
    fig_heatmap.update_layout(title="Defect Heatmap — Model × Category", **CHART_LAYOUT_DEFAULTS)

    # --- Root cause analysis ---
    issues = compute_root_causes(frame)
    if issues:
        root_cause_children = [
            html.Div(
                className="root-cause-item",
                children=[
                    html.Div(f"#{i + 1}", className="root-cause-rank"),
                    html.Div(
                        [
                            html.Div(
                                f"{issue['model']} — {issue['category']}",
                                className="root-cause-title",
                            ),
                            html.Div(issue["recommendation"], className="root-cause-recommendation"),
                        ],
                        className="root-cause-body",
                    ),
                    html.Div(
                        f"+{issue['pct_change']:.0f}%",
                        className="root-cause-pct",
                        style={"color": RED if issue["pct_change"] > 0 else GREEN},
                    ),
                ],
            )
            for i, issue in enumerate(issues)
        ]
    else:
        root_cause_children = [html.P("Not enough data to compute month-over-month root causes.")]

    # --- Incidents table ---
    incidents = frame.sort_values(
        ["severity_score", "defect_count"], ascending=False
    ).head(10).copy()
    incidents["date"] = incidents["date"].dt.strftime("%Y-%m")
    incidents_data = incidents[
        [
            "date",
            "vehicle_model",
            "defect_category",
            "plant_line",
            "defect_count",
            "severity_score",
            "defect_rate",
        ]
    ].to_dict("records")

    date_range_txt = ""
    if len(frame):
        date_range_txt = (
            f"{frame['date'].min().strftime('%b %Y')} – {frame['date'].max().strftime('%b %Y')}"
        )

    return (
        f"{total_defects:,}",
        defects_txt,
        {"color": defects_color},
        f"{total_warranty:,}",
        warranty_txt,
        {"color": warranty_color},
        f"{total_complaints:,}",
        complaints_txt,
        {"color": complaints_color},
        f"{avg_rate:.2f}%",
        rate_txt,
        {"color": rate_color},
        f"{avg_pp100:.1f}",
        pp100_txt,
        {"color": pp100_color},
        fig_trend,
        fig_category,
        fig_anomaly,
        fig_forecast,
        fig_impact,
        fig_heatmap,
        root_cause_children,
        incidents_data,
        date_range_txt,
        frame.to_json(date_format="iso", orient="split"),
    )


@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("filtered-data-store", "data"),
    prevent_initial_call=True,
)
def download_filtered_csv(_n_clicks, stored_json):
    frame = pd.read_json(StringIO(stored_json), orient="split")
    filename = f"quality_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return dcc.send_data_frame(frame.to_csv, filename, index=False)


app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) { window.print(); }
        return "";
    }
    """,
    Output("btn-print", "title"),
    Input("btn-print", "n_clicks"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
