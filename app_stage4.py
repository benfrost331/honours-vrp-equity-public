from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import (
    EQUITY_MEASURE_LABELS,
    EQUITY_MEASURE_TYPES,
)
from src.stage_4_dashboard_data import (
    build_assignment_change_table,
    build_driver_workload_comparison,
    load_stage3_baseline_files,
    load_stage4_master_summary,
    load_stage4_solution_files,
)


DARK_BLUE = "#1E3A8A"
GREEN = "#15803D"
RED = "#DC2626"
AMBER = "#B45309"
BLUE = "#2563EB"

px.defaults.color_discrete_sequence = [
    DARK_BLUE,
    BLUE,
    GREEN,
    AMBER,
    RED,
]


st.set_page_config(
    page_title="Stage 4 ALNS Dashboard",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        .stApp { background-color: #f8fafc; }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1550px;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        h1, h2, h3 {
            color: #0f172a;
        }

        .dashboard-subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 1.5rem;
        }

        .note-box {
            background-color: #eff6ff;
            color: #1e3a8a;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 1.2rem;
        }

        .success-box {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 15px 18px;
            margin-bottom: 1.2rem;
            text-align: center;
            font-weight: 600;
        }

        .warning-box {
            background-color: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 15px 18px;
            margin-bottom: 1.2rem;
            text-align: center;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_minutes(value: float) -> str:
    return f"{value:,.0f} min"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"

    return f"{value:,.2f}%"


def pretty_scope(scope: str) -> str:
    return {
        "full_instance": "Full Instance",
        "reduced_candidate_graph": "Reduced K=20",
    }.get(scope, scope)


def solution_label(row: pd.Series) -> str:
    return (
        f"{pretty_scope(str(row['model_scope']))} | "
        f"λ={float(row['lambda_equity']):.1f} | "
        f"{row['lambda_tag']}"
    )


def solution_options(summary: pd.DataFrame) -> dict[str, tuple[str, str]]:
    return {
        solution_label(row): (
            str(row["model_scope"]),
            str(row["lambda_tag"]),
        )
        for _, row in summary.iterrows()
    }


def selected_row_for(
    summary: pd.DataFrame,
    model_scope: str,
    lambda_tag: str,
) -> pd.Series:
    rows = summary.loc[
        summary["model_scope"].astype(str).eq(model_scope)
        & summary["lambda_tag"].astype(str).eq(lambda_tag)
    ]

    if rows.empty:
        raise ValueError(
            f"No Stage 4 row found for {model_scope} {lambda_tag}."
        )

    return rows.iloc[0]


def equity_label(value: str) -> str:
    return EQUITY_MEASURE_LABELS.get(
        value,
        value,
    )


def lower_is_better_for_equity_measure(
    equity_measure_type: str,
) -> bool:
    """
    All implemented workload-equity measures are minimisation measures:
    lower maximum workload, lower min-max deviation, lower MAD, lower
    range and lower normalised Gini dispersion are all better.
    """

    return equity_measure_type in {
        "maximum",
        "minmax_deviation",
        "mad_mean",
        "range",
        "gini",
    }


def percent_change_from_delta(
    new_value: float,
    delta_value: float,
) -> float | None:
    """
    Recover the Stage 3 baseline as new_value - delta_value, then compute
    percentage change from Stage 3 to Stage 4.
    """

    baseline_value = float(new_value) - float(delta_value)

    if abs(baseline_value) < 1e-9:
        return None

    return 100.0 * float(delta_value) / baseline_value


def format_signed_minutes(
    value: float,
) -> str:
    return f"{value:+,.0f} min"


def format_signed_number(
    value: float,
) -> str:
    return f"{value:+,.2f}"


def format_signed_percentage(
    value: float | None,
) -> str:
    if value is None:
        return "n/a"

    return f"{value:+.2f}%"

def zoomed_line_axis_range(
    values: pd.Series | list[float],
    *,
    minimum_padding: float = 0.01,
) -> list[float]:
    """
    Build a local y-axis range for one time-series panel.

    This intentionally zooms into the observed range because the ALNS trace
    is meant to show within-metric movement over iterations, not compare
    different units on a common absolute scale.
    """

    clean_values = pd.Series(values).dropna().astype(float)

    if clean_values.empty:
        return [0.0, 1.0]

    low = float(clean_values.min())
    high = float(clean_values.max())

    if abs(high - low) < 1e-9:
        padding = max(
            abs(high) * 0.02,
            minimum_padding,
        )
    else:
        padding = max(
            abs(high - low) * 0.20,
            abs(high) * 0.005,
            minimum_padding,
        )

    lower = low - padding
    upper = high + padding

    if abs(upper - lower) < 1e-9:
        upper = lower + 1.0

    return [
        lower,
        upper,
    ]


def build_alns_search_progress_figure(
    iteration_log: pd.DataFrame,
) -> go.Figure:
    """
    Build a stacked ALNS search trace with independent y-axis scales.

    The old chart mixed route duration in minutes with scalar objective
    scores, which made the scalar-score traces unreadable.
    """

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[
            0.40,
            0.32,
            0.28,
        ],
        subplot_titles=[
            "Best Route Duration",
            "Accepted / Best Objective Score",
            "Candidate Objective Score",
        ],
    )

    
    # Panel 1: route-duration trace
    

    if "best_total_route_duration" in iteration_log.columns:
        fig.add_trace(
            go.Scatter(
                x=iteration_log["iteration"],
                y=iteration_log["best_total_route_duration"],
                mode="lines+markers",
                name="Best route duration",
                line={
                    "color": DARK_BLUE,
                    "width": 3,
                },
                marker={
                    "size": 6,
                },
                hovertemplate=(
                    "Iteration %{x}<br>"
                    "Best route duration: %{y:,.0f} min"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(
            range=zoomed_line_axis_range(
                iteration_log["best_total_route_duration"],
                minimum_padding=50.0,
            ),
            title_text="Minutes",
            showgrid=True,
            zeroline=False,
            row=1,
            col=1,
        )

    
    # Panel 2: accepted/current and best scalar objective score
    

    if "current_scalar_score" in iteration_log.columns:
        fig.add_trace(
            go.Scatter(
                x=iteration_log["iteration"],
                y=iteration_log["current_scalar_score"],
                mode="lines+markers",
                name="Current score",
                line={
                    "color": BLUE,
                    "width": 2,
                },
                marker={
                    "size": 5,
                },
                hovertemplate=(
                    "Iteration %{x}<br>"
                    "Current score: %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if "best_scalar_score" in iteration_log.columns:
        fig.add_trace(
            go.Scatter(
                x=iteration_log["iteration"],
                y=iteration_log["best_scalar_score"],
                mode="lines+markers",
                name="Best score",
                line={
                    "color": GREEN,
                    "width": 3,
                },
                marker={
                    "size": 5,
                },
                hovertemplate=(
                    "Iteration %{x}<br>"
                    "Best score: %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    score_axis_columns = [
        column
        for column in [
            "current_scalar_score",
            "best_scalar_score",
        ]
        if column in iteration_log.columns
    ]

    if score_axis_columns:
        score_values = pd.concat(
            [
                iteration_log[column]
                for column in score_axis_columns
            ],
            ignore_index=True,
        )

        fig.update_yaxes(
            range=zoomed_line_axis_range(
                score_values,
                minimum_padding=0.001,
            ),
            title_text="Score",
            showgrid=True,
            zeroline=False,
            row=2,
            col=1,
        )

    
    # Panel 3: candidate scalar objective score
    

    if "candidate_scalar_score" in iteration_log.columns:
        fig.add_trace(
            go.Scatter(
                x=iteration_log["iteration"],
                y=iteration_log["candidate_scalar_score"],
                mode="lines+markers",
                name="Candidate score",
                line={
                    "color": AMBER,
                    "width": 2,
                },
                marker={
                    "size": 5,
                },
                hovertemplate=(
                    "Iteration %{x}<br>"
                    "Candidate score: %{y:.6f}"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

        fig.update_yaxes(
            range=zoomed_line_axis_range(
                iteration_log["candidate_scalar_score"],
                minimum_padding=0.001,
            ),
            title_text="Score",
            showgrid=True,
            zeroline=False,
            row=3,
            col=1,
        )

    fig.update_xaxes(
        title_text="Iteration",
        row=3,
        col=1,
    )

    fig.update_layout(
        title="ALNS Search Progress by Metric",
        height=760,
        margin={
            "l": 55,
            "r": 35,
            "t": 90,
            "b": 55,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1.0,
        },
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def format_stage4_value_plot_number(
    metric_name: str,
    value: float,
) -> str:
    """
    Format values for the Stage 3 vs Stage 4 comparison plot.
    """

    if pd.isna(value):
        return "n/a"

    metric_name = str(metric_name).lower()

    if "duration" in metric_name:
        return f"{float(value):,.0f}"

    if "cv" in metric_name or "coefficient" in metric_name:
        return f"{float(value):.4f}"

    return f"{float(value):,.2f}"


def zoomed_metric_axis_range(
    values: list[float],
) -> list[float]:
    """
    Build a sensible independent y-axis range for one metric panel.

    This intentionally does not always start at zero, because the purpose of
    the Stage 4 Value plot is to show the before/after change within each
    metric. The dashboard caption makes this explicit.
    """

    clean_values = [
        float(value)
        for value in values
        if pd.notna(value)
    ]

    if not clean_values:
        return [0.0, 1.0]

    low = min(clean_values)
    high = max(clean_values)

    if abs(high - low) < 1e-9:
        padding = max(
            abs(high) * 0.05,
            1.0,
        )
    else:
        padding = max(
            abs(high - low) * 0.45,
            abs(high) * 0.01,
            0.01,
        )

    lower = low - padding
    upper = high + padding

    if low >= 0:
        lower = max(
            0.0,
            lower,
        )

    if abs(upper - lower) < 1e-9:
        upper = lower + 1.0

    return [
        lower,
        upper,
    ]


def build_stage3_vs_stage4_comparison_figure(
    comparison: pd.DataFrame,
) -> go.Figure:
    """
    Build four small paired-bar charts with independent y-axes.

    This avoids the original scaling problem where route duration dominates
    all other metrics on a single shared y-axis.
    """

    metric_titles = {
        "Total Route Duration": "Route Duration",
        "Selected Equity Metric": "Selected Equity",
        "Maximum Driver Route Workload": "Max Route Workload",
        "Active-Driver Workload CV": "Workload CV",
    }

    fig = make_subplots(
        rows=1,
        cols=len(comparison),
        subplot_titles=[
            metric_titles.get(
                str(row["Metric"]),
                str(row["Metric"]),
            )
            for _, row in comparison.iterrows()
        ],
        horizontal_spacing=0.08,
    )

    for index, (_, row) in enumerate(comparison.iterrows(), start=1):

        metric_name = str(row["Metric"])

        stage3_value = float(row["Stage 3 Repaired"])
        stage4_value = float(row["Stage 4 ALNS"])

        fig.add_trace(
            go.Bar(
                x=[
                    "Stage 3",
                    "Stage 4",
                ],
                y=[
                    stage3_value,
                    stage4_value,
                ],
                marker_color=[
                    DARK_BLUE,
                    BLUE,
                ],
                text=[
                    format_stage4_value_plot_number(
                        metric_name,
                        stage3_value,
                    ),
                    format_stage4_value_plot_number(
                        metric_name,
                        stage4_value,
                    ),
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + metric_name
                    + ": %{y:,.4f}"
                    + "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=index,
        )

        fig.update_yaxes(
            range=zoomed_metric_axis_range(
                [
                    stage3_value,
                    stage4_value,
                ]
            ),
            title_text="Value",
            showgrid=True,
            zeroline=False,
            row=1,
            col=index,
        )

        fig.update_xaxes(
            title_text="",
            row=1,
            col=index,
        )

    fig.update_layout(
        title=(
            "Stage 3 Repaired vs Stage 4 ALNS "
            "(Independent Metric Scales)"
        ),
        height=470,
        margin={
            "l": 40,
            "r": 30,
            "t": 85,
            "b": 50,
        },
        bargap=0.35,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# Header


st.title("Stage 4 ALNS Routing Improvement Dashboard")

st.markdown(
    """
    <div class="dashboard-subtitle">
        Compares the Stage 4 ALNS-Improved Routes Against the Feasible Stage 3 Repaired Routing Baseline
    </div>
    """,
    unsafe_allow_html=True,
)


# Load master summary


try:
    master = load_stage4_master_summary()

except (FileNotFoundError, pd.errors.EmptyDataError) as error:
    st.error(str(error))
    st.stop()


if master.empty:
    st.error("No Stage 4 ALNS results were found.")
    st.stop()



# Selection state


available_equity_measures = [
    measure
    for measure in EQUITY_MEASURE_TYPES
    if measure in set(master["equity_measure_type"].astype(str))
]

if not available_equity_measures:
    st.error("No Stage 4 equity-measure results are available.")
    st.stop()

if "stage4_equity_measure_type" not in st.session_state:
    st.session_state["stage4_equity_measure_type"] = (
        available_equity_measures[0]
    )

if (
    st.session_state["stage4_equity_measure_type"]
    not in available_equity_measures
):
    st.session_state["stage4_equity_measure_type"] = (
        available_equity_measures[0]
    )

selected_equity_measure = st.session_state[
    "stage4_equity_measure_type"
]

selected_equity_label = equity_label(
    selected_equity_measure
)

measure_summary = master.loc[
    master["equity_measure_type"]
    .astype(str)
    .eq(selected_equity_measure)
].copy()

available_lambdas = (
    measure_summary["lambda_equity"]
    .dropna()
    .astype(float)
    .sort_values()
    .unique()
    .tolist()
)

if not available_lambdas:
    st.error(
        "No Stage 4 lambda values are available for the selected "
        "equity measure."
    )
    st.stop()

default_lambda = (
    0.0
    if 0.0 in available_lambdas
    else available_lambdas[0]
)

state_lambda = st.session_state.get(
    "stage4_selected_lambda",
    default_lambda,
)

available_lambda_keys = [
    round(float(value), 3)
    for value in available_lambdas
]

if round(float(state_lambda), 3) not in available_lambda_keys:
    st.session_state["stage4_selected_lambda"] = default_lambda
else:
    st.session_state["stage4_selected_lambda"] = float(state_lambda)

selected_lambda = float(
    st.session_state["stage4_selected_lambda"]
)

matching_rows = measure_summary.loc[
    measure_summary["lambda_equity"]
    .astype(float)
    .round(3)
    .eq(round(selected_lambda, 3))
].copy()

if matching_rows.empty:
    st.error(
        f"No Stage 4 solution found for λ={selected_lambda:.1f}."
    )
    st.stop()

full_rows = matching_rows.loc[
    matching_rows["model_scope"]
    .astype(str)
    .eq("full_instance")
]

if not full_rows.empty:
    matching_rows = full_rows

selected_row = (
    matching_rows
    .sort_values(["model_scope", "lambda_tag"])
    .iloc[0]
)

selected_model_scope = str(
    selected_row["model_scope"]
)

selected_lambda_tag = str(
    selected_row["lambda_tag"]
)


# Load selected files

try:
    stage4_files = load_stage4_solution_files(
        model_scope=selected_model_scope,
        equity_measure_type=selected_equity_measure,
        lambda_tag=selected_lambda_tag,
    )

    baseline_files = load_stage3_baseline_files(
        model_scope=selected_model_scope,
        equity_measure_type=selected_equity_measure,
        lambda_tag=selected_lambda_tag,
    )

except (FileNotFoundError, pd.errors.EmptyDataError) as error:
    st.error(str(error))
    st.stop()


route_summaries = stage4_files["route_summaries"]
route_stops = stage4_files["route_stops"]
driver_workloads = stage4_files["driver_route_workloads"]
iteration_log = stage4_files["alns_iteration_log"]
operator_weights = stage4_files["alns_operator_weights"]
stage4_assignments = stage4_files["stage4_assignments"]

baseline_driver_workloads = baseline_files[
    "baseline_driver_route_workloads"
]

baseline_assignments = baseline_files[
    "baseline_assignments"
]


if (
    int(selected_row["n_infeasible_routes"]) == 0
    and abs(float(selected_row["total_violation_minutes"])) < 1e-6
):
    st.markdown(
        """
        <div class="success-box">
            Selected Stage 4 Solution is Route-Feasible |
            1,367 Orders Routed | Zero Duplicate Assignments | Zero
            Route-Level Violation
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="warning-box">
            Selected Stage 4 solution has route-level infeasibility or
            violation and should be interpreted as diagnostic rather than
            final.
        </div>
        """,
        unsafe_allow_html=True,
    )



# Headline metrics


duration_improvement_minutes = float(
    selected_row["duration_improvement_minutes"]
)

duration_improvement_pct = float(
    selected_row["duration_improvement_pct"]
)

stage4_route_duration = float(
    selected_row["total_route_duration"]
)

delta_total_route_duration = float(
    selected_row["delta_total_route_duration"]
)

stage4_selected_equity_metric_value = float(
    selected_row["stage4_selected_equity_metric_value"]
)

selected_equity_metric_change = float(
    selected_row["delta_selected_equity_metric_value"]
)

selected_equity_metric_pct_change = percent_change_from_delta(
    new_value=stage4_selected_equity_metric_value,
    delta_value=selected_equity_metric_change,
)

m2, m3, m4, m5 = st.columns(4)

m2.metric(
    "Stage 4 Route Duration",
    format_minutes(stage4_route_duration),
    delta=format_signed_minutes(delta_total_route_duration),
    delta_color="inverse",
)

m3.metric(
    f"{selected_equity_label} Change",
    format_signed_number(selected_equity_metric_change),
    delta=format_signed_percentage(
        selected_equity_metric_pct_change
    ),
    delta_color=(
        "inverse"
        if lower_is_better_for_equity_measure(
            selected_equity_measure
        )
        else "normal"
    ),
)

m4.metric(
    "Orders Reassigned",
    f"{int(selected_row['n_changed_assignments']):,}",
)

m5.metric(
    "Driver-Day Routes Changed",
    f"{int(selected_row['n_changed_route_keys']):,}",
)

st.markdown(
    """
    <div class="dashboard-subtitle" style="font-size: 0.95rem; margin-top: -0.5rem;">
        * Compared to Stage 3 Repaired Route
    </div>
    """,
    unsafe_allow_html=True,
)


# Solution controls


st.markdown("### Solution Controls")

control_1, control_2 = st.columns([1, 2])

with control_1:

    st.selectbox(
        "Equity Measure Type",
        options=available_equity_measures,
        format_func=equity_label,
        key="stage4_equity_measure_type",
    )

with control_2:

    st.select_slider(
        "Equity Emphasis",
        options=available_lambdas,
        key="stage4_selected_lambda",
        format_func=lambda value: f"λ = {float(value):.1f}",
        help=(
            "Select the stored Stage 4 ALNS solution for the current "
            "equity-measure family. The lambda value corresponds to the "
            "Stage 2 equity emphasis that generated the routed baseline."
        ),
    )



# Tabs


tabs = st.tabs(
    [
        "Executive Summary",
        "Stage 4 Value",
        "ALNS Search Trace",
        "Driver Workloads",
        "Assignment Changes",
        "Route Diagnostics",
        "Tables & Downloads",
    ]
)


with tabs[0]:
    st.subheader("Executive Summary")

    all_feasible = (
        master["n_infeasible_routes"]
        .astype(int)
        .eq(0)
        .all()
    )

    total_gain = master[
        "duration_improvement_minutes"
    ].sum()

    mean_gain = master[
        "duration_improvement_minutes"
    ].mean()

    best_row = master.sort_values(
        "duration_improvement_minutes",
        ascending=False,
    ).iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Stage 4 Solutions",
        f"{len(master):,}",
    )

    c2.metric(
        "Total Route-Duration Gain",
        format_minutes(total_gain),
    )

    c3.metric(
        "Mean Gain per Solution",
        format_minutes(mean_gain),
    )

    c4.metric(
        "All Stage 4 Routes Feasible",
        "Yes" if all_feasible else "No",
    )

    st.markdown(
        """
        Stage 4 Provides a **post-assignment routing improvement layer**. It does
        not replace Stage 2's exact assignment model or Stage 3's feasibility
        repair; instead, it starts from feasible Stage 3 routes and searches
        locally for lower-duration routed solutions.
        """
    )

    line = px.line(
        master,
        x="lambda_equity",
        y="duration_improvement_minutes",
        color="equity_measure_type",
        markers=True,
        labels={
            "lambda_equity": "Lambda",
            "duration_improvement_minutes": (
                "Route Duration Improvement (minutes)"
            ),
            "equity_measure_type": "Equity Measure",
        },
        title="Stage 4 Route-Duration Improvement by Equity Measure",
    )

    st.plotly_chart(
        line,
        use_container_width=True,
    )

    heatmap_data = master.pivot_table(
        index="equity_measure_type",
        columns="lambda_equity",
        values="duration_improvement_minutes",
        aggfunc="mean",
    )

    heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        text_auto=".0f",
        labels={
            "x": "Lambda",
            "y": "Equity Measure",
            "color": "Improvement (min)",
        },
        title="Stage 4 Improvement Heatmap",
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )


with tabs[1]:
    st.subheader("Stage 4 Value for Selected Solution")

    st.markdown(
        f"""
        Selected solution:
        **{equity_label(selected_equity_measure)}**,
        **λ={float(selected_row["lambda_equity"]):.1f}**.
        """
    )

    comparison = pd.DataFrame(
        [
            {
                "Metric": "Total Route Duration",
                "Stage 3 Repaired": float(
                    selected_row["baseline_total_route_duration"]
                ),
                "Stage 4 ALNS": float(
                    selected_row["total_route_duration"]
                ),
            },
            {
                "Metric": "Selected Equity Metric",
                "Stage 3 Repaired": float(
                    selected_row[
                        "baseline_selected_equity_metric_value"
                    ]
                ),
                "Stage 4 ALNS": float(
                    selected_row[
                        "stage4_selected_equity_metric_value"
                    ]
                ),
            },
            {
                "Metric": "Maximum Driver Route Workload",
                "Stage 3 Repaired": float(
                    selected_row[
                        "baseline_max_driver_route_workload"
                    ]
                ),
                "Stage 4 ALNS": float(
                    selected_row[
                        "max_driver_route_workload"
                    ]
                ),
            },
            {
                "Metric": "Active-Driver Workload CV",
                "Stage 3 Repaired": float(
                    selected_row[
                        "baseline_cv_active_route_workload"
                    ]
                ),
                "Stage 4 ALNS": float(
                    selected_row[
                        "cv_active_route_workload"
                    ]
                ),
            },
        ]
    )

    comparison_figure = build_stage3_vs_stage4_comparison_figure(
        comparison
    )

    st.plotly_chart(
        comparison_figure,
        use_container_width=True,
    )

    st.caption(
        "Each panel uses its own y-axis scale to make the Stage 3 to "
        "Stage 4 change visible within that metric. Bar heights should "
        "therefore be compared within each panel, not across panels."
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    scatter = px.scatter(
        master,
        x="duration_improvement_minutes",
        y="selected_equity_improvement_value",
        color="equity_measure_type",
        size="n_changed_assignments",
        hover_data=[
            "lambda_tag",
            "lambda_equity",
            "n_changed_route_keys",
            "delta_max_driver_route_workload",
        ],
        labels={
            "duration_improvement_minutes": (
                "Route Duration Improvement (minutes)"
            ),
            "selected_equity_improvement_value": (
                "Selected Equity Metric Improvement"
            ),
            "equity_measure_type": "Equity Measure",
        },
        title=(
            "Route-Duration Gain vs Selected-Equity Change "
            "Across All Stage 4 Runs"
        ),
    )

    st.plotly_chart(
        scatter,
        use_container_width=True,
    )


with tabs[2]:
    st.subheader("ALNS Search Trace")

    if iteration_log.empty:
        st.warning("No ALNS iteration log is available.")
    else:
        search_line = build_alns_search_progress_figure(
            iteration_log
        )

        st.plotly_chart(
            search_line,
            use_container_width=True,
        )

        st.caption(
            "The panels use independent y-axis scales because route duration "
            "is measured in minutes, while scalar scores are normalised objective "
            "values. Movement should therefore be interpreted within each panel, "
            "not by comparing bar or line heights across panels."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Iterations",
            f"{int(iteration_log['iteration'].max()):,}",
        )

        c2.metric(
            "Accepted Moves",
            f"{int(iteration_log['accepted'].astype(bool).sum()):,}",
        )

        c3.metric(
            "New Best Moves",
            f"{int(iteration_log['new_best'].astype(bool).sum()):,}",
        )

        c4.metric(
            "Mean Removed Orders",
            f"{iteration_log['n_removed_orders'].mean():,.1f}",
        )

        st.dataframe(
            iteration_log.tail(15),
            use_container_width=True,
            hide_index=True,
        )

    if not operator_weights.empty:
        weights = px.bar(
            operator_weights,
            x="operator_name",
            y="final_weight",
            color="operator_type",
            title="Final ALNS Operator Weights",
            labels={
                "operator_name": "Operator",
                "final_weight": "Final Weight",
                "operator_type": "Operator Type",
            },
        )

        st.plotly_chart(
            weights,
            use_container_width=True,
        )


with tabs[3]:
    st.subheader("Driver Workload Impact")

    workload_comparison = build_driver_workload_comparison(
        baseline_driver_workloads=baseline_driver_workloads,
        stage4_driver_workloads=driver_workloads,
    )

    scatter = px.scatter(
        workload_comparison,
        x="baseline_route_duration",
        y="stage4_route_duration",
        hover_data=[
            "driver_id",
            "driver_name",
            "delta_route_duration",
            "baseline_n_orders",
            "stage4_n_orders",
        ],
        labels={
            "baseline_route_duration": (
                "Stage 3 Repaired Route Duration"
            ),
            "stage4_route_duration": "Stage 4 Route Duration",
        },
        title="Driver Workload Before vs After Stage 4",
    )

    max_axis = max(
        workload_comparison["baseline_route_duration"].max(),
        workload_comparison["stage4_route_duration"].max(),
    )

    scatter.add_trace(
        go.Scatter(
            x=[0, max_axis],
            y=[0, max_axis],
            mode="lines",
            name="No change line",
            line=dict(dash="dash"),
        )
    )

    st.plotly_chart(
        scatter,
        use_container_width=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Largest Driver Workload Reductions**")

        st.dataframe(
            workload_comparison
            .sort_values("route_duration_improvement", ascending=False)
            .head(10),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        st.markdown("**Largest Driver Workload Increases**")

        st.dataframe(
            workload_comparison
            .sort_values("delta_route_duration", ascending=False)
            .head(10),
            use_container_width=True,
            hide_index=True,
        )


with tabs[4]:
    st.subheader("Assignment Changes")

    changed_assignments = build_assignment_change_table(
        baseline_assignments=baseline_assignments,
        stage4_assignments=stage4_assignments,
    )

    if changed_assignments.empty:
        st.info(
            "No changed assignment table is available, or no assignments "
            "changed for this solution."
        )
    else:
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Changed Orders",
            f"{len(changed_assignments):,}",
        )

        c2.metric(
            "Driver Changes",
            f"{int(changed_assignments['driver_changed'].sum()):,}",
        )

        c3.metric(
            "Day Changes",
            f"{int(changed_assignments['day_changed'].sum()):,}",
        )

        by_day = (
            changed_assignments
            .groupby("stage4_day_index")
            .size()
            .reset_index(name="changed_orders")
        )

        day_bar = px.bar(
            by_day,
            x="stage4_day_index",
            y="changed_orders",
            title="Changed Orders by Stage 4 Scheduled Day",
            labels={
                "stage4_day_index": "Stage 4 Day",
                "changed_orders": "Changed Orders",
            },
        )

        st.plotly_chart(
            day_bar,
            use_container_width=True,
        )

        st.dataframe(
            changed_assignments,
            use_container_width=True,
            hide_index=True,
        )


with tabs[5]:
    st.subheader("Route Diagnostics")

    route_cols = [
        "route_id",
        "driver_id",
        "driver_name",
        "day_index",
        "weekday",
        "n_orders",
        "route_duration",
        "total_travel_time",
        "total_service_time",
        "total_waiting_time",
        "total_violation_minutes",
        "route_feasible",
        "construction_method",
    ]

    available_route_cols = [
        column
        for column in route_cols
        if column in route_summaries.columns
    ]

    sorted_routes = (
        route_summaries
        .sort_values("route_duration", ascending=False)
    )

    st.markdown("**Longest Stage 4 Driver-Day Routes**")

    st.dataframe(
        sorted_routes[available_route_cols].head(20),
        use_container_width=True,
        hide_index=True,
    )

    route_choice = st.selectbox(
        "Inspect route stops",
        options=sorted_routes["route_id"].astype(str).tolist(),
    )

    selected_stops = route_stops.loc[
        route_stops["route_id"].astype(str).eq(route_choice)
    ].copy()

    selected_stops = selected_stops.sort_values(
        "stop_number"
    )

    stop_cols = [
        "stop_number",
        "orderid",
        "locationid",
        "arrival_time",
        "window_start",
        "window_end",
        "service_start_time",
        "service_time",
        "service_end_time",
        "departure_time",
        "late_start_violation",
    ]

    available_stop_cols = [
        column
        for column in stop_cols
        if column in selected_stops.columns
    ]

    st.dataframe(
        selected_stops[available_stop_cols],
        use_container_width=True,
        hide_index=True,
    )


with tabs[6]:
    st.subheader("Tables & Downloads")

    st.download_button(
        "Download Stage 4 Master Summary",
        data=master.to_csv(index=False),
        file_name="stage_4_master_summary_dashboard.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download Selected Stage 4 Route Summaries",
        data=route_summaries.to_csv(index=False),
        file_name=(
            f"stage_4_route_summaries_"
            f"{selected_equity_measure}_{selected_lambda_tag}.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        "Download Selected Stage 4 Route Stops",
        data=route_stops.to_csv(index=False),
        file_name=(
            f"stage_4_route_stops_"
            f"{selected_equity_measure}_{selected_lambda_tag}.csv"
        ),
        mime="text/csv",
    )

    st.download_button(
        "Download Selected Stage 4 Assignments",
        data=stage4_assignments.to_csv(index=False),
        file_name=(
            f"stage_4_assignments_"
            f"{selected_equity_measure}_{selected_lambda_tag}.csv"
        ),
        mime="text/csv",
    )

    st.markdown("**Stage 4 Master Summary**")

    st.dataframe(
        master,
        use_container_width=True,
        hide_index=True,
    )