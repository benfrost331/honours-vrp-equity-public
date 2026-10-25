import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    DATA_PROCESSED_DIR,
    EQUITY_MEASURE_LABELS,
    EQUITY_MEASURE_TYPES,
)

from src.dashboard_data import (
    load_equity_assignments,
    load_equity_day_workloads,
    load_equity_driver_workloads,
    load_equity_lambda_summary,
)

from pathlib import Path

DARK_BLUE = "#1E3A8A"

px.defaults.color_discrete_sequence = [DARK_BLUE]


# PAGE CONFIGURATION


st.set_page_config(
    page_title="Stage 2 Assignment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# DISPLAY SETTINGS


DISPLAY_LAMBDAS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]



# SIMPLE MODERN STYLING


st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #475569;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
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

        .method-note {
            background-color: #eff6ff;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 14px 18px;
            color: #1e3a8a;
            margin-top: 1rem;
        }

        .solution-status-banner {
        background-color: #dff3e6;
        color: #11812c;
        text-align: center;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)



# HELPER FUNCTIONS


def convert_to_boolean(series: pd.Series) -> pd.Series:
    """
    Handle either real booleans or string booleans read from CSV.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def format_minutes(value: float) -> str:
    """
    Format a minute value with a thousands separator.
    """

    return f"{value:,.0f} min"


def format_fulfilment_date(date_value: pd.Timestamp) -> str:
    """
    Format a planning-horizon date without a leading zero.
    Example: 17 January 2025.
    """

    return f"{date_value.day} {date_value.strftime('%B %Y')}"


@st.cache_data(show_spinner=False)
def load_stage2_planning_day_labels(
    orders_path_string: str,
    modified_time_ns: int | None,
) -> dict[int, str]:
    """
    Build the displayed date label for each planning day.

    Stage 2 uses day_index values internally. The dashboard displays the
    corresponding working date so that the Assignment Details filter is
    easier to interpret.
    """

    del modified_time_ns

    orders_path = Path(orders_path_string)

    if orders_path.exists():
        orders = pd.read_csv(
            orders_path,
            usecols=["datefrom"],
        )

        start_date = pd.to_datetime(
            orders["datefrom"].dropna().min()
        )

    else:
        # Safe fallback for deployment environments where only result
        # outputs are available. This matches the current project data.
        start_date = pd.Timestamp("2025-01-07")

    working_dates = pd.bdate_range(
        start=start_date,
        periods=16,
    )

    return {
        index + 1: format_fulfilment_date(date_value)
        for index, date_value in enumerate(working_dates)
    }


def get_stage2_planning_day_labels() -> dict[int, str]:
    """
    Wrapper that includes the processed-orders modified time in the cache
    key when the file is available.
    """

    orders_path = DATA_PROCESSED_DIR / "orders_clean.csv"

    modified_time_ns = (
        orders_path.stat().st_mtime_ns
        if orders_path.exists()
        else None
    )

    return load_stage2_planning_day_labels(
        orders_path_string=str(orders_path),
        modified_time_ns=modified_time_ns,
    )



# HEADER


st.title("Stage 2 Efficiency-Equity Assignment Dashboard")

st.markdown(
    """
    <div class="dashboard-subtitle">
        Explore How Increasing Workload-Equity Emphasis Changes
        Estimated Assignment Efficiency & Driver Workload Allocation
    </div>
    """,
    unsafe_allow_html=True,
)


if "stage2_equity_measure_type" not in st.session_state:
    st.session_state["stage2_equity_measure_type"] = (
        EQUITY_MEASURE_TYPES[0]
    )

selected_equity_measure = st.session_state[
    "stage2_equity_measure_type"
]

selected_equity_label = EQUITY_MEASURE_LABELS[
    selected_equity_measure
]


# LOAD AND VALIDATE THE SUMMARY DATA



try:
    summary = load_equity_lambda_summary(selected_equity_measure)

except (FileNotFoundError, pd.errors.EmptyDataError) as error:

    st.error(str(error))
    st.stop()


summary["lambda_equity"] = (
    summary["lambda_equity"].astype(float)
)

summary = summary[
    summary["lambda_equity"]
    .round(3)
    .isin(DISPLAY_LAMBDAS)
].copy()


if "solution_valid" in summary.columns:

    summary["solution_valid_bool"] = (
        convert_to_boolean(summary["solution_valid"])
    )

    invalid_solutions = summary[
        ~summary["solution_valid_bool"]
    ]

    if not invalid_solutions.empty:

        invalid_values = (
            invalid_solutions["lambda_equity"]
            .tolist()
        )

        st.error(
            "The following stored lambda solutions failed validation: "
            f"{invalid_values}"
        )

        st.stop()


available_lambdas = (
    summary["lambda_equity"]
    .sort_values()
    .round(3)
    .tolist()
)


missing_lambdas = sorted(
    set(DISPLAY_LAMBDAS)
    - set(available_lambdas)
)


if missing_lambdas:

    st.warning(
        "Some requested solutions are missing and will not appear "
        f"in the slider: {missing_lambdas}"
    )


if not available_lambdas:

    st.error(
        "None of the four requested lambda solutions are available."
    )

    st.stop()



# CALCULATE COMPARATIVE METRICS IF NEEDED


baseline_row = (
    summary
    .sort_values("lambda_equity")
    .iloc[0]
)

baseline_cost = float(
    baseline_row["total_estimated_assignment_time"]
)

baseline_max_workload = float(
    baseline_row["max_driver_workload"]
)

baseline_drivers_used = int(
    baseline_row["n_drivers_used"]
)

baseline_workload_cv = float(
    baseline_row.get("workload_cv_active", 0.0)
)


if "efficiency_cost_increase_pct" not in summary.columns:

    summary["efficiency_cost_increase_pct"] = (
        100
        * (
            summary["total_estimated_assignment_time"]
            - baseline_cost
        )
        / baseline_cost
    )


if "max_workload_reduction_pct" not in summary.columns:

    summary["max_workload_reduction_pct"] = (
        100
        * (
            baseline_max_workload
            - summary["max_driver_workload"]
        )
        / baseline_max_workload
    )



# SELECTED LAMBDA STATE


default_lambda = (
    0.0
    if 0.0 in available_lambdas
    else available_lambdas[0]
)

state_lambda = st.session_state.get(
    "stage2_selected_lambda",
    default_lambda,
)

available_lambda_keys = [
    round(float(value), 3)
    for value in available_lambdas
]

if round(float(state_lambda), 3) not in available_lambda_keys:
    st.session_state["stage2_selected_lambda"] = default_lambda
else:
    st.session_state["stage2_selected_lambda"] = float(state_lambda)

selected_lambda = float(
    st.session_state["stage2_selected_lambda"]
)


selected_row = (
    summary[
        summary["lambda_equity"].round(3)
        == round(selected_lambda, 3)
    ]
    .iloc[0]
)


selected_tag = str(
    selected_row["lambda_tag"]
)



# LOAD THE SELECTED SOLUTION


try:
    assignments = load_equity_assignments(
        selected_equity_measure,
        selected_tag,
    )

    driver_workloads = load_equity_driver_workloads(
        selected_equity_measure,
        selected_tag,
    )

    day_workloads = load_equity_day_workloads(
        selected_equity_measure,
        selected_tag,
    )

except FileNotFoundError as error:

    st.error(str(error))
    st.stop()



# SOLUTION STATUS


gap_target = float(
    selected_row.get("relative_gap_target", 0.0)
)

status = str(
    selected_row.get("status", "Unknown")
)


st.markdown(
    f"""
    <div class="solution-status-banner">
        Displaying Stored Solution for λ = {selected_lambda:.1f}
        &nbsp;|&nbsp;
        Solver Status: {status}
        &nbsp;|&nbsp;
        Configured Relative-Gap Target: {100 * gap_target:.1f}%
    </div>
    """,
    unsafe_allow_html=True,
)



# TOP-LEVEL METRICS


total_time = float(
    selected_row["total_estimated_assignment_time"]
)

maximum_workload = float(
    selected_row["max_driver_workload"]
)

cost_increase = float(
    selected_row["efficiency_cost_increase_pct"]
)

workload_reduction = float(
    selected_row["max_workload_reduction_pct"]
)

drivers_used = int(
    selected_row["n_drivers_used"]
)

drivers_available = int(
    selected_row["n_drivers_available"]
)

workload_cv = float(
    selected_row.get("workload_cv_active", 0.0)
)

# Percentage change in the number of active drivers.
if baseline_drivers_used > 0:
    drivers_used_change_pct = (
        100
        * (drivers_used - baseline_drivers_used)
        / baseline_drivers_used
    )
else:
    drivers_used_change_pct = 0.0


# Reduction in CV relative to the efficiency-only baseline.
# A positive value means workload allocation has become more equal.
if baseline_workload_cv > 0:
    workload_cv_reduction_pct = (
        100
        * (baseline_workload_cv - workload_cv)
        / baseline_workload_cv
    )
else:
    workload_cv_reduction_pct = 0.0

metric_1, metric_2, metric_3, metric_4 = st.columns(4)


metric_1.metric(
    label="Total Estimated Time",
    value=format_minutes(total_time),
    delta=f"{cost_increase:.2f}% vs. Baseline",
    delta_color="inverse",
)


baseline_equity_metric = float(
    baseline_row.get(
        "equity_penalty_metric_value",
        baseline_row["max_driver_workload"],
    )
)

selected_equity_metric = float(
    selected_row.get(
        "equity_penalty_metric_value",
        selected_row["max_driver_workload"],
    )
)

if baseline_equity_metric > 0:
    equity_metric_reduction_pct = (
        100
        * (
            baseline_equity_metric
            - selected_equity_metric
        )
        / baseline_equity_metric
    )
else:
    equity_metric_reduction_pct = 0.0


metric_2.metric(
    label="Selected Equity Penalty",
    value=f"{selected_equity_metric:,.2f}",
    delta=f"{equity_metric_reduction_pct:.2f}% Reduction vs. Baseline",
)


metric_3.metric(
    label="Number of Drivers Used",
    value=f"{drivers_used} / {drivers_available}",
    delta=f"{drivers_used_change_pct:+.2f}% vs. Baseline",
    delta_color="off",
)


metric_4.metric(
    label="Active-Driver Workload Coefficient of Variation",
    value=f"{workload_cv:.3f}",
    delta=f"{workload_cv_reduction_pct:.2f}% Reduction vs. Baseline",
    delta_color="normal",
)


# SOLUTION CONTROLS


st.markdown("### Solution Controls")

control_1, control_2 = st.columns([1, 2])

with control_1:

    st.selectbox(
        "Equity Measure Type",
        options=EQUITY_MEASURE_TYPES,
        format_func=lambda value: EQUITY_MEASURE_LABELS[value],
        key="stage2_equity_measure_type",
    )

with control_2:

    st.select_slider(
        "Equity Emphasis",
        options=available_lambdas,
        key="stage2_selected_lambda",
        format_func=lambda value: f"λ = {value:.1f}",
        help=(
            "λ = 0 gives efficiency-only emphasis. "
            "Increasing λ places greater normalised emphasis on "
            "the selected equity measure."
        ),
    )


# DASHBOARD TABS


tradeoff_tab, workload_tab, assignments_tab = st.tabs(
    [
        "Efficiency–Equity Trade-Off",
        "Workload Allocation",
        "Assignment Details",
    ]
)


# ===================================================================
# TAB 1: TRADE-OFF
# ===================================================================

with tradeoff_tab:

    st.subheader("Assignment Efficiency–Equity Frontier")

    frontier_data = (
        summary
        .sort_values("lambda_equity")
        .copy()
    )

    frontier_data["lambda_label"] = (
        frontier_data["lambda_equity"]
        .map(lambda value: f"λ={value:.1f}")
    )

    frontier_figure = px.line(
        frontier_data,
        x="total_estimated_assignment_time",
        y="equity_penalty_metric_value",
        markers=True,
        text="lambda_label",
        custom_data=[
            "lambda_equity",
            "efficiency_cost_increase_pct",
            "max_workload_reduction_pct",
            "n_drivers_used",
            "status",
        ],
        labels={
            "total_estimated_assignment_time":
                "Total Estimated Assignment Time (minutes)",
            "equity_penalty_metric_value":
                selected_equity_label,
        },
    )

    frontier_figure.update_traces(
        textposition="top center",
        line={"width": 3},
        marker={"size": 10},
        hovertemplate=(
            "<b>λ=%{customdata[0]:.1f}</b><br>"
            "Total estimated time: %{x:,.0f} min<br>"
            f"{selected_equity_label}: %{{y:,.2f}}<br>"
            "Cost increase: %{customdata[1]:.2f}%<br>"
            "Max-workload reduction: %{customdata[2]:.2f}%<br>"
            "Drivers used: %{customdata[3]}<br>"
            "Status: %{customdata[4]}"
            "<extra></extra>"
        ),
    )

    selected_equity_metric = float(
        selected_row["equity_penalty_metric_value"]
    )

    frontier_figure.add_scatter(
        x=[total_time],
        y=[selected_equity_metric],
        mode="markers",
        marker={
            "size": 18,
            "symbol": "diamond",
        },
        name=f"Selected λ={selected_lambda:.1f}",
        hovertemplate=(
            f"<b>Selected λ={selected_lambda:.1f}</b>"
            "<extra></extra>"
        ),
    )

    frontier_figure.update_layout(
        height=520,
        legend_title_text="",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )

    st.plotly_chart(
        frontier_figure,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )

    st.subheader("Stored Solution Comparison")

    preferred_columns = [
        "lambda_equity",
        "status",
        "total_estimated_assignment_time",
        "efficiency_cost_increase_pct",
        "equity_penalty_metric_value",
        "max_driver_workload",
        "workload_range",
        "workload_mad_mean",
        "workload_minmax_deviation",
        "workload_gini_coefficient",
        "workload_cv_active",
        "n_drivers_used",
        "solve_runtime_seconds",
    ]

    comparison_columns = [
        column
        for column in preferred_columns
        if column in summary.columns
    ]

    pretty_column_names = {
        "lambda_equity": "λ",
        "status": "Solver Status",
        "total_estimated_assignment_time": "Total Estimated Time (min)",
        "efficiency_cost_increase_pct": "Cost Increase vs Baseline (%)",
        "equity_penalty_metric_value": selected_equity_label,
        "max_driver_workload": "Max Driver Workload (min)",
        "max_workload_reduction_pct": "Max Workload Reduction (%)",
        "workload_std_active": "Active-Driver Workload SD",
        "workload_cv_active": "Active-Driver Workload CV",
        "n_drivers_used": "Drivers Used",
        "solve_runtime_seconds": "Solve Time (sec)",
    }

    comparison_table = (
        summary[comparison_columns]
        .rename(columns=pretty_column_names)
        .copy()
    )

    st.dataframe(
    comparison_table,
    hide_index=True,
    width="stretch",
    column_config={
        "Equity Weight (λ)": st.column_config.NumberColumn(
            format="%.1f"
        ),
        "Total Estimated Time (min)": st.column_config.NumberColumn(
            format="%.0f"
        ),
        "Cost Increase vs Baseline (%)": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Maximum Driver Workload (min)": st.column_config.NumberColumn(
            format="%,.0f"
        ),
        "Maximum Workload Reduction (%)": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "Active-Driver Workload SD": st.column_config.NumberColumn(
            format="%.1f"
        ),
        "Active-Driver Workload CV": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "Drivers Used": st.column_config.NumberColumn(
            format="%d"
        ),
        "Solve Time (sec)": st.column_config.NumberColumn(
            format="%,.1f"
        ),
    },
)

# ===================================================================
# TAB 2: WORKLOAD ALLOCATION
# ===================================================================

with workload_tab:

    driver_plot_data = driver_workloads.copy()

    st.caption(
        "All drivers are included, including drivers receiving no "
        "assignments in the selected solution."
    )


    driver_plot_data = (
        driver_plot_data
        .sort_values(
            "total_estimated_assignment_time",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    driver_plot_data["driver_label"] = (
        "Driver "
        + driver_plot_data["driver_id"]
        .astype(int)
        .astype(str)
    )


    st.subheader("Total Estimated Workload per Driver")

    driver_figure = px.bar(
        driver_plot_data,
        x="driver_label",
        y="total_estimated_assignment_time",
        hover_data={
            "driver_name": True,
            "n_orders": True,
            "total_job_time": ":.1f",
            "total_outbound_time": ":.1f",
            "total_return_time": ":.1f",
            "horizon_utilisation_pct": ":.1f",
        },
        labels={
            "driver_label": "Driver",
            "total_estimated_assignment_time":
                "Estimated Workload (minutes)",
        },
    )

    if not driver_plot_data.empty:

        mean_workload = float(
            driver_plot_data[
                "total_estimated_assignment_time"
            ].mean()
        )

        driver_figure.add_hline(
            y=mean_workload,
            line_dash="dash",
            annotation_text="All-Driver Mean",
            annotation_position="top right",
        )

    driver_figure.update_layout(
        height=560,
        showlegend=False,
        xaxis_tickangle=-60,
        margin={"l": 20, "r": 20, "t": 40, "b": 80},
    )

    st.plotly_chart(
        driver_figure,
        width="stretch",
        config={"displaylogo": False},
    )


    st.subheader("Total Estimated Workload by Planning Day")

    day_plot_data = (
        day_workloads
        .sort_values("day_index")
        .copy()
    )

    day_plot_data["day_label"] = (
        "Day "
        + day_plot_data["day_index"]
        .astype(int)
        .astype(str)
        + " — "
        + day_plot_data["weekday"].astype(str)
    )

    day_figure = px.bar(
        day_plot_data,
        x="day_label",
        y="total_estimated_assignment_time",
        hover_data={
            "n_orders": True,
            "n_active_drivers": True,
            "total_job_time": ":.1f",
            "total_travel_time": ":.1f",
        },
        labels={
            "day_label": "Planning Day",
            "total_estimated_assignment_time":
                "Estimated Workload (mins)",
        },
    )

    day_figure.update_layout(
        height=470,
        showlegend=False,
        xaxis_tickangle=-35,
        margin={"l": 20, "r": 20, "t": 40, "b": 70},
    )

    st.plotly_chart(
        day_figure,
        width="stretch",
        config={"displaylogo": False},
    )


    with st.expander("View Driver Workload Table"):

        driver_workload_column_names = {
            "driver_id": "Driver ID",
            "driver_name": "Driver Name",
            "shift_length": "Daily Shift Length (min)",
            "n_orders": "Orders Assigned",
            "total_job_time": "Total Job Time (min)",
            "total_outbound_time": "Total Outbound Time (min)",
            "total_return_time": "Total Return Time (min)",
            "total_estimated_assignment_time": "Total Estimated Workload (min)",
            "is_active": "Active Driver",
            "horizon_capacity": "Planning-Horizon Capacity (min)",
            "horizon_utilisation_pct": "Horizon Utilisation (%)",
        }

        driver_workload_table = (
            driver_workloads
            .rename(columns=driver_workload_column_names)
            .copy()
        )

        st.dataframe(
            driver_workload_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Driver ID": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Driver Name": st.column_config.TextColumn(
                    width="medium",
                ),
                "Daily Shift Length (min)": st.column_config.NumberColumn(
                    format="%,.0f",
                ),
                "Orders Assigned": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Total Job Time (min)": st.column_config.NumberColumn(
                    format="%,.1f",
                ),
                "Total Outbound Time (min)": st.column_config.NumberColumn(
                    format="%,.1f",
                ),
                "Total Return Time (min)": st.column_config.NumberColumn(
                    format="%,.1f",
                ),
                "Total Estimated Workload (min)": st.column_config.NumberColumn(
                    format="%,.1f",
                ),
                "Active Driver": st.column_config.CheckboxColumn(),
                "Planning-Horizon Capacity (min)": st.column_config.NumberColumn(
                    format="%,.0f",
                ),
                "Horizon Utilisation (%)": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )


# ===================================================================
# TAB 3: ASSIGNMENT DETAILS
# ===================================================================

with assignments_tab:

    st.subheader(
        f"Order Assignments for λ = {selected_lambda:.1f}"
    )

    filter_1, filter_2 = st.columns(2)


    driver_options = sorted(
        assignments["driver_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    with filter_1:

        selected_drivers = st.multiselect(
            "Filter by Driver",
            options=driver_options,
        )


    day_options = sorted(
        assignments["day_index"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    planning_day_labels = get_stage2_planning_day_labels()

    with filter_2:

        selected_days = st.multiselect(
            "Filter by Planning Day",
            options=day_options,
            format_func=lambda day_index: planning_day_labels.get(
                int(day_index),
                f"Day {int(day_index)}",
            ),
        )


    displayed_assignments = assignments.copy()


    if selected_drivers:

        displayed_assignments = (
            displayed_assignments[
                displayed_assignments[
                    "driver_name"
                ].astype(str)
                .isin(selected_drivers)
            ]
        )


    if selected_days:

        displayed_assignments = (
            displayed_assignments[
                displayed_assignments[
                    "day_index"
                ].astype(int)
                .isin(selected_days)
            ]
        )


    displayed_assignments = (
        displayed_assignments
        .sort_values(
            ["day_index", "driver_id", "orderid"]
        )
    )


    st.caption(
        f"Displaying {len(displayed_assignments):,} "
        f"of {len(assignments):,} Assignments"
    )


    
    # PREPARE ORDER ASSIGNMENTS TABLE FOR DISPLAY
    

    assignment_column_names = {
        "lambda_equity": "Equity Weight (λ)",
        "orderid": "Order ID",
        "day_index": "Planning Day",
        "weekday": "Weekday",
        "driver_id": "Driver ID",
        "driver_name": "Driver Name",
        "order_locationid": "Order Location ID",
        "driver_locationid": "Driver Base Location ID",
        "jobtime": "Job Time (min)",
        "outbound_time": "Outbound Travel Time (min)",
        "return_time": "Return Travel Time (min)",
        "estimated_assignment_time": "Estimated Assignment Time (min)",
    }

    assignment_table = (
        displayed_assignments
        # Remove the internal solution identifier from the displayed table.
        .drop(columns=["lambda_tag"], errors="ignore")
        .rename(columns=assignment_column_names)
        .copy()
    )

    st.dataframe(
        assignment_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Equity Weight (λ)": st.column_config.NumberColumn(
                format="%.1f",
            ),
            "Order ID": st.column_config.NumberColumn(
                format="%d",
            ),
            "Planning Day": st.column_config.NumberColumn(
                format="%d",
            ),
            "Weekday": st.column_config.TextColumn(
                width="small",
            ),
            "Driver ID": st.column_config.NumberColumn(
                format="%d",
            ),
            "Driver Name": st.column_config.TextColumn(
                width="medium",
            ),
            "Order Location ID": st.column_config.NumberColumn(
                format="%d",
            ),
            "Driver Base Location ID": st.column_config.NumberColumn(
                format="%d",
            ),
            "Job Time (min)": st.column_config.NumberColumn(
                format="%,.1f",
            ),
            "Outbound Travel Time (min)": st.column_config.NumberColumn(
                format="%,.1f",
            ),
            "Return Travel Time (min)": st.column_config.NumberColumn(
                format="%,.1f",
            ),
            "Estimated Assignment Time (min)": st.column_config.NumberColumn(
                format="%,.1f",
            ),
        },
    )


    csv_data = (
    assignment_table
    .to_csv(index=False)
    .encode("utf-8")
    )


    st.download_button(
        label="Download Displayed Assignments",
        data=csv_data,
        file_name=(
            f"stage_2_assignments_lambda_"
            f"{selected_lambda:.1f}.csv"
        ),
        mime="text/csv",
    )



# FOOTER


st.divider()

st.caption(
    "These are Stage 2 order–day–driver assignments. Estimated "
    "workload treats each order as an individual base-to-order-to-base "
    "activity and does not yet represent sequenced multi-order daily "
    "routes."
)