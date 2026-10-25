from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from src.stage_3_dashboard_data import (
    load_stage3_locations,
    load_stage3_master_summary,
    load_stage3_repair_comparison,
    load_stage3_solution_files,
)

from src.config import (
    DATA_PROCESSED_DIR,
    EQUITY_MEASURE_LABELS,
    EQUITY_MEASURE_TYPES,
)

from src.equity_measures import (
    compute_workload_equity_metrics,
    equity_metric_value_for_measure,
)

DARK_BLUE = "#1E3A8A"
GREEN = "#15803D"
RED = "#DC2626"
AMBER = "#B45309"
BLUE = "#2563EB"
LIGHT_GREY = "#CBD5E1"
ORDER_PENDING_RED = "#FF2D2D"
ORDER_SERVICE_BLUE = "#00A3FF"
ORDER_COMPLETED_GREEN = "#00C853"

ROUTE_ANIMATION_STEP_MINUTES = 2

PLAYBACK_SPEED_TO_FRAME_DURATION_MS = {
    "1x": 500,
    "2x": 250,
    "5x": 100,
    "10x": 50,
}

DRIVER_POSITION_ICON = "🚚"
DRIVER_ICON_FONT_SIZE = 26
DRIVER_SERVICE_OFFSET_FRACTION = 0.025

OVERALL_ANIMATION_STEP_MINUTES = 5
OVERALL_ORDER_DOT_SIZE = 8
OVERALL_DRIVER_DOT_SIZE = 9
OVERALL_BASE_MARKER_SIZE = 8

OVERALL_DRIVER_BLUE = "#0066FF"
OVERALL_BASE_BLUE = "#1E3A8A"

px.defaults.color_discrete_sequence = [DARK_BLUE]


st.set_page_config(
    page_title="Stage 3 Routing Dashboard",
    page_icon="🛣️",
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

        h1, h2, h3 { color: #0f172a; }

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

def format_fulfilment_date(date_value: pd.Timestamp) -> str:
    """
    Format a planning-horizon date without a leading zero.
    Example: 17 January 2025.
    """

    return f"{date_value.day} {date_value.strftime('%B %Y')}"

@st.cache_data(show_spinner=False)
def load_stage3_planning_day_labels(
    orders_path_string: str,
    modified_time_ns: int | None,
) -> dict[int, str]:
    """
    Build displayed date labels for the Stage 3 planning days.

    The optimisation uses numeric day_index values internally. The dashboard
    displays the corresponding fulfilment date so that route and day filters
    are easier to interpret.
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

        # Deployment-safe fallback matching the current project data.
        start_date = pd.Timestamp("2025-01-07")

    working_dates = pd.bdate_range(
        start=start_date,
        periods=16,
    )

    return {
        index + 1: format_fulfilment_date(date_value)
        for index, date_value in enumerate(working_dates)
    }

def get_stage3_planning_day_labels() -> dict[int, str]:
    """
    Return planning-day labels, invalidating the cache if orders_clean.csv
    changes.
    """

    orders_path = DATA_PROCESSED_DIR / "orders_clean.csv"

    modified_time_ns = (
        orders_path.stat().st_mtime_ns
        if orders_path.exists()
        else None
    )

    return load_stage3_planning_day_labels(
        orders_path_string=str(orders_path),
        modified_time_ns=modified_time_ns,
    )

def planning_day_label(
    day_index: int,
    planning_day_labels: dict[int, str],
) -> str:
    return planning_day_labels.get(
        int(day_index),
        f"Day {int(day_index)}",
    )

def selected_stage3_row_for_lambda(
    summary: pd.DataFrame,
    selected_lambda: float,
) -> pd.Series:
    """
    Select the stored Stage 3 solution row for the selected lambda.

    The current equity-measure outputs are full-instance outputs. This helper
    still keeps a stable fallback if more than one model_scope appears.
    """

    matching_rows = summary.loc[
        np.isclose(
            summary["lambda_equity"].astype(float),
            float(selected_lambda),
        )
    ].copy()

    if matching_rows.empty:
        raise ValueError(
            f"No Stage 3 solution found for λ={selected_lambda:.1f}."
        )

    full_rows = matching_rows.loc[
        matching_rows["model_scope"].astype(str).eq("full_instance")
    ]

    if not full_rows.empty:
        matching_rows = full_rows

    return (
        matching_rows
        .sort_values(["model_scope", "lambda_tag"])
        .iloc[0]
    )

def route_workload_selected_equity_value(
    driver_workloads: pd.DataFrame,
    equity_measure_type: str,
) -> float:
    """
    Compute the selected equity metric using routed driver workloads.

    This is dashboard-side only. It allows the Stage 3 frontier y-axis to
    follow the selected equity measure rather than always showing maximum
    routed workload.
    """

    workloads = pd.Series(
        driver_workloads["total_route_duration"].astype(float).to_numpy()
    )

    metrics = compute_workload_equity_metrics(
        workloads
    )

    return equity_metric_value_for_measure(
        metrics=metrics,
        equity_measure_type=equity_measure_type,
    )

def safe_percentage_change(
    current_value: float,
    baseline_value: float,
) -> float:
    """
    Return percentage change from baseline to current.

    Positive means the current value is higher than the baseline.
    """
    if pd.isna(current_value) or pd.isna(baseline_value):
        return 0.0

    baseline_value = float(baseline_value)

    if abs(baseline_value) < 1e-9:
        return 0.0

    return (
        100.0
        * (float(current_value) - baseline_value)
        / baseline_value
    )


def safe_percentage_reduction(
    current_value: float,
    baseline_value: float,
) -> float:
    """
    Return percentage reduction from baseline to current.

    Positive means the current value is lower/better than the baseline.
    """
    if pd.isna(current_value) or pd.isna(baseline_value):
        return 0.0

    baseline_value = float(baseline_value)

    if abs(baseline_value) < 1e-9:
        return 0.0

    return (
        100.0
        * (baseline_value - float(current_value))
        / baseline_value
    )


def format_equity_penalty_metric(
    value: float,
) -> str:
    """
    Match the Stage 2 selected-equity card format.
    """
    if pd.isna(value):
        return "n/a"

    return f"{float(value):,.2f}"

def add_selected_route_equity_metric(
    summary: pd.DataFrame,
    use_repaired: bool,
    equity_measure_type: str,
) -> pd.DataFrame:
    """
    Add selected_route_equity_metric_value to the Stage 3 frontier table.
    """

    result = summary.copy()
    selected_values = []

    for _, row in result.iterrows():

        try:
            solution_files = load_stage3_solution_files(
                model_scope=str(row["model_scope"]),
                lambda_tag=str(row["lambda_tag"]),
                use_repaired=use_repaired,
                equity_measure_type=equity_measure_type,
            )

            value = route_workload_selected_equity_value(
                driver_workloads=solution_files["driver_route_workloads"],
                equity_measure_type=equity_measure_type,
            )

        except Exception:
            value = np.nan

        selected_values.append(value)

    result["selected_route_equity_metric_value"] = selected_values

    return result

def format_time(minutes: float) -> str:
    if pd.isna(minutes):
        return ""

    total_minutes = int(round(float(minutes)))
    hours = total_minutes // 60
    mins = total_minutes % 60

    return f"{hours:02d}:{mins:02d}"

def bool_value(value: object) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )

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

def build_solution_options(
    summary: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
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
    row = summary.loc[
        summary["model_scope"].astype(str).eq(model_scope)
        & summary["lambda_tag"].astype(str).eq(lambda_tag)
    ]

    if row.empty:
        raise ValueError(
            f"No summary row found for {model_scope} {lambda_tag}."
        )

    return row.iloc[0]

def route_options(
    route_summaries: pd.DataFrame,
    planning_day_labels: dict[int, str] | None = None,
) -> dict[str, str]:
    routes = route_summaries.sort_values(
        ["day_index", "driver_id", "route_id"]
    )

    if planning_day_labels is None:
        planning_day_labels = {}

    return {
        (
            f"{planning_day_label(int(row.day_index), planning_day_labels)} "
            f"({row.weekday}) · "
            f"{row.driver_name} · "
            f"{int(row.n_orders)} Orders · "
            f"{row.route_duration:.1f} min"
        ): str(row.route_id)
        for row in routes.itertuples()
    }

def route_coordinates(
    route_id: str,
    route_summaries: pd.DataFrame,
    route_stops: pd.DataFrame,
    locations: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    route_row = route_summaries.loc[
        route_summaries["route_id"].astype(str).eq(route_id)
    ].iloc[0]

    stops = route_stops.loc[
        route_stops["route_id"].astype(str).eq(route_id)
    ].copy()

    stops = stops.sort_values("stop_number")

    stops = stops.merge(
        locations["orders"][["orderid", "x", "y"]],
        on="orderid",
        how="left",
    )

    driver_row = locations["drivers"].loc[
        locations["drivers"]["driver_id"]
        .astype(int)
        .eq(int(route_row["driver_id"]))
    ].iloc[0]

    return route_row, stops, driver_row

def build_overall_day_animation_times(
    day_routes: pd.DataFrame,
    day_stops: pd.DataFrame,
) -> list[float]:
    """
    Build animation times for a full-day overview.

    Regular time steps keep the driver movement smooth enough, while exact
    event times ensure orders turn green immediately when completed.
    """

    start_time = float(
        day_routes["shift_start"].min()
    )

    end_time = float(
        day_routes["return_arrival_time"].max()
    )

    regular_times = list(
        np.arange(
            start_time,
            end_time + OVERALL_ANIMATION_STEP_MINUTES,
            OVERALL_ANIMATION_STEP_MINUTES,
        )
    )

    event_columns = [
        "arrival_time",
        "service_end_time",
        "departure_time",
        "return_arrival_time",
    ]

    event_times = []

    for column in event_columns:

        if column in day_stops.columns:

            event_times.extend(
                day_stops[column]
                .dropna()
                .astype(float)
                .tolist()
            )

        elif column in day_routes.columns:

            event_times.extend(
                day_routes[column]
                .dropna()
                .astype(float)
                .tolist()
            )

    if "departure_time" in day_routes.columns:
        event_times.extend(
            day_routes["departure_time"]
            .dropna()
            .astype(float)
            .tolist()
        )

    all_times = (
        regular_times
        + event_times
        + [
            start_time,
            end_time,
        ]
    )

    cleaned_times = sorted(
        {
            round(float(value), 6)
            for value in all_times
            if start_time <= float(value) <= end_time
        }
    )

    if not cleaned_times:
        cleaned_times = [start_time]

    return cleaned_times

def overall_order_colour_at_time(
    stop: pd.Series,
    current_time: float,
) -> str:
    """
    Overall animation order state.

    Orders start red and turn green once service is completed.
    """

    service_end_time = float(
        stop.service_end_time
    )

    if current_time >= service_end_time:
        return ORDER_COMPLETED_GREEN

    return ORDER_PENDING_RED

def overall_driver_position_at_time(
    current_time: float,
    route_row: pd.Series,
    stops_for_route: pd.DataFrame,
    driver_row: pd.Series,
) -> tuple[float, float, str]:
    """
    Return a driver's position for the overall day animation.

    Before the route starts and after it finishes, the driver is shown at
    the base. During the route, the existing route interpolation logic is
    reused.
    """

    base_x = float(
        driver_row["x"]
    )

    base_y = float(
        driver_row["y"]
    )

    shift_start = float(
        route_row["shift_start"]
    )

    route_departure_time = float(
        route_row.get(
            "departure_time",
            shift_start,
        )
    )

    return_arrival_time = float(
        route_row["return_arrival_time"]
    )

    if current_time < route_departure_time:

        return (
            base_x,
            base_y,
            "Waiting at base",
        )

    if current_time > return_arrival_time:

        return (
            base_x,
            base_y,
            "Returned to base",
        )

    driver_x, driver_y, state = driver_position_at_time(
        current_time=current_time,
        route_row=route_row,
        stops=stops_for_route,
        driver_row=driver_row,
        marker_offset=0.0,
    )

    return (
        driver_x,
        driver_y,
        state,
    )

def build_static_route_figure(
    route_id: str,
    route_summaries: pd.DataFrame,
    route_stops: pd.DataFrame,
    locations: dict[str, pd.DataFrame],
) -> go.Figure:
    route_row, stops, driver_row = route_coordinates(
        route_id=route_id,
        route_summaries=route_summaries,
        route_stops=route_stops,
        locations=locations,
    )

    x_values = (
        [float(driver_row["x"])]
        + stops["x"].astype(float).tolist()
        + [float(driver_row["x"])]
    )

    y_values = (
        [float(driver_row["y"])]
        + stops["y"].astype(float).tolist()
        + [float(driver_row["y"])]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line={"color": DARK_BLUE, "width": 3},
            name="Route path",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[float(driver_row["x"])],
            y=[float(driver_row["y"])],
            mode="markers+text",
            marker={
                "symbol": "triangle-up",
                "size": 15,
                "color": DARK_BLUE,
                "line": {"color": "white", "width": 1},
            },
            text=["Base"],
            textposition="top center",
            name="Driver base",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stops["x"],
            y=stops["y"],
            mode="markers+text",
            marker={
                "size": 12,
                "color": RED,
                "line": {"color": "white", "width": 1},
            },
            text=stops["stop_number"].astype(str),
            textposition="top center",
            customdata=np.stack(
                [
                    stops["orderid"],
                    stops["arrival_time"].map(format_time),
                    stops["service_start_time"].map(format_time),
                    stops["service_end_time"].map(format_time),
                    stops["waiting_time"],
                ],
                axis=-1,
            ),
            name="Orders",
            hovertemplate=(
                "<b>Stop %{text}</b><br>"
                "Order: %{customdata[0]}<br>"
                "Arrival: %{customdata[1]}<br>"
                "Service start: %{customdata[2]}<br>"
                "Service end: %{customdata[3]}<br>"
                "Waiting: %{customdata[4]:.1f} min"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"{route_row['driver_name']} | "
            f"Day {int(route_row['day_index'])} "
            f"({route_row['weekday']}) | "
            f"{int(route_row['n_orders'])} orders | "
            f"duration {route_row['route_duration']:.1f} min"
        ),
        height=650,
        margin={
            "l": 20,
            "r": 20,
            "t": 90,
            "b": 20,
        },
        xaxis_title="x coordinate",
        yaxis_title="y coordinate",
        plot_bgcolor="white",
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig

def _interpolate(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    start_time: float,
    end_time: float,
    current_time: float,
) -> tuple[float, float]:
    if end_time <= start_time:
        return end_x, end_y

    ratio = min(
        1.0,
        max(
            0.0,
            (current_time - start_time)
            / (end_time - start_time),
        ),
    )

    return (
        start_x + ratio * (end_x - start_x),
        start_y + ratio * (end_y - start_y),
    )

def driver_position_at_time(
    current_time: float,
    route_row: pd.Series,
    stops: pd.DataFrame,
    driver_row: pd.Series,
    marker_offset: float,
) -> tuple[float, float, str]:
    """
    Return the driver's plotted position at a given time.

    During service, the marker is deliberately offset slightly from the
    order location. This avoids covering the pulsing blue order marker
    while still showing that the driver is at that stop.
    """

    base_x = float(driver_row["x"])
    base_y = float(driver_row["y"])

    route_departure_time = float(
        route_row.get(
            "departure_time",
            route_row["shift_start"],
        )
    )

    if current_time < route_departure_time:
        return base_x, base_y, "Waiting at base"

    previous_x = base_x
    previous_y = base_y
    previous_departure = route_departure_time

    for stop in stops.itertuples():

        arrival_time = float(stop.arrival_time)
        service_start = float(stop.service_start_time)
        departure_time = float(stop.departure_time)

        stop_x = float(stop.x)
        stop_y = float(stop.y)

        if current_time < arrival_time:

            driver_x, driver_y = _interpolate(
                previous_x,
                previous_y,
                stop_x,
                stop_y,
                previous_departure,
                arrival_time,
                current_time,
            )

            return driver_x, driver_y, "Travelling"

        if current_time <= departure_time:

            # During waiting/service, nudge the driver marker slightly
            # up and right so the active order dot remains visible.
            return (
                stop_x + marker_offset,
                stop_y + marker_offset,
                "At stop",
            )

        previous_x = stop_x
        previous_y = stop_y
        previous_departure = departure_time

    return_arrival = float(route_row["return_arrival_time"])

    if current_time <= return_arrival:

        driver_x, driver_y = _interpolate(
            previous_x,
            previous_y,
            base_x,
            base_y,
            previous_departure,
            return_arrival,
            current_time,
        )

        return driver_x, driver_y, "Returning"

    return base_x, base_y, "Returned"

def order_status_at_time(
    stop: pd.Series,
    current_time: float,
    frame_index: int,
) -> tuple[str, str, int]:
    """
    Return order visual state at a given animation time.

    The order turns blue as soon as the driver arrives, not only when
    formal service starts. This avoids the visual delay where the driver
    marker reaches the order but the order still appears pending.
    """

    arrival_time = float(
        stop.arrival_time
    )

    service_end_time = float(
        stop.service_end_time
    )

    if current_time < arrival_time:
        return (
            "Pending",
            ORDER_PENDING_RED,
            12,
        )

    if current_time <= service_end_time:

        pulse_size = (
            22
            if frame_index % 2 == 0
            else 15
        )

        return (
            "Being visited",
            ORDER_SERVICE_BLUE,
            pulse_size,
        )

    return (
        "Completed",
        ORDER_COMPLETED_GREEN,
        13,
    )

def build_route_animation_times(
    start_time: float,
    end_time: float,
    stops: pd.DataFrame,
) -> list[float]:
    """
    Build animation times using a fixed smooth time step, plus exact
    event times. Including arrival and service times avoids visual delays
    where the driver reaches an order before the order marker changes state.
    """

    regular_times = list(
        np.arange(
            start_time,
            end_time + ROUTE_ANIMATION_STEP_MINUTES,
            ROUTE_ANIMATION_STEP_MINUTES,
        )
    )

    event_columns = [
        "arrival_time",
        "service_start_time",
        "service_end_time",
        "departure_time",
    ]

    event_times = []

    for column in event_columns:

        if column in stops.columns:

            event_times.extend(
                stops[column]
                .dropna()
                .astype(float)
                .tolist()
            )

    all_times = (
        regular_times
        + event_times
        + [
            start_time,
            end_time,
        ]
    )

    cleaned_times = sorted(
        {
            round(float(value), 6)
            for value in all_times
            if start_time <= float(value) <= end_time
        }
    )

    if not cleaned_times:
        cleaned_times = [start_time]

    return cleaned_times

def build_route_visualisation_figure(
    route_id: str,
    route_summaries: pd.DataFrame,
    route_stops: pd.DataFrame,
    locations: dict[str, pd.DataFrame],
    playback_speed: str = "2x",
) -> go.Figure:
    """
    Build a single route visualisation figure.

    The initial state is a static route overview. The same figure also
    contains animation frames, Play/Pause controls and a time slider.
    """

    route_row, stops, driver_row = route_coordinates(
        route_id=route_id,
        route_summaries=route_summaries,
        route_stops=route_stops,
        locations=locations,
    )

    start_time = float(route_row["shift_start"])
    route_departure_time = float(
        route_row.get(
            "departure_time",
            start_time,
        )
    )
    end_time = float(route_row["return_arrival_time"])

    time_values = build_route_animation_times(
        start_time=start_time,
        end_time=end_time,
        stops=stops,
    )

    time_values = sorted(
        {
            *time_values,
            round(route_departure_time, 6),
        }
    )

    base_x = float(driver_row["x"])
    base_y = float(driver_row["y"])

    path_x = (
        [base_x]
        + stops["x"].astype(float).tolist()
        + [base_x]
    )

    path_y = (
        [base_y]
        + stops["y"].astype(float).tolist()
        + [base_y]
    )

    x_span = max(path_x) - min(path_x)
    y_span = max(path_y) - min(path_y)

    marker_offset = (
        DRIVER_SERVICE_OFFSET_FRACTION
        * max(x_span, y_span, 0.01)
    )

    frame_duration_ms = (
        PLAYBACK_SPEED_TO_FRAME_DURATION_MS[
            playback_speed
        ]
    )

    def make_frame_data(
        current_time: float,
        frame_index: int,
    ):
        driver_x, driver_y, driver_state = (
            driver_position_at_time(
                current_time=current_time,
                route_row=route_row,
                stops=stops,
                driver_row=driver_row,
                marker_offset=marker_offset,
            )
        )

        status_data = [
            order_status_at_time(
                stop,
                current_time,
                frame_index,
            )
            for _, stop in stops.iterrows()
        ]

        statuses = [
            item[0]
            for item in status_data
        ]

        colours = [
            item[1]
            for item in status_data
        ]

        sizes = [
            item[2]
            for item in status_data
        ]

        order_custom_data = np.stack(
            [
                stops["orderid"],
                statuses,
                stops["arrival_time"].map(format_time),
                stops["service_start_time"].map(format_time),
                stops["service_end_time"].map(format_time),
                stops["waiting_time"],
            ],
            axis=-1,
        )

        return [
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines",
                line={
                    "color": DARK_BLUE,
                    "width": 3,
                },
                name="Route path",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[base_x],
                y=[base_y],
                mode="markers+text",
                marker={
                    "symbol": "triangle-up",
                    "size": 15,
                    "color": DARK_BLUE,
                    "line": {
                        "color": "white",
                        "width": 1,
                    },
                },
                text=["Base"],
                textposition="top center",
                name="Driver base",
                hovertemplate=(
                    "<b>Driver base</b><br>"
                    f"{route_row['driver_name']}"
                    "<extra></extra>"
                ),
            ),
            go.Scatter(
                x=[driver_x],
                y=[driver_y],
                mode="text",
                text=[DRIVER_POSITION_ICON],
                textfont={
                    "size": DRIVER_ICON_FONT_SIZE,
                },
                name="Driver position",
                hovertemplate=(
                    f"<b>{route_row['driver_name']}</b><br>"
                    f"Time: {format_time(current_time)}<br>"
                    f"Status: {driver_state}"
                    "<extra></extra>"
                ),
            ),
            go.Scatter(
                x=stops["x"],
                y=stops["y"],
                mode="markers+text",
                marker={
                    "size": sizes,
                    "color": colours,
                    "line": {
                        "color": "white",
                        "width": 1,
                    },
                },
                text=stops["stop_number"].astype(str),
                textposition="top center",
                customdata=order_custom_data,
                name="Orders",
                hovertemplate=(
                    "<b>Stop %{text}</b><br>"
                    "Order: %{customdata[0]}<br>"
                    "Status: %{customdata[1]}<br>"
                    "Arrival: %{customdata[2]}<br>"
                    "Service start: %{customdata[3]}<br>"
                    "Service end: %{customdata[4]}<br>"
                    "Waiting: %{customdata[5]:.1f} min"
                    "<extra></extra>"
                ),
            ),
        ]

    frames = [
        go.Frame(
            data=make_frame_data(
                time_value,
                frame_index,
            ),
            name=format_time(time_value),
        )
        for frame_index, time_value in enumerate(
            time_values
        )
    ]

    fig = go.Figure(
        data=make_frame_data(
            time_values[0],
            0,
        ),
        frames=frames,
    )

    fig.update_layout(
        title=None,
        height=620,
        margin={
            "l": 20,
            "r": 20,
            "t": 20,
            "b": 20,
        },
        xaxis_title="x coordinate",
        yaxis_title="y coordinate",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
        },
    )
    
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig

def build_overall_day_visualisation_figure(
    day_index: int,
    route_summaries: pd.DataFrame,
    route_stops: pd.DataFrame,
    locations: dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Build an animated full-day route overview.

    This visualisation intentionally does not draw route edges because all
    driver routes shown simultaneously would become visually chaotic.
    """

    day_routes = route_summaries.loc[
        route_summaries["day_index"]
        .astype(int)
        .eq(int(day_index))
    ].copy()

    if day_routes.empty:

        fig = go.Figure()

        fig.update_layout(
            title=None,
            height=620,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    route_ids = (
        day_routes["route_id"]
        .astype(str)
        .tolist()
    )

    day_stops = route_stops.loc[
        route_stops["route_id"]
        .astype(str)
        .isin(route_ids)
    ].copy()

    day_stops = day_stops.merge(
        locations["orders"][
            [
                "orderid",
                "x",
                "y",
            ]
        ],
        on="orderid",
        how="left",
    )

    day_stops = day_stops.sort_values(
        [
            "route_id",
            "stop_number",
        ]
    ).reset_index(drop=True)

    driver_locations = (
        locations["drivers"]
        .set_index("driver_id")
    )

    time_values = build_overall_day_animation_times(
        day_routes=day_routes,
        day_stops=day_stops,
    )

    routes_by_id = {
        str(row["route_id"]): row
        for _, row in day_routes.iterrows()
    }

    stops_by_route_id = {
        str(route_id): group.copy().reset_index(drop=True)
        for route_id, group
        in day_stops.groupby("route_id", sort=False)
    }

    base_x_values = []
    base_y_values = []
    base_custom_data = []

    for _, route in day_routes.iterrows():

        driver_id = int(
            route["driver_id"]
        )

        driver_row = driver_locations.loc[
            driver_id
        ]

        base_x_values.append(
            float(driver_row["x"])
        )

        base_y_values.append(
            float(driver_row["y"])
        )

        base_custom_data.append(
            [
                route["driver_name"],
                int(route["driver_id"]),
                int(route["n_orders"]),
            ]
        )

    def make_frame_data(
        current_time: float,
        frame_index: int,
    ):

        del frame_index

        order_colours = [
            overall_order_colour_at_time(
                stop=row,
                current_time=current_time,
            )
            for _, row in day_stops.iterrows()
        ]

        order_custom_data = np.stack(
            [
                day_stops["orderid"],
                day_stops["driver_name"],
                day_stops["stop_number"],
                day_stops["service_end_time"].map(format_time),
            ],
            axis=-1,
        )

        driver_x_values = []
        driver_y_values = []
        driver_custom_data = []

        for route_id, route_row in routes_by_id.items():

            driver_id = int(
                route_row["driver_id"]
            )

            driver_row = driver_locations.loc[
                driver_id
            ]

            stops_for_route = stops_by_route_id.get(
                route_id,
                pd.DataFrame(),
            )

            driver_x, driver_y, state = overall_driver_position_at_time(
                current_time=current_time,
                route_row=route_row,
                stops_for_route=stops_for_route,
                driver_row=driver_row,
            )

            driver_x_values.append(
                driver_x
            )

            driver_y_values.append(
                driver_y
            )

            driver_custom_data.append(
                [
                    route_row["driver_name"],
                    driver_id,
                    state,
                    int(route_row["n_orders"]),
                    format_time(
                        float(route_row["shift_start"])
                    ),
                    format_time(
                        float(route_row["return_arrival_time"])
                    ),
                ]
            )

        return [
            go.Scatter(
                x=day_stops["x"],
                y=day_stops["y"],
                mode="markers",
                marker={
                    "size": OVERALL_ORDER_DOT_SIZE,
                    "color": order_colours,
                    "line": {
                        "color": "white",
                        "width": 0.5,
                    },
                },
                customdata=order_custom_data,
                name="Orders",
                hovertemplate=(
                    "<b>Order %{customdata[0]}</b><br>"
                    "Driver: %{customdata[1]}<br>"
                    "Stop: %{customdata[2]}<br>"
                    "Completed by: %{customdata[3]}"
                    "<extra></extra>"
                ),
            ),
            go.Scatter(
                x=base_x_values,
                y=base_y_values,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": OVERALL_BASE_MARKER_SIZE,
                    "color": OVERALL_BASE_BLUE,
                    "opacity": 0.55,
                    "line": {
                        "color": "white",
                        "width": 0.5,
                    },
                },
                customdata=np.array(
                    base_custom_data
                ),
                name="Driver bases",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Driver ID: %{customdata[1]}<br>"
                    "Orders today: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ),
            go.Scatter(
                x=driver_x_values,
                y=driver_y_values,
                mode="markers",
                marker={
                    "size": OVERALL_DRIVER_DOT_SIZE,
                    "color": OVERALL_DRIVER_BLUE,
                    "line": {
                        "color": "white",
                        "width": 0.75,
                    },
                },
                customdata=np.array(
                    driver_custom_data
                ),
                name="Drivers",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Driver ID: %{customdata[1]}<br>"
                    "Status: %{customdata[2]}<br>"
                    "Orders today: %{customdata[3]}<br>"
                    "Start: %{customdata[4]}<br>"
                    "Return: %{customdata[5]}"
                    "<extra></extra>"
                ),
            ),
        ]

    frames = [
        go.Frame(
            data=make_frame_data(
                time_value,
                frame_index,
            ),
            name=format_time(time_value),
        )
        for frame_index, time_value in enumerate(
            time_values
        )
    ]

    fig = go.Figure(
        data=make_frame_data(
            time_values[0],
            0,
        ),
        frames=frames,
    )

    fig.update_layout(
        title=None,
        height=620,
        margin={
            "l": 20,
            "r": 20,
            "t": 20,
            "b": 20,
        },
        xaxis_title="x coordinate",
        yaxis_title="y coordinate",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig

def render_route_player(
    fig: go.Figure,
    playback_speed: str,
    height: int = 760,
) -> None:
    """
    Render the Plotly route chart with external Play/Pause controls.

    This avoids the layout problems caused by Plotly's built-in animation
    buttons and frame-title updates.
    """

    frame_duration_ms = PLAYBACK_SPEED_TO_FRAME_DURATION_MS[
        playback_speed
    ]

    div_id = "route_plot_" + uuid.uuid4().hex

    plot_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        auto_play=False,
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True,
        },
    )

    frame_names = [
        frame.name
        for frame in fig.frames
    ]

    initial_time = (
        frame_names[0]
        if frame_names
        else ""
    )

    html_content = f"""
    <div class="route-player-wrapper">

        <div class="route-player-controls">
            <button id="{div_id}_play" class="route-play-button">
                ▶ Play
            </button>

            <span id="{div_id}_time" class="route-time-label">
                Time: {initial_time}
            </span>
        </div>

        <div class="route-plot">
            {plot_html}
        </div>

        <div class="route-slider-wrapper">
            <input
                id="{div_id}_slider"
                class="route-slider"
                type="range"
                min="0"
                max="{max(len(frame_names) - 1, 0)}"
                value="0"
                step="1"
            />
        </div>

    </div>

    <style>
        .route-player-wrapper {{
            width: 100%;
            background: white;
            border-radius: 12px;
        }}

        .route-player-controls {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 0 0 8px 0;
        }}

        .route-play-button {{
            background: #1E3A8A;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: 700;
            cursor: pointer;
            font-size: 14px;
            min-width: 84px;
        }}

        .route-play-button:hover {{
            background: #1D4ED8;
        }}

        .route-time-label {{
            color: #334155;
            font-size: 14px;
            font-weight: 600;
        }}

        .route-slider-wrapper {{
            margin: -12px 44px 0 58px;
        }}

        .route-slider {{
            width: 100%;
            accent-color: #1E3A8A;
        }}
    </style>

    <script>
        const graphDiv = document.getElementById("{div_id}");
        const playButton = document.getElementById("{div_id}_play");
        const timeLabel = document.getElementById("{div_id}_time");
        const slider = document.getElementById("{div_id}_slider");

        const frameNames = {json.dumps(frame_names)};
        const frameDurationMs = {int(frame_duration_ms)};

        let frameIndex = 0;
        let isPlaying = false;
        let timer = null;

        function setFrame(index) {{
            if (frameNames.length === 0) {{
                return;
            }}

            frameIndex = Math.max(
                0,
                Math.min(index, frameNames.length - 1)
            );

            Plotly.animate(
                graphDiv,
                [frameNames[frameIndex]],
                {{
                    mode: "immediate",
                    frame: {{
                        duration: 0,
                        redraw: true
                    }},
                    transition: {{
                        duration: 0
                    }}
                }}
            );

            slider.value = frameIndex;
            timeLabel.innerText = "Time: " + frameNames[frameIndex];
        }}

        function pauseAnimation() {{
            isPlaying = false;
            playButton.innerText = "▶ Play";

            if (timer !== null) {{
                clearInterval(timer);
                timer = null;
            }}
        }}

        function playAnimation() {{
            if (frameNames.length === 0) {{
                return;
            }}

            isPlaying = true;
            playButton.innerText = "⏸ Pause";

            timer = setInterval(
                function() {{
                    if (frameIndex >= frameNames.length - 1) {{
                        pauseAnimation();
                        return;
                    }}

                    setFrame(frameIndex + 1);
                }},
                frameDurationMs
            );
        }}

        playButton.addEventListener(
            "click",
            function() {{
                if (isPlaying) {{
                    pauseAnimation();
                }} else {{
                    playAnimation();
                }}
            }}
        );

        slider.addEventListener(
            "input",
            function() {{
                pauseAnimation();
                setFrame(parseInt(slider.value));
            }}
        );
    </script>
    """

    components.html(
        html_content,
        height=height,
        scrolling=False,
    )

st.title("Stage 3 Routing & Repair Dashboard")

st.markdown(
    """
    <div class="dashboard-subtitle">
        Inspect the Decomposed Driver-Day Routing Layer, Compare Unrepaired
        and Repaired Route Feasibility & Visualise Selected Daily Routes
    </div>
    """,
    unsafe_allow_html=True,
)

use_repaired = st.toggle(
    "Use Repaired Stage 3 Results",
    value=True,
    help=(
        "Turn this off to inspect the unrepaired routing-validation baseline."
    ),
)

if "stage3_equity_measure_type" not in st.session_state:
    st.session_state["stage3_equity_measure_type"] = (
        EQUITY_MEASURE_TYPES[0]
    )

selected_equity_measure = st.session_state[
    "stage3_equity_measure_type"
]

selected_equity_label = EQUITY_MEASURE_LABELS[
    selected_equity_measure
]

try:
    master_summary = load_stage3_master_summary(
        use_repaired=use_repaired,
        equity_measure_type=selected_equity_measure,
    )

except Exception as error:
    st.error(str(error))
    st.stop()


available_lambdas = (
    master_summary["lambda_equity"]
    .dropna()
    .astype(float)
    .sort_values()
    .unique()
    .tolist()
)

if not available_lambdas:
    st.error(
        "No Stage 3 lambda values are available for the selected equity measure."
    )
    st.stop()

default_lambda = (
    0.0
    if 0.0 in available_lambdas
    else available_lambdas[0]
)

state_lambda = st.session_state.get(
    "stage3_selected_lambda",
    default_lambda,
)

available_lambda_keys = [
    round(float(value), 3)
    for value in available_lambdas
]

if round(float(state_lambda), 3) not in available_lambda_keys:
    st.session_state["stage3_selected_lambda"] = default_lambda
else:
    st.session_state["stage3_selected_lambda"] = float(state_lambda)

selected_lambda = float(
    st.session_state["stage3_selected_lambda"]
)

selected_summary_row = selected_stage3_row_for_lambda(
    summary=master_summary,
    selected_lambda=selected_lambda,
)

selected_model_scope = str(
    selected_summary_row["model_scope"]
)

selected_lambda_tag = str(
    selected_summary_row["lambda_tag"]
)

try:
    solution_data = load_stage3_solution_files(
        model_scope=selected_model_scope,
        lambda_tag=selected_lambda_tag,
        use_repaired=use_repaired,
        equity_measure_type=selected_equity_measure,
    )

    locations = load_stage3_locations()

except Exception as error:
    st.error(str(error))
    st.stop()


route_summaries = solution_data["route_summaries"]
route_stops = solution_data["route_stops"]
driver_workloads = solution_data["driver_route_workloads"]
repair_move_log = solution_data["repair_move_log"]
planning_day_labels = get_stage3_planning_day_labels()


# BASELINE COMPARISON METRICS


baseline_lambda = (
    0.0
    if 0.0 in available_lambdas
    else min(available_lambdas)
)

baseline_summary_row = selected_stage3_row_for_lambda(
    summary=master_summary,
    selected_lambda=baseline_lambda,
)

baseline_model_scope = str(
    baseline_summary_row["model_scope"]
)

baseline_lambda_tag = str(
    baseline_summary_row["lambda_tag"]
)

if (
    baseline_model_scope == selected_model_scope
    and baseline_lambda_tag == selected_lambda_tag
):
    baseline_driver_workloads = driver_workloads

else:
    baseline_solution_data = load_stage3_solution_files(
        model_scope=baseline_model_scope,
        lambda_tag=baseline_lambda_tag,
        use_repaired=use_repaired,
        equity_measure_type=selected_equity_measure,
    )

    baseline_driver_workloads = baseline_solution_data[
        "driver_route_workloads"
    ]


selected_total_route_duration = float(
    selected_summary_row["total_route_duration"]
)

baseline_total_route_duration = float(
    baseline_summary_row["total_route_duration"]
)

selected_route_equity_metric = route_workload_selected_equity_value(
    driver_workloads=driver_workloads,
    equity_measure_type=selected_equity_measure,
)

baseline_route_equity_metric = route_workload_selected_equity_value(
    driver_workloads=baseline_driver_workloads,
    equity_measure_type=selected_equity_measure,
)

selected_route_cv = float(
    selected_summary_row["cv_active_route_workload"]
)

baseline_route_cv = float(
    baseline_summary_row["cv_active_route_workload"]
)

route_duration_change_pct = safe_percentage_change(
    current_value=selected_total_route_duration,
    baseline_value=baseline_total_route_duration,
)

selected_equity_metric_reduction_pct = safe_percentage_reduction(
    current_value=selected_route_equity_metric,
    baseline_value=baseline_route_equity_metric,
)

route_cv_reduction_pct = safe_percentage_reduction(
    current_value=selected_route_cv,
    baseline_value=baseline_route_cv,
)

if bool_value(
    selected_summary_row.get("all_routes_feasible", False)
):

    st.markdown(
        """
        <div class='success-box'>
            All Routes are Feasible Under the Routed Schedule
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class='warning-box'>
            This Solution has Unrepaired Route-Level Violations
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label="Total Routed Duration",
    value=format_minutes(
        selected_total_route_duration
    ),
    delta=f"{route_duration_change_pct:+.2f}% vs. Baseline",
    delta_color="inverse",
)

col2.metric(
    label="Selected Equity Penalty",
    value=format_equity_penalty_metric(
        selected_route_equity_metric
    ),
    delta=(
        f"{selected_equity_metric_reduction_pct:.2f}% "
        "Reduction vs. Baseline"
    ),
    delta_color="normal",
)

col3.metric(
    label="Active-Driver Routed Workload CoV",
    value=f"{selected_route_cv:.3f}",
    delta=(
        f"{route_cv_reduction_pct:.2f}% "
        "Reduction vs. Baseline"
    ),
    delta_color="normal",
)

col4.metric(
    label="Infeasible Routes",
    value=f"{int(selected_summary_row['n_infeasible_routes'])}",
)

col5.metric(
    label="Repair Moves",
    value=f"{int(selected_summary_row.get('n_repair_moves', 0))}",
)


# SOLUTION CONTROLS


st.markdown("### Solution Controls")

control_1, control_2 = st.columns([1, 2])

with control_1:

    st.selectbox(
        "Equity Measure Type",
        options=EQUITY_MEASURE_TYPES,
        format_func=lambda value: EQUITY_MEASURE_LABELS[value],
        key="stage3_equity_measure_type",
    )

with control_2:

    st.select_slider(
        "Equity Emphasis",
        options=available_lambdas,
        key="stage3_selected_lambda",
        format_func=lambda value: f"λ = {float(value):.1f}",
        help=(
            "Select the stored routed solution for the current equity "
            "measure. The selected value corresponds to the Stage 2 "
            "equity-emphasis lambda that generated the routed solution."
        ),
    )

tabs = st.tabs(
    [
        "Solution Summary",
        "Repair Impact",
        "Driver Workloads",
        "Route Explorer",
        "Driver Route Visualisation",
        "Overall Route Visualisation",
        "Tables & Downloads",
    ]
)

with tabs[0]:

    st.subheader("Routed Efficiency-Equity Frontier")

    frontier = add_selected_route_equity_metric(
        summary=master_summary,
        use_repaired=use_repaired,
        equity_measure_type=selected_equity_measure,
    )

    frontier["solution_label"] = frontier.apply(
        solution_label,
        axis=1,
    )

    fig = px.line(
        frontier,
        x="total_route_duration",
        y="selected_route_equity_metric_value",
        text="lambda_equity",
        markers=True,
        labels={
            "total_route_duration":
                "Total routed duration (minutes)",
            "selected_route_equity_metric_value":
                selected_equity_label,
        },
    )

    fig.update_traces(
        line={"color": DARK_BLUE, "width": 3},
        marker={"size": 10, "color": DARK_BLUE},
        texttemplate="λ=%{text:.1f}",
        textposition="top center",
    )

    fig.add_scatter(
        x=[float(selected_summary_row["total_route_duration"])],
        y=[
            float(
                frontier.loc[
                    frontier["lambda_tag"]
                    .astype(str)
                    .eq(selected_lambda_tag),
                    "selected_route_equity_metric_value",
                ].iloc[0]
            )
        ],
        mode="markers",
        marker={
            "size": 19,
            "symbol": "diamond",
            "color": AMBER,
            "line": {"color": "white", "width": 2},
        },
        name="Selected",
    )

    fig.update_traces(
        hovertemplate=(
            "λ=%{text:.1f}<br>"
            "Total routed duration: %{x:,.1f} min<br>"
            f"{selected_equity_label}: %{{y:,.4f}}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=540,
        plot_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    display_columns = [
        "model_scope",
        "lambda_tag",
        "lambda_equity",
        "n_driver_day_routes",
        "n_infeasible_routes",
        "total_route_duration",
        "selected_route_equity_metric_value",
        "total_travel_time",
        "total_service_time",
        "total_waiting_time",
        "max_driver_route_workload",
        "cv_active_route_workload",
        "route_duration_vs_assignment_proxy_pct",
        "n_repair_moves",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in frontier.columns
    ]

    st.dataframe(
        frontier[display_columns],
        hide_index=True,
        width="stretch",
    )

    with tabs[1]:

        st.subheader("Unrepaired vs. Repaired Route Feasibility")

        try:
            comparison = load_stage3_repair_comparison(
                equity_measure_type=selected_equity_measure,
            )
        except Exception:
            comparison = pd.DataFrame()

        if comparison.empty:

            st.info(
                "No repair comparison table is available yet."
            )

        else:

            comparison["solution"] = comparison["lambda_equity"].apply(
                lambda value: f"λ={float(value):.1f}"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=comparison["solution"],
                    y=comparison[
                        "total_violation_minutes_unrepaired"
                    ],
                    name="Unrepaired Violation Minutes",
                    marker_color=RED,
                )
            )

            fig.add_trace(
                go.Bar(
                    x=comparison["solution"],
                    y=comparison[
                        "total_violation_minutes_repaired"
                    ],
                    name="Repaired Violation Minutes",
                    marker_color=GREEN,
                )
            )

            fig.update_layout(
                barmode="group",
                height=500,
                yaxis_title="Total Violation (mins)",
                xaxis_title="λ",
                plot_bgcolor="white",
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 30,
                    "b": 90,
                },
            )

            fig.update_xaxes(
                tickangle=0,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                comparison,
                hide_index=True,
                width="stretch",
            )

        st.subheader("Repair Moves for Selected Solution")

        if repair_move_log.empty:
            st.success(
                "No repair moves were required for this solution."
            )

        else:
            st.dataframe(
                repair_move_log,
                hide_index=True,
                width="stretch",
            )

    with tabs[2]:

        st.subheader("Driver Routed Workload Distribution")

        all_driver_workloads = driver_workloads.copy()

        st.caption(
            "All drivers are displayed, including drivers receiving no routed "
            "workload in the selected solution."
        )

        sort_metric = st.selectbox(
            "Sort Drivers By",
            options=[
                "total_route_duration",
                "horizon_utilisation_pct",
                "n_orders",
                "n_driver_days",
            ],
            format_func=lambda value: (
                value.replace("_", " ").title()
            ),
        )

        if sort_metric not in all_driver_workloads.columns:
            st.warning(
                f"The selected sort metric '{sort_metric}' is not available "
                "for this solution. Drivers are being sorted by total route duration."
            )
            sort_metric = "total_route_duration"

        displayed_drivers = (
            all_driver_workloads
            .sort_values(
                sort_metric,
                ascending=False,
            )
        )

        fig = px.bar(
            displayed_drivers,
            x="driver_name",
            y="total_route_duration",
            custom_data=[
                "driver_id",
                "n_orders",
                "n_driver_days",
                "horizon_utilisation_pct",
                "total_travel_time",
                "total_service_time",
                "total_waiting_time",
            ],
            labels={
                "driver_name": "Driver",
                "total_route_duration":
                    "Total Routed Workload (mins)",
            },
        )

        fig.update_traces(
            marker_color=DARK_BLUE,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Driver ID: %{customdata[0]}<br>"
                "Orders: %{customdata[1]}<br>"
                "Driver-days: %{customdata[2]}<br>"
                "Utilisation: %{customdata[3]:.1f}%<br>"
                "Travel: %{customdata[4]:,.1f} min<br>"
                "Service: %{customdata[5]:,.1f} min<br>"
                "Waiting: %{customdata[6]:,.1f} min"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            height=640,
            plot_bgcolor="white",
            margin={"l": 20, "r": 20, "t": 30, "b": 100},
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.dataframe(
            driver_workloads,
            hide_index=True,
            width="stretch",
        )
    
    with tabs[3]:

        st.subheader("Driver-Day Route Explorer")

        c1, c2, c3 = st.columns(3)

        day_values = sorted(
            route_summaries["day_index"].astype(int).unique()
        )

        selected_day = c1.selectbox(
            "Planning Day",
            options=day_values,
            format_func=lambda day_index: planning_day_label(
                day_index,
                planning_day_labels,
            ),
        )

        day_routes = route_summaries.loc[
            route_summaries["day_index"]
            .astype(int)
            .eq(selected_day)
        ].copy()

        driver_options = (
            ["All Drivers"]
            + sorted(
                day_routes["driver_name"]
                .astype(str)
                .unique()
                .tolist()
            )
        )

        selected_driver = c2.selectbox(
            "Driver",
            options=driver_options,
        )

        if selected_driver != "All Drivers":

            day_routes = day_routes.loc[
                day_routes["driver_name"]
                .astype(str)
                .eq(selected_driver)
            ]

        feasible_filter = c3.selectbox(
            "Route Feasibility",
            options=[
                "All Routes",
                "Feasible Only",
                "Infeasible Only",
            ],
        )

        if feasible_filter == "Feasible Only":

            day_routes = day_routes.loc[
                day_routes["route_feasible"].astype(bool)
            ]

        elif feasible_filter == "Infeasible Only":

            day_routes = day_routes.loc[
                ~day_routes["route_feasible"].astype(bool)
            ]

        st.dataframe(
            day_routes.sort_values(
                ["driver_id", "route_id"]
            ),
            hide_index=True,
            width="stretch",
        )

        if not day_routes.empty:

            route_lookup = route_options(
                day_routes,
                planning_day_labels=planning_day_labels,
            )

            selected_route_label = st.selectbox(
                "Select Route to Inspect",
                options=list(route_lookup.keys()),
            )

            selected_route_id = route_lookup[
                selected_route_label
            ]

            selected_stops = route_stops.loc[
                route_stops["route_id"]
                .astype(str)
                .eq(selected_route_id)
            ].sort_values("stop_number")

            st.dataframe(
                selected_stops,
                hide_index=True,
                width="stretch",
            )

    with tabs[4]:

        st.subheader("Driver Route Visualisation")

        filter_col_1, filter_col_2, filter_col_3 = st.columns(
            [0.9, 1.4, 3.2]
        )

        visual_day = filter_col_1.selectbox(
            "Day",
            options=sorted(
                route_summaries["day_index"]
                .astype(int)
                .unique()
            ),
            format_func=lambda day_index: planning_day_label(
                day_index,
                planning_day_labels,
            ),
            key="visual_day",
        )

        visual_routes = route_summaries.loc[
            route_summaries["day_index"]
            .astype(int)
            .eq(visual_day)
        ].copy()

        visual_driver_options = (
            ["All Drivers"]
            + sorted(
                visual_routes["driver_name"]
                .astype(str)
                .unique()
                .tolist()
            )
        )

        visual_driver = filter_col_2.selectbox(
            "Driver",
            options=visual_driver_options,
            key="visual_driver",
        )

        if visual_driver != "All Drivers":

            visual_routes = visual_routes.loc[
                visual_routes["driver_name"]
                .astype(str)
                .eq(visual_driver)
            ]

        if visual_routes.empty:

            st.info(
                "No visualisation routes match the selected filters."
            )

        else:

            visual_route_lookup = route_options(
                visual_routes,
                planning_day_labels=planning_day_labels,
            )

            visual_route_label = filter_col_3.selectbox(
                "Route",
                options=list(visual_route_lookup.keys()),
                key="visual_route",
            )

            visual_route_id = visual_route_lookup[
                visual_route_label
            ]

            playback_speed = "10x"

            st.caption(
                "Use the Play/Pause button below to animate the selected route. "
                "Playback is fixed at 10x. The time slider below the chart can also "
                "be dragged manually."
            )

            selected_route_row = visual_routes.loc[
                visual_routes["route_id"]
                .astype(str)
                .eq(visual_route_id)
            ].iloc[0]

            route_day_label = planning_day_label(
                int(selected_route_row["day_index"]),
                planning_day_labels,
            )

            st.markdown(
                (
                    f"**{selected_route_row['driver_name']}** | "
                    f"{route_day_label} "
                    f"({selected_route_row['weekday']}) | "
                    f"{int(selected_route_row['n_orders'])} Orders | "
                    f"Duration {float(selected_route_row['route_duration']):.1f} min"
                )
            )

            visualisation_figure = build_route_visualisation_figure(
                route_id=visual_route_id,
                route_summaries=route_summaries,
                route_stops=route_stops,
                locations=locations,
                playback_speed=playback_speed,
            )

            render_route_player(
                fig=visualisation_figure,
                playback_speed=playback_speed,
            )

            st.caption(
                "Legend: bright red = pending order, bright blue = driver has arrived "
                "and the order is being visited, bright green = completed order, "
                "dark-blue triangle = driver base, "
                f"{DRIVER_POSITION_ICON} = driver position."
            )

    with tabs[5]:

        st.subheader("Overall Route Visualisation")

        overall_filter_col_1, overall_filter_col_2 = st.columns(
            [1.0, 4.0]
        )

        overall_day = overall_filter_col_1.selectbox(
            "Day",
            options=sorted(
                route_summaries["day_index"]
                .astype(int)
                .unique()
            ),
            format_func=lambda day_index: planning_day_label(
                day_index,
                planning_day_labels,
            ),
            key="overall_visual_day",
        )

        overall_playback_speed = "10x"

        day_routes = route_summaries.loc[
            route_summaries["day_index"]
            .astype(int)
            .eq(int(overall_day))
        ].copy()

        day_route_ids = (
            day_routes["route_id"]
            .astype(str)
            .tolist()
        )

        day_stops = route_stops.loc[
            route_stops["route_id"]
            .astype(str)
            .isin(day_route_ids)
        ].copy()

        day_weekday = (
            str(day_routes["weekday"].iloc[0])
            if not day_routes.empty
            else ""
        )

        n_day_drivers = int(
            day_routes["driver_id"].nunique()
        )

        n_day_orders = int(
            day_stops["orderid"].nunique()
        )

        overall_day_label = planning_day_label(
            int(overall_day),
            planning_day_labels,
        )


        if day_routes.empty or day_stops.empty:

            st.info(
                "No routes are available for the selected day."
            )

        else:

            overall_title = (
                f"{overall_day_label} ({day_weekday}) | "
                f"{n_day_orders:,} Orders | "
                f"{n_day_drivers:,} Active Drivers"
            )

            st.markdown(
                f"**{overall_title}**"
            )

            overall_figure = build_overall_day_visualisation_figure(
                day_index=overall_day,
                route_summaries=route_summaries,
                route_stops=route_stops,
                locations=locations,
            )

            render_route_player(
                fig=overall_figure,
                playback_speed=overall_playback_speed,
                height=760,
            )

            st.caption(
                "Legend: red = scheduled but not yet completed order, "
                "green = completed order, blue dot = driver position, "
                "faint dark-blue triangle = driver base. Route edges are omitted "
                "in this full-day view to avoid visual clutter."
            )

    with tabs[6]:

        st.subheader("Tables and Downloads")

        st.write("Route Summaries")

        st.dataframe(
            route_summaries,
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download Route Summaries",
            route_summaries.to_csv(index=False),
            file_name=(
                f"stage3_route_summaries_"
                f"{selected_model_scope}_{selected_lambda_tag}.csv"
            ),
            mime="text/csv",
        )

        st.write("Route Stops")

        st.dataframe(
            route_stops,
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download Route Stops",
            route_stops.to_csv(index=False),
            file_name=(
                f"stage3_route_stops_"
                f"{selected_model_scope}_{selected_lambda_tag}.csv"
            ),
            mime="text/csv",
        )

        st.write("Driver Route Workloads")

        st.dataframe(
            driver_workloads,
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download Driver Route Workloads",
            driver_workloads.to_csv(index=False),
            file_name=(
                f"stage3_driver_route_workloads_"
                f"{selected_model_scope}_{selected_lambda_tag}.csv"
            ),
            mime="text/csv",
        )