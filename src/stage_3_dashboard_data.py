from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    DATA_PROCESSED_DIR,
    STAGE3_REPAIRED_RESULTS_DIR,
    STAGE3_RESULTS_DIR,
)

REPAIR_MOVE_LOG_COLUMNS = [
    "iteration",
    "orderid",
    "source_route_id",
    "source_day_index",
    "source_weekday",
    "source_driver_id",
    "source_driver_name",
    "target_day_index",
    "target_weekday",
    "target_driver_id",
    "target_driver_name",
    "old_affected_violation_minutes",
    "new_affected_violation_minutes",
    "old_affected_infeasible_routes",
    "new_affected_infeasible_routes",
    "old_affected_route_duration",
    "new_affected_route_duration",
]


@st.cache_data(show_spinner=False)
def _read_csv_cached(
    path_string: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    """
    Cached CSV reader whose cache invalidates when the file changes.
    """

    del modified_time_ns

    return pd.read_csv(path_string)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return _read_csv_cached(
        path_string=str(path),
        modified_time_ns=path.stat().st_mtime_ns,
    )

def _load_optional_csv(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load an optional CSV, returning an empty DataFrame for blank files.

    Some repaired Stage 3 solutions require no repair moves. Earlier runs
    created a blank repair_move_log.csv for those solutions, which makes
    pandas raise EmptyDataError. Treat those files as valid empty logs.
    """

    if columns is None:
        columns = []

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        return _read_csv_cached(
            path_string=str(path),
            modified_time_ns=path.stat().st_mtime_ns,
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

def stage3_root(use_repaired: bool) -> Path:
    return (
        STAGE3_REPAIRED_RESULTS_DIR
        if use_repaired
        else STAGE3_RESULTS_DIR
    )


def load_stage3_master_summary(
    use_repaired: bool = True,
    equity_measure_type: str = "maximum",
) -> pd.DataFrame:
    root = stage3_root(use_repaired)

    legacy_master = (
        root / "stage_3_repaired_master_summary.csv"
        if use_repaired
        else root / "stage_3_master_summary.csv"
    )

    summary_filename = "stage_3_solution_summary.csv"

    equity_root = (
        root
        / "full_instance"
        / equity_measure_type
    )

    if equity_root.exists():

        rows = []

        for path in sorted(
            equity_root.glob(f"*/{summary_filename}")
        ):
            rows.append(
                _load_csv(path)
            )

        if rows:
            summary = pd.concat(
                rows,
                ignore_index=True,
            )

            if "equity_measure_type" not in summary.columns:
                summary["equity_measure_type"] = equity_measure_type

            return (
                summary
                .sort_values(["model_scope", "lambda_equity"])
                .reset_index(drop=True)
            )

    if (
        equity_measure_type == "maximum"
        and legacy_master.exists()
    ):
        summary = _load_csv(
            legacy_master
        ).copy()

        if "equity_measure_type" not in summary.columns:
            summary["equity_measure_type"] = "maximum"

        return (
            summary
            .sort_values(["model_scope", "lambda_equity"])
            .reset_index(drop=True)
        )

    raise FileNotFoundError(
        "No Stage 3 routed outputs were found for equity measure "
        f"'{equity_measure_type}'. Run Stage 3 routing and repair for "
        "this equity measure before selecting it in the Stage 3 dashboard."
    )


def load_stage3_repair_comparison(
    equity_measure_type: str = "maximum",
) -> pd.DataFrame:
    unrepaired = load_stage3_master_summary(
        use_repaired=False,
        equity_measure_type=equity_measure_type,
    )

    repaired = load_stage3_master_summary(
        use_repaired=True,
        equity_measure_type=equity_measure_type,
    )

    key_columns = [
        "model_scope",
        "lambda_tag",
        "lambda_equity",
    ]

    if "equity_measure_type" in unrepaired.columns:
        key_columns.append("equity_measure_type")

    metric_columns = [
        "n_infeasible_routes",
        "total_violation_minutes",
        "total_route_duration",
        "max_driver_route_workload",
        "cv_active_route_workload",
    ]

    available_unrepaired = [
        column
        for column in key_columns + metric_columns
        if column in unrepaired.columns
    ]

    available_repaired = [
        column
        for column in key_columns + metric_columns
        if column in repaired.columns
    ]

    return unrepaired[available_unrepaired].merge(
        repaired[available_repaired],
        on=key_columns,
        how="inner",
        suffixes=("_unrepaired", "_repaired"),
    )


def solution_directory(
    model_scope: str,
    lambda_tag: str,
    use_repaired: bool = True,
    equity_measure_type: str = "maximum",
) -> Path:
    root = stage3_root(use_repaired)

    if model_scope == "full_instance":

        equity_directory = (
            root
            / model_scope
            / equity_measure_type
            / lambda_tag
        )

        if equity_directory.exists():
            return equity_directory

        legacy_directory = (
            root
            / model_scope
            / lambda_tag
        )

        if (
            equity_measure_type == "maximum"
            and legacy_directory.exists()
        ):
            return legacy_directory

        return equity_directory

    return (
        root
        / model_scope
        / lambda_tag
    )


def load_stage3_solution_files(
    model_scope: str,
    lambda_tag: str,
    use_repaired: bool = True,
    equity_measure_type: str = "maximum",
) -> dict[str, pd.DataFrame]:
    root = solution_directory(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        use_repaired=use_repaired,
        equity_measure_type=equity_measure_type,
    )

    result = {
        "route_summaries": _load_csv(
            root / "route_summaries.csv"
        ),
        "route_stops": _load_csv(
            root / "route_stops.csv"
        ),
        "driver_route_workloads": _load_csv(
            root / "driver_route_workloads.csv"
        ),
    }

    repair_log_path = root / "repair_move_log.csv"

    result["repair_move_log"] = _load_optional_csv(
        repair_log_path,
        columns=REPAIR_MOVE_LOG_COLUMNS,
    )

    repaired_assignments_path = (
        root / "repaired_assignments.csv"
    )

    result["assignments"] = _load_optional_csv(
        repaired_assignments_path,
    )

    return result


@st.cache_data(show_spinner=False)
def load_stage3_locations() -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(
        DATA_PROCESSED_DIR / "orders_clean.csv"
    )

    drivers = pd.read_csv(
        DATA_PROCESSED_DIR / "drivers_clean.csv"
    ).reset_index(drop=True)

    drivers["driver_id"] = drivers.index + 1

    order_locations = (
        orders[
            [
                "orderid",
                "locationid",
                "x",
                "y",
                "jobtime",
                "timefrom",
                "timeto",
            ]
        ]
        .drop_duplicates("orderid")
        .copy()
    )

    driver_locations = (
        drivers[
            [
                "driver_id",
                "driver.name",
                "locationid",
                "x",
                "y",
                "shiftstart",
                "shiftend",
            ]
        ]
        .rename(columns={"driver.name": "driver_name"})
        .copy()
    )

    return {
        "orders": order_locations,
        "drivers": driver_locations,
    }