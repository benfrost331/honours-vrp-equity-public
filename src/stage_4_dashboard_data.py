from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    EQUITY_MEASURE_TYPES,
    STAGE3_REPAIRED_RESULTS_DIR,
    STAGE4_RESULTS_DIR,
)


@st.cache_data(show_spinner=False)
def _read_csv_cached(
    path_string: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    del modified_time_ns

    return pd.read_csv(path_string)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Stage 4 dashboard file not found: {path}"
        )

    return _read_csv_cached(
        path_string=str(path),
        modified_time_ns=path.stat().st_mtime_ns,
    )


def stage4_solution_directory(
    model_scope: str,
    equity_measure_type: str,
    lambda_tag: str,
) -> Path:
    if model_scope == "full_instance":
        return (
            STAGE4_RESULTS_DIR
            / model_scope
            / equity_measure_type
            / lambda_tag
        )

    return (
        STAGE4_RESULTS_DIR
        / model_scope
        / lambda_tag
    )


def stage3_repaired_solution_directory(
    model_scope: str,
    equity_measure_type: str,
    lambda_tag: str,
) -> Path:
    if model_scope == "full_instance":
        return (
            STAGE3_REPAIRED_RESULTS_DIR
            / model_scope
            / equity_measure_type
            / lambda_tag
        )

    return (
        STAGE3_REPAIRED_RESULTS_DIR
        / model_scope
        / lambda_tag
    )


def _as_bool(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def _safe_pct(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.astype(float)

    return numerator.astype(float).where(
        denominator.abs() > 1e-9
    ) / denominator.where(
        denominator.abs() > 1e-9
    ) * 100.0


def add_stage4_derived_columns(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()

    result["lambda_equity"] = (
        result["lambda_equity"].astype(float)
    )

    result["duration_improvement_minutes"] = (
        -result["delta_total_route_duration"].astype(float)
    )

    result["duration_improvement_pct"] = _safe_pct(
        result["duration_improvement_minutes"],
        result["baseline_total_route_duration"],
    )

    result["selected_equity_improvement_value"] = (
        -result["delta_selected_equity_metric_value"].astype(float)
    )

    result["selected_equity_improvement_pct"] = _safe_pct(
        result["selected_equity_improvement_value"],
        result["baseline_selected_equity_metric_value"],
    )

    result["max_workload_improvement_minutes"] = (
        -result["delta_max_driver_route_workload"].astype(float)
    )

    result["cv_improvement"] = (
        -result["delta_cv_active_route_workload"].astype(float)
    )

    if "all_routes_feasible" in result.columns:
        result["all_routes_feasible_bool"] = _as_bool(
            result["all_routes_feasible"]
        )
    else:
        result["all_routes_feasible_bool"] = (
            result["n_infeasible_routes"].astype(int).eq(0)
        )

    result["duration_gain_flag"] = (
        result["duration_improvement_minutes"] > 1e-9
    )

    return result


def load_stage4_master_summary() -> pd.DataFrame:
    """
    Load Stage 4 master data.

    Prefer the individual solution summaries, because they reflect the
    latest committed folders. Fall back to stage_4_master_summary.csv if
    needed.
    """

    rows = []

    for path in sorted(
        (
            STAGE4_RESULTS_DIR
            / "full_instance"
        ).glob("*/lambda_*/stage_4_solution_summary.csv")
    ):
        rows.append(
            _load_csv(path)
        )

    if rows:
        summary = pd.concat(
            rows,
            ignore_index=True,
        )
    else:
        summary = _load_csv(
            STAGE4_RESULTS_DIR
            / "stage_4_master_summary.csv"
        )

    summary = add_stage4_derived_columns(
        summary
    )

    summary["equity_measure_order"] = (
        summary["equity_measure_type"]
        .apply(
            lambda value: (
                EQUITY_MEASURE_TYPES.index(value)
                if value in EQUITY_MEASURE_TYPES
                else 999
            )
        )
    )

    return (
        summary
        .sort_values(
            [
                "equity_measure_order",
                "lambda_equity",
            ]
        )
        .drop(columns=["equity_measure_order"])
        .reset_index(drop=True)
    )


def load_stage4_solution_files(
    model_scope: str,
    equity_measure_type: str,
    lambda_tag: str,
) -> dict[str, pd.DataFrame | dict]:
    root = stage4_solution_directory(
        model_scope=model_scope,
        equity_measure_type=equity_measure_type,
        lambda_tag=lambda_tag,
    )

    result: dict[str, pd.DataFrame | dict] = {
        "stage4_summary": _load_csv(
            root / "stage_4_solution_summary.csv"
        ),
        "stage4_assignments": _load_csv(
            root / "stage_4_assignments.csv"
        ),
        "route_summaries": _load_csv(
            root / "route_summaries.csv"
        ),
        "route_stops": _load_csv(
            root / "route_stops.csv"
        ),
        "driver_route_workloads": _load_csv(
            root / "driver_route_workloads.csv"
        ),
        "alns_iteration_log": _load_csv(
            root / "alns_iteration_log.csv"
        ),
        "alns_operator_weights": _load_csv(
            root / "alns_operator_weights.csv"
        ),
    }

    manifest_path = root / "stage_4_manifest.json"

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            result["manifest"] = json.load(handle)
    else:
        result["manifest"] = {}

    return result


def load_stage3_baseline_files(
    model_scope: str,
    equity_measure_type: str,
    lambda_tag: str,
) -> dict[str, pd.DataFrame]:
    root = stage3_repaired_solution_directory(
        model_scope=model_scope,
        equity_measure_type=equity_measure_type,
        lambda_tag=lambda_tag,
    )

    result = {
        "baseline_summary": _load_csv(
            root / "stage_3_solution_summary.csv"
        ),
        "baseline_route_summaries": _load_csv(
            root / "route_summaries.csv"
        ),
        "baseline_route_stops": _load_csv(
            root / "route_stops.csv"
        ),
        "baseline_driver_route_workloads": _load_csv(
            root / "driver_route_workloads.csv"
        ),
    }

    assignments_path = root / "repaired_assignments.csv"

    if assignments_path.exists():
        result["baseline_assignments"] = _load_csv(
            assignments_path
        )
    else:
        result["baseline_assignments"] = pd.DataFrame()

    return result


def build_assignment_change_table(
    baseline_assignments: pd.DataFrame,
    stage4_assignments: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_assignments.empty or stage4_assignments.empty:
        return pd.DataFrame()

    baseline = baseline_assignments[
        [
            "orderid",
            "day_index",
            "weekday",
            "driver_id",
            "driver_name",
        ]
    ].rename(
        columns={
            "day_index": "baseline_day_index",
            "weekday": "baseline_weekday",
            "driver_id": "baseline_driver_id",
            "driver_name": "baseline_driver_name",
        }
    )

    stage4 = stage4_assignments[
        [
            "orderid",
            "day_index",
            "weekday",
            "driver_id",
            "driver_name",
        ]
    ].rename(
        columns={
            "day_index": "stage4_day_index",
            "weekday": "stage4_weekday",
            "driver_id": "stage4_driver_id",
            "driver_name": "stage4_driver_name",
        }
    )

    merged = baseline.merge(
        stage4,
        on="orderid",
        how="inner",
    )

    merged["driver_changed"] = (
        merged["baseline_driver_id"]
        != merged["stage4_driver_id"]
    )

    merged["day_changed"] = (
        merged["baseline_day_index"]
        != merged["stage4_day_index"]
    )

    return (
        merged[
            merged["driver_changed"]
            | merged["day_changed"]
        ]
        .sort_values("orderid")
        .reset_index(drop=True)
    )


def build_driver_workload_comparison(
    baseline_driver_workloads: pd.DataFrame,
    stage4_driver_workloads: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_driver_workloads[
        [
            "driver_id",
            "driver_name",
            "total_route_duration",
            "n_orders",
            "n_driver_days",
            "horizon_utilisation_pct",
        ]
    ].rename(
        columns={
            "total_route_duration": "baseline_route_duration",
            "n_orders": "baseline_n_orders",
            "n_driver_days": "baseline_n_driver_days",
            "horizon_utilisation_pct": "baseline_utilisation_pct",
        }
    )

    stage4 = stage4_driver_workloads[
        [
            "driver_id",
            "driver_name",
            "total_route_duration",
            "n_orders",
            "n_driver_days",
            "horizon_utilisation_pct",
        ]
    ].rename(
        columns={
            "total_route_duration": "stage4_route_duration",
            "n_orders": "stage4_n_orders",
            "n_driver_days": "stage4_n_driver_days",
            "horizon_utilisation_pct": "stage4_utilisation_pct",
        }
    )

    comparison = baseline.merge(
        stage4,
        on=[
            "driver_id",
            "driver_name",
        ],
        how="outer",
    ).fillna(0)

    comparison["delta_route_duration"] = (
        comparison["stage4_route_duration"]
        - comparison["baseline_route_duration"]
    )

    comparison["route_duration_improvement"] = (
        -comparison["delta_route_duration"]
    )

    comparison["delta_n_orders"] = (
        comparison["stage4_n_orders"]
        - comparison["baseline_n_orders"]
    )

    return comparison.sort_values(
        "delta_route_duration"
    ).reset_index(drop=True)