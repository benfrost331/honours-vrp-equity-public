from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.assignment_costs import build_route_lookup
from src.config import (
    DATA_PROCESSED_DIR,
    STAGE3_EXACT_ROUTE_LIMIT,
    STAGE3_REPRESENTATIVE_SOLUTIONS,
    STAGE3_RESULTS_DIR,
)
from src.experiment_paths import (
    FULL_LAMBDA_PATHS,
    REDUCED_LAMBDA_PATHS,
    make_equity_experiment_paths,
)
from src.routing_engine import (
    build_route_for_driver_day,
    prepare_assignments_for_routing,
    prepare_driver_table,
)


def get_assignment_path(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
) -> Path:
    if model_scope == "full_instance":

        paths = make_equity_experiment_paths(
            equity_measure_type=equity_measure_type,
            model_scope="full_instance",
        )

        equity_path = paths.assignments_dir / f"{lambda_tag}.csv"

        if equity_path.exists():
            return equity_path

        if equity_measure_type == "maximum":
            return (
                FULL_LAMBDA_PATHS.assignments_dir
                / f"{lambda_tag}.csv"
            )

        return equity_path

    if model_scope == "reduced_candidate_graph":
        return (
            REDUCED_LAMBDA_PATHS.assignments_dir
            / f"{lambda_tag}.csv"
        )

    raise ValueError(
        f"Unknown model_scope: {model_scope}"
    )


def get_summary_path(
    model_scope: str,
    equity_measure_type: str = "maximum",
) -> Path:
    if model_scope == "full_instance":

        paths = make_equity_experiment_paths(
            equity_measure_type=equity_measure_type,
            model_scope="full_instance",
        )

        equity_summary_path = paths.summary_path

        if equity_summary_path.exists():
            return equity_summary_path

        if equity_measure_type == "maximum":
            return FULL_LAMBDA_PATHS.summary_path

        return equity_summary_path

    if model_scope == "reduced_candidate_graph":
        return REDUCED_LAMBDA_PATHS.summary_path

    raise ValueError(
        f"Unknown model_scope: {model_scope}"
    )


def load_solution_metadata(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
) -> dict:
    summary_path = get_summary_path(
        model_scope=model_scope,
        equity_measure_type=equity_measure_type,
    )

    summary = pd.read_csv(summary_path)

    row = summary.loc[
        summary["lambda_tag"].astype(str) == lambda_tag
    ]

    if row.empty:
        raise ValueError(
            f"No summary row found for {model_scope} "
            f"{equity_measure_type} {lambda_tag}."
        )

    return row.iloc[0].to_dict()


def stage_3_solution_dir(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
) -> Path:
    if model_scope == "full_instance":
        return (
            STAGE3_RESULTS_DIR
            / model_scope
            / equity_measure_type
            / lambda_tag
        )

    return (
        STAGE3_RESULTS_DIR
        / model_scope
        / lambda_tag
    )


def create_driver_route_workload_summary(
    route_summaries: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise routed workload by driver across the planning horizon.
    """

    drivers_prepared = prepare_driver_table(
        drivers
    )

    driver_base = drivers_prepared[
        [
            "driver_id",
            "driver.name",
            "locationid",
            "workingdays",
            "shift_length",
        ]
    ].rename(
        columns={
            "driver.name": "driver_name",
            "locationid": "driver_locationid",
        }
    )

    if route_summaries.empty:

        aggregated = pd.DataFrame(
            columns=[
                "driver_id",
                "n_driver_days",
                "n_orders",
                "total_route_duration",
                "total_travel_time",
                "total_service_time",
                "total_waiting_time",
                "total_violation_minutes",
                "n_infeasible_driver_days",
            ]
        )

    else:

        aggregated = (
            route_summaries
            .groupby("driver_id", as_index=False)
            .agg(
                n_driver_days=("day_index", "nunique"),
                n_orders=("n_orders", "sum"),
                total_route_duration=("route_duration", "sum"),
                total_travel_time=("total_travel_time", "sum"),
                total_service_time=("total_service_time", "sum"),
                total_waiting_time=("total_waiting_time", "sum"),
                total_violation_minutes=(
                    "total_violation_minutes",
                    "sum",
                ),
                n_infeasible_driver_days=(
                    "route_feasible",
                    lambda values: int((~values.astype(bool)).sum()),
                ),
            )
        )

    driver_summary = driver_base.merge(
        aggregated,
        on="driver_id",
        how="left",
    )

    fill_zero_columns = [
        "n_driver_days",
        "n_orders",
        "total_route_duration",
        "total_travel_time",
        "total_service_time",
        "total_waiting_time",
        "total_violation_minutes",
        "n_infeasible_driver_days",
    ]

    for column in fill_zero_columns:
        driver_summary[column] = (
            driver_summary[column]
            .fillna(0)
        )

    driver_summary["is_active"] = (
        driver_summary["n_orders"] > 0
    )

    driver_summary["horizon_capacity"] = (
        driver_summary["workingdays"]
        * driver_summary["shift_length"]
    )

    driver_summary["horizon_utilisation_pct"] = (
        100
        * driver_summary["total_route_duration"]
        / driver_summary["horizon_capacity"]
    )

    return driver_summary


def create_stage_3_solution_summary(
    model_scope: str,
    lambda_tag: str,
    metadata: dict,
    assignments: pd.DataFrame,
    route_summaries: pd.DataFrame,
    driver_route_workloads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a one-row routed-solution summary.
    """

    active_drivers = driver_route_workloads.loc[
        driver_route_workloads["is_active"]
    ].copy()

    total_route_duration = float(
        route_summaries["route_duration"].sum()
    )

    total_travel_time = float(
        route_summaries["total_travel_time"].sum()
    )

    total_service_time = float(
        route_summaries["total_service_time"].sum()
    )

    total_waiting_time = float(
        route_summaries["total_waiting_time"].sum()
    )

    total_violation_minutes = float(
        route_summaries["total_violation_minutes"].sum()
    )

    assignment_proxy_total = float(
        assignments["estimated_assignment_time"].sum()
    )

    if active_drivers.empty:

        max_driver_route_workload = 0.0
        mean_active_route_workload = 0.0
        std_active_route_workload = 0.0
        cv_active_route_workload = 0.0
        n_active_drivers = 0

    else:

        workloads = active_drivers[
            "total_route_duration"
        ]

        max_driver_route_workload = float(
            workloads.max()
        )

        mean_active_route_workload = float(
            workloads.mean()
        )

        std_active_route_workload = float(
            workloads.std(ddof=1)
        )

        if pd.isna(std_active_route_workload):
            std_active_route_workload = 0.0

        cv_active_route_workload = (
            std_active_route_workload
            / mean_active_route_workload
            if mean_active_route_workload > 0
            else 0.0
        )

        n_active_drivers = int(
            active_drivers.shape[0]
        )

    summary = pd.DataFrame([{
        "model_scope": model_scope,
        "lambda_tag": lambda_tag,
        "lambda_equity": metadata.get("lambda_equity"),
        "assignment_solution_hash": metadata.get("solution_hash"),
        "assignment_status": metadata.get("status"),
        "assignment_proven_optimal": (
            bool(metadata.get("proven_optimal"))
            if pd.notna(metadata.get("proven_optimal"))
            else str(metadata.get("status", "")).strip().lower() == "optimal"
        ),
        "assignment_total_estimated_time": assignment_proxy_total,
        "assignment_max_driver_workload": metadata.get("max_driver_workload"),
        "assignment_workload_cv_active": metadata.get("workload_cv_active"),
        "n_orders": int(assignments["orderid"].nunique()),
        "n_driver_day_routes": int(route_summaries.shape[0]),
        "n_active_drivers": n_active_drivers,
        "n_feasible_routes": int(route_summaries["route_feasible"].sum()),
        "n_infeasible_routes": int((~route_summaries["route_feasible"]).sum()),
        "total_route_duration": total_route_duration,
        "total_travel_time": total_travel_time,
        "total_service_time": total_service_time,
        "total_waiting_time": total_waiting_time,
        "total_violation_minutes": total_violation_minutes,
        "max_driver_route_workload": max_driver_route_workload,
        "mean_active_route_workload": mean_active_route_workload,
        "std_active_route_workload": std_active_route_workload,
        "cv_active_route_workload": cv_active_route_workload,
        "route_duration_vs_assignment_proxy_pct": (
            100
            * (
                total_route_duration
                / assignment_proxy_total
                - 1
            )
            if assignment_proxy_total > 0
            else pd.NA
        ),
        "all_routes_feasible": bool(
            total_violation_minutes <= 1e-6
        ),
        "exact_route_limit": STAGE3_EXACT_ROUTE_LIMIT,
    }])

    return summary


def route_stored_assignment_solution(
    model_scope: str,
    lambda_tag: str,
    overwrite: bool = True,
    equity_measure_type: str = "maximum",
) -> pd.DataFrame:
    """
    Route one stored Stage 2 assignment solution.
    """

    solution_dir = stage_3_solution_dir(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    summary_path = (
        solution_dir
        / "stage_3_solution_summary.csv"
    )

    if summary_path.exists() and not overwrite:
        return pd.read_csv(
            summary_path
        )

    solution_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    orders = pd.read_csv(
        DATA_PROCESSED_DIR / "orders_clean.csv"
    )

    drivers = pd.read_csv(
        DATA_PROCESSED_DIR / "drivers_clean.csv"
    )

    routes = pd.read_csv(
        DATA_PROCESSED_DIR / "routes_clean.csv"
    )

    assignment_path = get_assignment_path(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    if not assignment_path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {assignment_path}"
        )

    assignments = pd.read_csv(
        assignment_path
    )

    metadata = load_solution_metadata(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    routing_assignments = prepare_assignments_for_routing(
        assignments=assignments,
        orders=orders,
        drivers=drivers,
    )

    route_lookup = build_route_lookup(
        routes
    )

    route_summary_rows = []
    stop_tables = []

    grouped = routing_assignments.groupby(
        [
            "day_index",
            "weekday",
            "driver_id",
            "driver_name",
        ],
        sort=True,
    )

    print(
        f"Routing {model_scope} {lambda_tag}: "
        f"{len(grouped):,} driver-day groups"
    )

    for (
        day_index,
        weekday,
        driver_id,
        driver_name,
    ), group in grouped:

        route_id = (
            f"{model_scope}_{lambda_tag}"
            f"_d{int(day_index):02d}"
            f"_k{int(driver_id):03d}"
        )

        route_summary, route_stops = build_route_for_driver_day(
            group=group,
            route_lookup=route_lookup,
            exact_route_limit=STAGE3_EXACT_ROUTE_LIMIT,
        )

        route_summary.update({
            "model_scope": model_scope,
            "lambda_tag": lambda_tag,
            "lambda_equity": metadata.get("lambda_equity"),
            "equity_measure_type": equity_measure_type,
            "route_id": route_id,
        })

        route_stops = route_stops.copy()

        route_stops.insert(
            0,
            "route_id",
            route_id,
        )

        route_stops.insert(
            0,
            "lambda_tag",
            lambda_tag,
        )

        route_stops.insert(
            2,
            "equity_measure_type",
            equity_measure_type,
        )

        route_stops.insert(
            0,
            "model_scope",
            model_scope,
        )

        route_summary_rows.append(
            route_summary
        )

        stop_tables.append(
            route_stops
        )

    route_summaries = pd.DataFrame(
        route_summary_rows
    )

    route_stops = (
        pd.concat(
            stop_tables,
            ignore_index=True,
        )
        if stop_tables
        else pd.DataFrame()
    )

    driver_route_workloads = (
        create_driver_route_workload_summary(
            route_summaries=route_summaries,
            drivers=drivers,
        )
    )

    solution_summary = (
        create_stage_3_solution_summary(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            metadata=metadata,
            assignments=assignments,
            route_summaries=route_summaries,
            driver_route_workloads=driver_route_workloads,
        )
    )

    solution_summary["equity_measure_type"] = equity_measure_type

    route_summaries.to_csv(
        solution_dir / "route_summaries.csv",
        index=False,
    )

    route_stops.to_csv(
        solution_dir / "route_stops.csv",
        index=False,
    )

    driver_route_workloads.to_csv(
        solution_dir / "driver_route_workloads.csv",
        index=False,
    )

    solution_summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_scope": model_scope,
        "lambda_tag": lambda_tag,
        "lambda_equity": metadata.get("lambda_equity"),
        "assignment_solution_hash": metadata.get("solution_hash"),
        "routing_layer": "decomposed_driver_day",
        "construction_methods": sorted(
            route_summaries[
                "construction_method"
            ].unique().tolist()
        ),
        "exact_route_limit": STAGE3_EXACT_ROUTE_LIMIT,
        "n_driver_day_routes": int(
            route_summaries.shape[0]
        ),
        "n_orders": int(
            assignments["orderid"].nunique()
        ),
        "equity_measure_type": equity_measure_type,
    }

    with open(
        solution_dir / "stage_3_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        solution_summary[
            [
                "model_scope",
                "lambda_tag",
                "n_driver_day_routes",
                "n_infeasible_routes",
                "total_route_duration",
                "max_driver_route_workload",
                "cv_active_route_workload",
            ]
        ].to_string(index=False)
    )

    return solution_summary


def run_stage_3_representative_solutions(
    representative_solutions: list[tuple[str, str]] | None = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Route the representative Stage 2 lambda solutions.
    """

    if representative_solutions is None:
        representative_solutions = (
            STAGE3_REPRESENTATIVE_SOLUTIONS
        )

    STAGE3_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for model_scope, lambda_tag in representative_solutions:

        summary = route_stored_assignment_solution(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            overwrite=overwrite,
        )

        summaries.append(
            summary
        )

    master_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    master_summary.to_csv(
        STAGE3_RESULTS_DIR
        / "stage_3_master_summary.csv",
        index=False,
    )

    print(
        "\nStage 3 representative routing completed."
    )

    print(
        master_summary[
            [
                "model_scope",
                "lambda_tag",
                "n_driver_day_routes",
                "n_infeasible_routes",
                "total_route_duration",
                "max_driver_route_workload",
                "cv_active_route_workload",
                "route_duration_vs_assignment_proxy_pct",
            ]
        ].to_string(index=False)
    )

    return master_summary