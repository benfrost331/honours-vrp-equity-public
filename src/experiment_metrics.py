import hashlib

import numpy as np
import pandas as pd

from src.equity_measures import (
    compute_workload_equity_metrics,
    equity_metric_value_for_measure,
)



# CREATE A HASH FOR EACH COMPLETE ASSIGNMENT


def create_solution_hash(assignments: pd.DataFrame) -> str:
    """
    Create a reproducible identifier for a complete assignment solution.

    Lambda values that produce the same order-day-driver assignments
    will therefore have the same solution hash.
    """

    canonical = (
        assignments
        .sort_values(["orderid", "day_index", "driver_id"])
        [["orderid", "day_index", "driver_id"]]
        .to_csv(index=False)
        .encode("utf-8")
    )

    return hashlib.sha256(canonical).hexdigest()



# DRIVER WORKLOAD SUMMARY


def create_driver_workload_summary(
    assignments: pd.DataFrame,
    drivers: pd.DataFrame,
    planning_days: int,
) -> pd.DataFrame:
    """
    Create workload metrics for every available driver.

    Drivers receiving no orders are retained with zero workload.
    """

    driver_master = drivers.copy().reset_index(drop=True)

    driver_master["driver_id"] = driver_master.index + 1

    driver_master = driver_master.rename(
        columns={"driver.name": "driver_name"}
    )

    aggregated = (
        assignments
        .groupby("driver_id", as_index=False)
        .agg(
            n_orders=("orderid", "count"),
            total_job_time=("jobtime", "sum"),
            total_outbound_time=("outbound_time", "sum"),
            total_return_time=("return_time", "sum"),
            total_estimated_assignment_time=(
                "estimated_assignment_time",
                "sum",
            ),
        )
    )

    summary = (
        driver_master[
            ["driver_id", "driver_name", "shift_length"]
        ]
        .merge(
            aggregated,
            on="driver_id",
            how="left",
        )
    )

    numeric_columns = [
        "n_orders",
        "total_job_time",
        "total_outbound_time",
        "total_return_time",
        "total_estimated_assignment_time",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns].fillna(0)
    )

    summary["n_orders"] = summary["n_orders"].astype(int)

    summary["is_active"] = summary["n_orders"] > 0

    summary["horizon_capacity"] = (
        summary["shift_length"] * planning_days
    )

    summary["horizon_utilisation_pct"] = np.where(
        summary["horizon_capacity"] > 0,
        (
            100
            * summary["total_estimated_assignment_time"]
            / summary["horizon_capacity"]
        ),
        0.0,
    )

    return summary



# DAY WORKLOAD SUMMARY


def create_day_workload_summary(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise assigned workload for each planning day.
    """

    data = assignments.copy()

    data["travel_time"] = (
        data["outbound_time"]
        + data["return_time"]
    )

    summary = (
        data
        .groupby(
            ["day_index", "weekday"],
            as_index=False,
        )
        .agg(
            n_orders=("orderid", "count"),
            n_active_drivers=("driver_id", "nunique"),
            total_job_time=("jobtime", "sum"),
            total_travel_time=("travel_time", "sum"),
            total_estimated_assignment_time=(
                "estimated_assignment_time",
                "sum",
            ),
        )
        .sort_values("day_index")
        .reset_index(drop=True)
    )

    return summary



# SOLUTION VALIDATION


def validate_solution(
    assignments: pd.DataFrame,
    orders: pd.DataFrame,
    drivers: pd.DataFrame,
    feasibility_matrix: pd.DataFrame,
    tolerance: float = 1e-5,
) -> dict:
    """
    Confirm that a stored solution satisfies the main Stage 2 rules.
    """

    expected_orders = set(
        orders["orderid"].astype(int)
    )

    assigned_orders = set(
        assignments["orderid"].astype(int)
    )

    one_assignment_per_order = (
        len(assignments)
        == assignments["orderid"].nunique()
    )

    all_orders_assigned = (
        assigned_orders == expected_orders
    )

    feasibility_data = feasibility_matrix.copy()

    feasibility_data["orderid"] = (
        feasibility_data["orderid"].astype(int)
    )

    feasibility_data["day_index"] = (
        feasibility_data["day_index"].astype(int)
    )

    feasibility_check = (
        assignments[["orderid", "day_index"]]
        .merge(
            feasibility_data[
                ["orderid", "day_index", "is_feasible"]
            ],
            on=["orderid", "day_index"],
            how="left",
        )
    )

    all_days_feasible = bool(
        feasibility_check["is_feasible"]
        .fillna(False)
        .all()
    )

    driver_master = drivers.copy().reset_index(drop=True)

    driver_master["driver_id"] = driver_master.index + 1

    daily_workload = (
        assignments
        .groupby(
            ["day_index", "driver_id"],
            as_index=False,
        )
        ["estimated_assignment_time"]
        .sum()
        .merge(
            driver_master[
                ["driver_id", "shift_length"]
            ],
            on="driver_id",
            how="left",
        )
    )

    daily_workload["capacity_violation"] = (
        daily_workload["estimated_assignment_time"]
        - daily_workload["shift_length"]
    )

    if daily_workload.empty:
        max_capacity_violation = 0.0
    else:
        max_capacity_violation = max(
            0.0,
            float(
                daily_workload[
                    "capacity_violation"
                ].max()
            ),
        )

    capacity_respected = (
        max_capacity_violation <= tolerance
    )

    solution_valid = all(
        [
            all_orders_assigned,
            one_assignment_per_order,
            all_days_feasible,
            capacity_respected,
        ]
    )

    return {
        "all_orders_assigned": all_orders_assigned,
        "one_assignment_per_order": one_assignment_per_order,
        "all_days_feasible": all_days_feasible,
        "capacity_respected": capacity_respected,
        "max_capacity_violation": max_capacity_violation,
        "solution_valid": solution_valid,
    }



# COMPLETE SOLUTION SUMMARY


def create_solution_summary(
    assignments: pd.DataFrame,
    driver_summary: pd.DataFrame,
    model_info: dict,
    validation: dict,
) -> dict:
    """
    Create the summary metrics stored for each lambda value.
    """

    active_drivers = driver_summary[
        driver_summary["is_active"]
    ].copy()

    all_workloads = (
        driver_summary[
            "total_estimated_assignment_time"
        ]
    )

    active_workloads = (
        active_drivers[
            "total_estimated_assignment_time"
        ]
    )

    if active_workloads.empty:
        min_active_workload = 0.0
        mean_active_workload = 0.0
        max_active_workload = 0.0
        std_active_workload = 0.0
    else:
        min_active_workload = float(
            active_workloads.min()
        )

        mean_active_workload = float(
            active_workloads.mean()
        )

        max_active_workload = float(
            active_workloads.max()
        )

        std_active_workload = float(
            active_workloads.std(ddof=0)
        )

    if mean_active_workload > 0:
        coefficient_of_variation = (
            std_active_workload
            / mean_active_workload
        )
    else:
        coefficient_of_variation = 0.0

    all_driver_equity_metrics = compute_workload_equity_metrics(
        all_workloads
    )

    equity_measure_type = model_info.get(
        "equity_measure_type",
        "maximum",
    )

    if (
        equity_measure_type == "gini"
        and model_info.get("equity_penalty_value") is not None
    ):
        equity_penalty_metric_value = float(
            model_info["equity_penalty_value"]
        )
    else:
        equity_penalty_metric_value = (
            equity_metric_value_for_measure(
                metrics=all_driver_equity_metrics,
                equity_measure_type=equity_measure_type,
            )
        )

    solver_solution_usable = bool(
        model_info.get(
            "solver_solution_usable",
            model_info.get("proven_optimal", False),
        )
    )

    combined_validation = dict(
        validation
    )

    combined_validation["solution_valid"] = bool(
        combined_validation.get("solution_valid", False)
        and solver_solution_usable
    )

    return {
        "lambda_equity": model_info["lambda_equity"],
        "status": model_info["status"],
        "solver_name": model_info.get("solver_name", "unknown",),
        "solver_threads": model_info.get("solver_threads",),
        "warm_start_source": model_info.get(
            "warm_start_source",
            "unknown",
        ),
        "warm_start_lambda": model_info.get(
            "warm_start_lambda",
        ),
        "objective_value": model_info["objective_value"],
        "objective_mode": model_info["objective_mode"],
        "cost_reference": model_info["cost_reference"],
        "workload_reference": model_info.get(
            "workload_reference",
            model_info.get("equity_reference"),
        ),
        "equity_reference": model_info.get(
            "equity_reference",
            model_info.get("workload_reference"),
        ),
        "solve_runtime_seconds": model_info[
            "solve_runtime_seconds"
        ],
        "time_limit_seconds": model_info[
            "time_limit_seconds"
        ],
        "relative_gap_target": model_info[
            "relative_gap_target"
        ],
        "n_variables": model_info["n_variables"],
        "n_assignments": len(assignments),
        "n_drivers_available": len(driver_summary),
        "n_drivers_used": int(
            driver_summary["is_active"].sum()
        ),
        "total_estimated_assignment_time": float(
            assignments[
                "estimated_assignment_time"
            ].sum()
        ),
        "max_driver_workload": float(
            all_workloads.max()
        ),
        "min_all_driver_workload": float(
            all_workloads.min()
        ),
        "mean_all_driver_workload": float(
            all_workloads.mean()
        ),
        "workload_std_all": float(
            all_workloads.std(ddof=0)
        ),
        "min_active_driver_workload":
            min_active_workload,
        "mean_active_driver_workload":
            mean_active_workload,
        "workload_std_active":
            std_active_workload,
        "workload_cv_active":
            coefficient_of_variation,
        "active_workload_range": (
            max_active_workload
            - min_active_workload
        ),
        "solution_hash":
            create_solution_hash(assignments),
        "solution_status": model_info.get(
            "solution_status",
            "Unknown",
        ),
        "proven_optimal": model_info.get(
            "proven_optimal",
            False,
        ),
        "solver_solution_usable": solver_solution_usable,
        **combined_validation,
        "equity_measure_type": equity_measure_type,
        "equity_penalty_metric_value": equity_penalty_metric_value,

        "workload_range": all_driver_equity_metrics["workload_range"],
        "workload_mad_mean": all_driver_equity_metrics["workload_mad_mean"],
        "workload_minmax_deviation": all_driver_equity_metrics["workload_minmax_deviation"],
        "workload_gini_pairwise_sum": all_driver_equity_metrics["workload_gini_pairwise_sum"],
        "workload_gini_coefficient": all_driver_equity_metrics["workload_gini_coefficient"],
        "workload_gini_fixed_reference_coefficient": (
            model_info.get(
                "workload_gini_fixed_reference_coefficient"
            )
        ),
    }