from __future__ import annotations

import numpy as np
import pandas as pd
import pulp


EQUITY_MEASURE_TYPES = {
    "maximum",
    "minmax_deviation",
    "mad_mean",
    "range",
    "gini",
}


def validate_equity_measure_type(
    equity_measure_type: str,
) -> str:
    value = str(equity_measure_type).strip().lower()

    if value not in EQUITY_MEASURE_TYPES:
        raise ValueError(
            "Unknown equity_measure_type. Expected one of: "
            f"{sorted(EQUITY_MEASURE_TYPES)}"
        )

    return value


def workload_series_for_all_drivers(
    assignments: pd.DataFrame,
    driver_ids: list[int],
) -> pd.Series:
    totals = (
        assignments
        .groupby("driver_id")["estimated_assignment_time"]
        .sum()
    )

    return (
        pd.Series(index=driver_ids, dtype=float)
        .fillna(0.0)
        .add(totals, fill_value=0.0)
        .astype(float)
    )


def compute_workload_equity_metrics(
    workloads: pd.Series,
) -> dict:
    values = workloads.astype(float).to_numpy()

    n = len(values)

    if n == 0:
        return {
            "max_driver_workload": 0.0,
            "min_driver_workload": 0.0,
            "mean_driver_workload": 0.0,
            "workload_range": 0.0,
            "workload_mad_mean": 0.0,
            "workload_minmax_deviation": 0.0,
            "workload_gini_pairwise_sum": 0.0,
            "workload_gini_coefficient": 0.0,
        }

    mean_value = float(values.mean())
    max_value = float(values.max())
    min_value = float(values.min())

    absolute_deviations = np.abs(
        values - mean_value
    )

    pairwise_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            pairwise_sum += abs(
                float(values[i]) - float(values[j])
            )

    if mean_value > 0:
        gini_coefficient = (
            pairwise_sum
            / (n * n * mean_value)
        )
    else:
        gini_coefficient = 0.0

    return {
        "max_driver_workload": max_value,
        "min_driver_workload": min_value,
        "mean_driver_workload": mean_value,
        "workload_range": max_value - min_value,
        "workload_mad_mean": float(
            absolute_deviations.mean()
        ),
        "workload_minmax_deviation": float(
            absolute_deviations.max()
        ),
        "workload_gini_pairwise_sum": float(
            pairwise_sum
        ),
        "workload_gini_coefficient": float(
            gini_coefficient
        ),
    }


def equity_metric_value_for_measure(
    metrics: dict,
    equity_measure_type: str,
) -> float:
    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
    )

    if equity_measure_type == "maximum":
        return float(metrics["max_driver_workload"])

    if equity_measure_type == "minmax_deviation":
        return float(metrics["workload_minmax_deviation"])

    if equity_measure_type == "mad_mean":
        return float(metrics["workload_mad_mean"])

    if equity_measure_type == "range":
        return float(metrics["workload_range"])

    if equity_measure_type == "gini":
        return float(metrics["workload_gini_coefficient"])

    raise ValueError(equity_measure_type)


def add_equity_penalty_to_model(
    model: pulp.LpProblem,
    workload_by_driver: dict[int, list],
    total_assignment_time,
    driver_ids: list[int],
    equity_measure_type: str,
    gini_total_workload_reference: float | None = None,
):
    """
    Add equity variables and constraints to a PuLP model.

    Returns
    -------
    equity_penalty_expression:
        Linear expression used in the objective.

    equity_auxiliaries:
        Dictionary of created auxiliary variables.
    """

    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
    )

    n_drivers = len(driver_ids)

    use_linked_workload_variables = (
        equity_measure_type == "gini"
    )

    if use_linked_workload_variables:

        workload_expr = {}

        for driver_id in driver_ids:

            workload_variable = pulp.LpVariable(
                f"driver_workload_k{driver_id}",
                lowBound=0,
                cat="Continuous",
            )

            workload_expr[driver_id] = workload_variable

            model += (
                workload_variable
                == pulp.lpSum(
                    workload_by_driver[driver_id]
                ),
                f"define_driver_workload_k{driver_id}",
            )

    else:

        workload_expr = {
            driver_id: pulp.lpSum(
                workload_by_driver[driver_id]
            )
            for driver_id in driver_ids
        }

    auxiliaries = {
        "workload_expr": workload_expr,
    }

    mean_workload = None

    if equity_measure_type in {
        "minmax_deviation",
        "mad_mean",
    }:
        mean_workload = pulp.LpVariable(
            "mean_driver_workload",
            lowBound=0,
            cat="Continuous",
        )

        model += (
            n_drivers * mean_workload
            == total_assignment_time,
            "define_mean_driver_workload",
        )

        auxiliaries["mean_driver_workload"] = mean_workload

    if equity_measure_type == "maximum":

        max_workload = pulp.LpVariable(
            "max_driver_workload",
            lowBound=0,
            cat="Continuous",
        )

        for driver_id in driver_ids:
            model += (
                workload_expr[driver_id] <= max_workload,
                f"max_workload_k{driver_id}",
            )

        auxiliaries["max_driver_workload"] = max_workload

        return max_workload, auxiliaries

    if equity_measure_type == "minmax_deviation":

        max_deviation = pulp.LpVariable(
            "max_absolute_deviation_from_mean",
            lowBound=0,
            cat="Continuous",
        )

        for driver_id in driver_ids:
            model += (
                workload_expr[driver_id] - mean_workload
                <= max_deviation,
                f"minmax_dev_upper_k{driver_id}",
            )

            model += (
                mean_workload - workload_expr[driver_id]
                <= max_deviation,
                f"minmax_dev_lower_k{driver_id}",
            )

        auxiliaries["max_absolute_deviation_from_mean"] = (
            max_deviation
        )

        return max_deviation, auxiliaries

    if equity_measure_type == "mad_mean":

        deviation_variables = {}

        for driver_id in driver_ids:
            deviation = pulp.LpVariable(
                f"absolute_deviation_from_mean_k{driver_id}",
                lowBound=0,
                cat="Continuous",
            )

            deviation_variables[driver_id] = deviation

            model += (
                workload_expr[driver_id] - mean_workload
                <= deviation,
                f"mad_upper_k{driver_id}",
            )

            model += (
                mean_workload - workload_expr[driver_id]
                <= deviation,
                f"mad_lower_k{driver_id}",
            )

        mad_mean = (
            pulp.lpSum(deviation_variables.values())
            / n_drivers
        )

        auxiliaries["absolute_deviation_variables"] = (
            deviation_variables
        )

        return mad_mean, auxiliaries

    if equity_measure_type == "range":

        max_workload = pulp.LpVariable(
            "max_driver_workload",
            lowBound=0,
            cat="Continuous",
        )

        min_workload = pulp.LpVariable(
            "min_driver_workload",
            lowBound=0,
            cat="Continuous",
        )

        for driver_id in driver_ids:
            model += (
                workload_expr[driver_id] <= max_workload,
                f"range_max_k{driver_id}",
            )

            model += (
                workload_expr[driver_id] >= min_workload,
                f"range_min_k{driver_id}",
            )

        workload_range = max_workload - min_workload

        auxiliaries["max_driver_workload"] = max_workload
        auxiliaries["min_driver_workload"] = min_workload

        return workload_range, auxiliaries

    if equity_measure_type == "gini":

        if (
            gini_total_workload_reference is None
            or float(gini_total_workload_reference) <= 0
        ):
            
            raise ValueError(
            "A positive gini_total_workload_reference is "
            "required for the normalised Gini formulation."
            )

        pairwise_abs_diff = {}

        for left_index, left_driver in enumerate(driver_ids):
            for right_driver in driver_ids[left_index + 1:]:

                variable = pulp.LpVariable(
                    f"gini_absdiff_k{left_driver}_k{right_driver}",
                    lowBound=0,
                    cat="Continuous",
                )

                pairwise_abs_diff[
                    (left_driver, right_driver)
                ] = variable

                model += (
                    workload_expr[left_driver]
                    - workload_expr[right_driver]
                    <= variable,
                    f"gini_upper_k{left_driver}_k{right_driver}",
                )

                model += (
                    workload_expr[right_driver]
                    - workload_expr[left_driver]
                    <= variable,
                    f"gini_lower_k{left_driver}_k{right_driver}",
                )

        gini_pairwise_sum = pulp.lpSum(
            pairwise_abs_diff.values()
        )

        gini_denominator = (
            n_drivers
            * float(gini_total_workload_reference)
        )

        normalised_gini_dispersion = (
            gini_pairwise_sum
            / gini_denominator
        )

        auxiliaries["gini_pairwise_abs_diff"] = (
            pairwise_abs_diff
        )

        auxiliaries["gini_denominator"] = (
            gini_denominator
        )

        auxiliaries["gini_pairwise_sum"] = (
            gini_pairwise_sum
        )

        return normalised_gini_dispersion, auxiliaries

    raise ValueError(equity_measure_type)