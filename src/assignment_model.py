from collections import defaultdict
from pathlib import Path
from time import perf_counter
import pandas as pd
import pulp
from src.equity_measures import (
    add_equity_penalty_to_model,
    compute_workload_equity_metrics,
    equity_metric_value_for_measure,
    validate_equity_measure_type,
    workload_series_for_all_drivers,
)

def solver_returned_usable_solution(
    status: str,
    solution_status: str,
    model_status_code,
    model_solution_status_code,
) -> bool:
    """
    Decide whether PuLP/the underlying solver returned a usable incumbent.

    This prevents variable values from being extracted when the solver
    explicitly reports 'No Solution Exists'. That case can otherwise leave
    stale or partial variable values that pass basic assignment checks but
    are not defensible solver outputs.
    """

    status_text = (
        str(status)
        .strip()
        .lower()
    )

    solution_status_text = (
        str(solution_status)
        .strip()
        .lower()
    )

    if "no solution" in solution_status_text:
        return False

    if "no feasible" in solution_status_text:
        return False

    if "infeasible" in status_text:
        return False

    if model_status_code == pulp.LpStatusOptimal:
        return True

    if model_solution_status_code == pulp.LpSolutionOptimal:
        return True

    integer_feasible_code = getattr(
        pulp,
        "LpSolutionIntegerFeasible",
        None,
    )

    if (
        integer_feasible_code is not None
        and model_solution_status_code == integer_feasible_code
    ):
        return True

    if "integer feasible" in solution_status_text:
        return True

    if (
        "solution found" in solution_status_text
        and "no solution" not in solution_status_text
    ):
        return True

    return False


def solve_assignment_model(
    orders: pd.DataFrame,
    drivers: pd.DataFrame,
    feasibility_matrix: pd.DataFrame,
    assignment_cost_matrix: pd.DataFrame,
    lambda_equity: float = 0.10,
    time_limit_seconds: int | None = 300,
    objective_mode: str = "normalized",
    cost_reference: float | None = None,
    workload_reference: float | None = None,
    equity_reference: float | None = None,
    equity_measure_type: str = "maximum",
    relative_gap: float = 0.0,
    log_path: Path | None = None,
    show_solver_output: bool = False,
    initial_assignments: pd.DataFrame | None = None,
    solver_name: str = "cbc",
    solver_threads: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Solve the order-day-driver assignment model.

    objective_mode:
        "raw":
            C + lambda * M

        "normalized":
            (1-lambda)*(C/C_ref)
            + lambda*(M/M_ref)
            + small cost tie-break
    """

    if not 0 <= lambda_equity <= 1:
        raise ValueError("lambda_equity must lie between 0 and 1.")
    
    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
    )

    if equity_reference is None:
        equity_reference = workload_reference

    orders = orders.copy()
    drivers = drivers.copy().reset_index(drop=True)
    feasibility_matrix = feasibility_matrix.copy()
    assignment_cost_matrix = assignment_cost_matrix.copy()

    orders["orderid"] = orders["orderid"].astype(int)
    drivers["driver_id"] = drivers.index + 1

    order_ids = orders["orderid"].tolist()
    day_ids = sorted(
        feasibility_matrix["day_index"].astype(int).unique().tolist()
    )
    driver_ids = drivers["driver_id"].tolist()

    feasible_pairs = feasibility_matrix.loc[
        feasibility_matrix["is_feasible"].astype(bool),
        ["orderid", "day_index"],
    ].copy()

    feasible_pairs["orderid"] = feasible_pairs["orderid"].astype(int)
    feasible_pairs["day_index"] = feasible_pairs["day_index"].astype(int)

    feasible_order_day = set(
        zip(feasible_pairs["orderid"], feasible_pairs["day_index"])
    )

    valid_costs = assignment_cost_matrix.dropna(
        subset=["estimated_assignment_time"]
    ).copy()

    valid_costs["orderid"] = valid_costs["orderid"].astype(int)
    valid_costs["driver_id"] = valid_costs["driver_id"].astype(int)

    cost_lookup = {
        (int(row.orderid), int(row.driver_id)):
            float(row.estimated_assignment_time)
        for row in valid_costs.itertuples()
    }

    cost_detail_lookup = (
        valid_costs
        .set_index(["orderid", "driver_id"])
        .to_dict(orient="index")
    )

    weekday_lookup = (
        feasibility_matrix
        .drop_duplicates(["orderid", "day_index"])
        .assign(
            orderid=lambda df: df["orderid"].astype(int),
            day_index=lambda df: df["day_index"].astype(int),
        )
        .set_index(["orderid", "day_index"])["weekday"]
        .to_dict()
    )

    shift_lookup = {
        int(row.driver_id): float(row.shift_length)
        for row in drivers.itertuples()
    }

    model = pulp.LpProblem(
        "stage_2_order_day_driver_assignment",
        pulp.LpMinimize,
    )

    x: dict[tuple[int, int, int], pulp.LpVariable] = {}

    variables_by_order = defaultdict(list)
    workload_by_day_driver = defaultdict(list)
    workload_by_driver = defaultdict(list)
    all_cost_terms = []

    # PREPARE OPTIONAL WARM START
    
    initial_assignment_keys: set[tuple[int, int, int]] = set()

    if (
        initial_assignments is not None
        and not initial_assignments.empty
    ):
        initial_assignment_keys = set(
            zip(
                initial_assignments["orderid"].astype(int),
                initial_assignments["day_index"].astype(int),
                initial_assignments["driver_id"].astype(int),
            )
    )

    # CREATE VARIABLES AND INDEX THEM WHILE THEY ARE CREATED
    
    for order_id in order_ids:
        for day_id in day_ids:

            if (order_id, day_id) not in feasible_order_day:
                continue

            for driver_id in driver_ids:

                cost_key = (order_id, driver_id)

                if cost_key not in cost_lookup:
                    continue

                variable = pulp.LpVariable(
                    f"x_o{order_id}_d{day_id}_k{driver_id}",
                    lowBound=0,
                    upBound=1,
                    cat="Binary",
                )

                key = (order_id, day_id, driver_id)
                x[key] = variable

                if initial_assignment_keys:
                    variable.setInitialValue(1 if key in initial_assignment_keys else 0)

                cost_term = cost_lookup[cost_key] * variable

                variables_by_order[order_id].append(variable)
                workload_by_day_driver[(day_id, driver_id)].append(cost_term)
                workload_by_driver[driver_id].append(cost_term)
                all_cost_terms.append(cost_term)

    # EACH ORDER MUST BE ASSIGNED EXACTLY ONCE

    for order_id in order_ids:

        order_variables = variables_by_order[order_id]

        if not order_variables:
            raise ValueError(
                f"Order {order_id} has no feasible assignment variables."
            )

        model += (
            pulp.lpSum(order_variables) == 1,
            f"assign_order_{order_id}_once",
        )

    # DAILY DRIVER SHIFT CAPACITY
    
    for day_id in day_ids:
        for driver_id in driver_ids:

            model += (
                pulp.lpSum(
                    workload_by_day_driver[(day_id, driver_id)]
                )
                <= shift_lookup[driver_id],
                f"capacity_d{day_id}_k{driver_id}",
            )

    # MAXIMUM HORIZON WORKLOAD

    total_assignment_time = pulp.lpSum(all_cost_terms)

    include_equity_penalty_in_model = (
        float(lambda_equity) > 1e-9
    )

    if include_equity_penalty_in_model:

        gini_total_workload_reference = None

        if equity_measure_type == "gini":
            if cost_reference is None or cost_reference <= 0:
                raise ValueError(
                    "The normalised Gini formulation requires a "
                    "positive cost_reference."
                )

            gini_total_workload_reference = float(cost_reference)

        equity_penalty, equity_auxiliaries = add_equity_penalty_to_model(
            model=model,
            workload_by_driver=workload_by_driver,
            total_assignment_time=total_assignment_time,
            driver_ids=driver_ids,
            equity_measure_type=equity_measure_type,
            gini_total_workload_reference=(
                gini_total_workload_reference
            ),
        )

    else:

        equity_penalty = 0.0

        equity_auxiliaries = {
            "workload_expr": {
                driver_id: pulp.lpSum(
                    workload_by_driver[driver_id]
                )
                for driver_id in driver_ids
            }
        }

    # OBJECTIVE

    if objective_mode == "raw":

        if equity_measure_type != "maximum":
            raise ValueError(
                "Raw objective mode is only supported for the legacy "
                "maximum-workload model. Use objective_mode='normalized' "
                "for multi-measure equity experiments."
            )

        objective = (
            total_assignment_time
            + lambda_equity * equity_penalty
        )

    elif objective_mode == "normalized":

        if cost_reference is None or cost_reference <= 0:
            raise ValueError(
                "A positive cost_reference is required."
            )

        if equity_reference is None or equity_reference <= 0:
            raise ValueError(
                "A positive equity_reference is required."
            )

        normalized_cost = total_assignment_time / cost_reference
        normalized_equity = equity_penalty / equity_reference

        tie_break = 0.000001 * normalized_cost

        objective = (
            (1 - lambda_equity) * normalized_cost
            + lambda_equity * normalized_equity
            + tie_break
        )

    else:
        raise ValueError(
            "objective_mode must be 'raw' or 'normalized'."
        )

    model += objective

    # SELECT AND CONFIGURE SOLVER

    solver_name_normalized = solver_name.strip().lower()

    if log_path is not None:
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    if solver_name_normalized == "cbc":

        solver_kwargs = {
            "msg": show_solver_output,
            "timeLimit": time_limit_seconds,
            "gapRel": relative_gap,
            "warmStart": bool(initial_assignment_keys),
        }

        if time_limit_seconds is not None:
            solver_kwargs["timeLimit"] = (
                time_limit_seconds
            )

        if solver_threads is not None:
            solver_kwargs["threads"] = solver_threads

        if log_path is not None:
            solver_kwargs["logPath"] = str(log_path)

        solver = pulp.PULP_CBC_CMD(
            **solver_kwargs
        )

    elif solver_name_normalized == "highs":

        if initial_assignment_keys:
            print("Note: the direct HiGHS interface is running without the PuLP warm start.", flush=True,)

        solver_kwargs = {
            "mip": True,
            "msg": True,
            "timeLimit": time_limit_seconds,
            "gapRel": relative_gap,
            "output_flag": True,
            "log_to_console": True,
            "mip_min_logging_interval": 5.0,
        }

        if solver_threads is not None:
            solver_kwargs["threads"] = solver_threads

        if log_path is not None:
            log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            solver_kwargs["log_file"] = str(log_path)

        solver = pulp.HiGHS(
            **solver_kwargs
        )

    elif solver_name_normalized == "gurobi":

        if initial_assignment_keys:
            print(
                "Using PuLP/Gurobi warm start from supplied assignments.",
                flush=True,
            )

        solver_kwargs = {
            "msg": show_solver_output,
            "gapRel": relative_gap,
            "warmStart": bool(initial_assignment_keys),
        }

        if time_limit_seconds is not None:
            solver_kwargs["timeLimit"] = time_limit_seconds

        if solver_threads is not None:
            solver_kwargs["threads"] = solver_threads

        if log_path is not None:
            solver_kwargs["logPath"] = str(log_path)

        solver = pulp.GUROBI(
            **solver_kwargs
        )

    else:

        raise ValueError(
            "solver_name must be either 'cbc', 'highs' or 'gurobi'."
        )

    start_time = perf_counter()
    from datetime import datetime
    print("Calling solver at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True, )
    model.solve(solver)

    status = pulp.LpStatus.get(
        model.status,
        str(model.status),
    )

    solution_status = pulp.LpSolution.get(
        model.sol_status,
        str(model.sol_status),
    )

    solver_solution_usable = solver_returned_usable_solution(
        status=status,
        solution_status=solution_status,
        model_status_code=model.status,
        model_solution_status_code=model.sol_status,
    )

    proven_optimal = (
        model.status == pulp.LpStatusOptimal
        or model.sol_status == pulp.LpSolutionOptimal
    )

    print("Solver returned at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True, )
    solve_runtime = perf_counter() - start_time

    # EXTRACT ASSIGNMENTS
    
    assignment_rows = []

    if not solver_solution_usable:

        print(
            "Solver did not return a usable integer incumbent. "
            "No assignment solution will be extracted.",
            flush=True,
        )

    else:

        for (order_id, day_id, driver_id), variable in x.items():

            if variable.value() is None or variable.value() <= 0.5:
                continue

            detail = cost_detail_lookup[
                (order_id, driver_id)
            ]

            assignment_rows.append({
                "orderid": order_id,
                "day_index": day_id,
                "weekday": weekday_lookup[
                    (order_id, day_id)
                ],
                "driver_id": driver_id,
                "driver_name": detail["driver_name"],
                "order_locationid": detail["locationid"],
                "driver_locationid": detail["driver_locationid"],
                "jobtime": detail["jobtime"],
                "outbound_time": detail["outbound_time"],
                "return_time": detail["return_time"],
                "estimated_assignment_time":
                    detail["estimated_assignment_time"],
            })

    assignments = pd.DataFrame(assignment_rows)

    if assignments.empty:
        actual_total_time = None
        workload_metrics = {}
        actual_equity_penalty_value = None

    else:
        actual_total_time = float(
            assignments["estimated_assignment_time"].sum()
        )

        all_driver_workloads = workload_series_for_all_drivers(
            assignments=assignments,
            driver_ids=driver_ids,
        )

        workload_metrics = compute_workload_equity_metrics(
            all_driver_workloads
        )

        if (
            equity_measure_type == "gini"
            and cost_reference is not None
            and cost_reference > 0
        ):
            workload_metrics[
                "workload_gini_fixed_reference_coefficient"
            ] = (
                workload_metrics["workload_gini_pairwise_sum"]
                / (len(driver_ids) * float(cost_reference))
            )

            actual_equity_penalty_value = float(
                workload_metrics[
                    "workload_gini_fixed_reference_coefficient"
                ]
            )

        else:
            actual_equity_penalty_value = (
                equity_metric_value_for_measure(
                    metrics=workload_metrics,
                    equity_measure_type=equity_measure_type,
                )
            )

    model_info = {
        "status": status,
        "solution_status": solution_status,
        "proven_optimal": proven_optimal,
        "solver_solution_usable": solver_solution_usable,
        "objective_value": pulp.value(model.objective),
        "lambda_equity": lambda_equity,
        "objective_mode": objective_mode,
        "cost_reference": cost_reference,
        "workload_reference": equity_reference,
        "equity_reference": equity_reference,
        "equity_measure_type": equity_measure_type,
        "solver_name": solver_name_normalized,
        "solver_threads": solver_threads,
        "n_variables": len(x),
        "n_assignments": len(assignments),
        "solve_runtime_seconds": solve_runtime,
        "time_limit_seconds": time_limit_seconds,
        "relative_gap_target": relative_gap,
        "total_assignment_time": actual_total_time,
        "equity_penalty_value": actual_equity_penalty_value,
        "solver_equity_penalty_value": (
            pulp.value(equity_penalty)
            if include_equity_penalty_in_model
            else actual_equity_penalty_value
        ),
        "max_driver_workload": workload_metrics.get("max_driver_workload"),
        "min_driver_workload": workload_metrics.get("min_driver_workload"),
        "mean_driver_workload": workload_metrics.get("mean_driver_workload"),
        "workload_range": workload_metrics.get("workload_range"),
        "workload_mad_mean": workload_metrics.get("workload_mad_mean"),
        "workload_minmax_deviation": workload_metrics.get(
            "workload_minmax_deviation"
        ),
        "workload_gini_pairwise_sum": workload_metrics.get(
            "workload_gini_pairwise_sum"
        ),
        "workload_gini_coefficient": workload_metrics.get(
            "workload_gini_coefficient"
        ),
        "workload_gini_fixed_reference_coefficient": (
            workload_metrics.get(
                "workload_gini_fixed_reference_coefficient"
            )
        ),
    }

    return assignments, model_info