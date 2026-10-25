import json
from pathlib import Path

import pandas as pd

from src.assignment_costs import build_assignment_cost_matrix
from src.assignment_model import solve_assignment_model
from src.config import (
    DATA_PROCESSED_DIR,
    PLANNING_DAYS,
    EQUITY_MEASURE_LABELS,
)
from src.experiment_metrics import (
    create_day_workload_summary,
    create_driver_workload_summary,
    create_solution_summary,
    validate_solution,
)
from src.experiment_paths import (
    FULL_LAMBDA_PATHS,
    LambdaExperimentPaths,
)
from src.equity_measures import (
    compute_workload_equity_metrics,
    equity_metric_value_for_measure,
    validate_equity_measure_type,
    workload_series_for_all_drivers,
)


def lambda_tag(lambda_value: float) -> str:
    """
    Example:
        0.05 -> lambda_0_050
        1.00 -> lambda_1_000
    """

    return (
        f"lambda_{lambda_value:.3f}"
        .replace(".", "_")
    )


def create_result_directories(
    paths: LambdaExperimentPaths,
) -> None:

    for path in [
        paths.results_dir,
        paths.assignments_dir,
        paths.driver_workloads_dir,
        paths.day_workloads_dir,
        paths.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_stage_2_inputs():

    orders = pd.read_csv(
        DATA_PROCESSED_DIR / "orders_clean.csv"
    )

    drivers = pd.read_csv(
        DATA_PROCESSED_DIR / "drivers_clean.csv"
    )

    routes = pd.read_csv(
        DATA_PROCESSED_DIR / "routes_clean.csv"
    )

    feasibility = pd.read_csv(
        DATA_PROCESSED_DIR / "feasibility_matrix.csv"
    )

    cost_path = (
        DATA_PROCESSED_DIR / "assignment_cost_matrix.csv"
    )

    if cost_path.exists():
        print("Using existing assignment cost matrix.")
        costs = pd.read_csv(cost_path)

    else:
        print("Building assignment cost matrix once.")

        costs = build_assignment_cost_matrix(
            orders=orders,
            drivers=drivers,
            routes=routes,
        )

        costs.to_csv(cost_path, index=False)

    return orders, drivers, feasibility, costs


def update_summary_file(new_row: dict, paths: LambdaExperimentPaths,) -> pd.DataFrame:

    new_data = pd.DataFrame([new_row])

    if paths.summary_path.exists():

        summary = pd.read_csv(paths.summary_path)

        summary = summary[
            summary["lambda_tag"]
            != new_row["lambda_tag"]
        ]

        summary = pd.concat(
            [summary, new_data],
            ignore_index=True,
        )

    else:
        summary = new_data

    summary = (
        summary
        .sort_values("lambda_equity")
        .reset_index(drop=True)
    )

    baseline = summary.loc[
        summary["lambda_equity"] == 0
    ]

    if not baseline.empty:

        baseline_cost = float(
            baseline.iloc[0][
                "total_estimated_assignment_time"
            ]
        )

        baseline_max = float(
            baseline.iloc[0][
                "max_driver_workload"
            ]
        )

        summary["efficiency_cost_increase_pct"] = (
            100
            * (
                summary["total_estimated_assignment_time"]
                - baseline_cost
            )
            / baseline_cost
        )

        summary["max_workload_reduction_pct"] = (
            100
            * (
                baseline_max
                - summary["max_driver_workload"]
            )
            / baseline_max
        )

    summary["duplicate_solution"] = (
        summary.duplicated(
            subset=["solution_hash"],
            keep="first",
        )
    )

    # Determine Pareto efficiency using total cost and max workload.
    pareto_flags = []

    for index, candidate in summary.iterrows():

        dominated = False

        for other_index, other in summary.iterrows():

            if index == other_index:
                continue

            no_worse = (
                other["total_estimated_assignment_time"]
                <= candidate[
                    "total_estimated_assignment_time"
                ]
                and
                other["max_driver_workload"]
                <= candidate["max_driver_workload"]
            )

            strictly_better = (
                other["total_estimated_assignment_time"]
                < candidate[
                    "total_estimated_assignment_time"
                ]
                or
                other["max_driver_workload"]
                < candidate["max_driver_workload"]
            )

            if no_worse and strictly_better:
                dominated = True
                break

        pareto_flags.append(not dominated)

    summary["pareto_efficient"] = pareto_flags

    summary.to_csv(
        paths.summary_path,
        index=False,
    )

    return summary


def save_solution(
    assignments: pd.DataFrame,
    drivers: pd.DataFrame,
    orders: pd.DataFrame,
    feasibility: pd.DataFrame,
    model_info: dict,
    paths: LambdaExperimentPaths,
    ) -> dict:

    lambda_value = float(
        model_info["lambda_equity"]
    )

    tag = lambda_tag(lambda_value)

    assignments = assignments.copy()
    assignments.insert(0, "lambda_tag", tag)
    assignments.insert(1, "lambda_equity", lambda_value)

    driver_summary = create_driver_workload_summary(
        assignments=assignments,
        drivers=drivers,
        planning_days=PLANNING_DAYS,
    )

    day_summary = create_day_workload_summary(
        assignments
    )

    validation = validate_solution(
        assignments=assignments,
        orders=orders,
        drivers=drivers,
        feasibility_matrix=feasibility,
    )

    summary_row = create_solution_summary(
        assignments=assignments,
        driver_summary=driver_summary,
        model_info=model_info,
        validation=validation,
    )

    summary_row["lambda_tag"] = tag

    assignments.to_csv(
        paths.assignments_dir / f"{tag}.csv",
        index=False,
    )

    driver_summary.to_csv(
        paths.driver_workloads_dir / f"{tag}.csv",
        index=False,
    )

    day_summary.to_csv(
        paths.day_workloads_dir / f"{tag}.csv",
        index=False,
    )

    update_summary_file(new_row=summary_row, paths=paths,)

    return summary_row

def _lambda_key(lambda_value: float) -> float:
    """
    Return a stable key for matching floating-point lambda values.
    """

    return round(float(lambda_value), 3)


def _truthy(value) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def _stored_summary_row_is_usable(row: pd.Series) -> bool:
    status = (
        str(row.get("status", ""))
        .strip()
        .lower()
    )

    solution_status = (
        str(row.get("solution_status", ""))
        .strip()
        .lower()
    )

    if "no solution" in solution_status:
        return False

    if "no feasible" in solution_status:
        return False

    if "infeasible" in status:
        return False

    if "solver_solution_usable" in row.index:
        if not _truthy(
            row.get("solver_solution_usable")
        ):
            return False

    return _truthy(
        row.get("solution_valid", False)
    )


def prepare_anchor_warm_starts(
    anchor_assignments: dict[float, pd.DataFrame] | None,
    orders: pd.DataFrame,
    feasibility: pd.DataFrame,
    costs: pd.DataFrame,
    ) -> dict[float, pd.DataFrame]:
    """
    Validate and prepare full-instance assignment solutions for use as
    warm starts in a reduced candidate-graph experiment.

    Each anchor must:

    - assign every order exactly once;
    - use a feasible planning day;
    - use an order-driver arc retained in the supplied cost matrix.
    """

    if not anchor_assignments:
        return {}

    required_columns = {
        "orderid",
        "day_index",
        "driver_id",
    }

    expected_orders = set(
        orders["orderid"].astype(int)
    )

    feasibility_data = feasibility.copy()

    feasibility_mask = (
        feasibility_data["is_feasible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    feasible_order_day_pairs = set(
        map(
            tuple,
            feasibility_data.loc[
                feasibility_mask,
                ["orderid", "day_index"],
            ]
            .astype(int)
            .to_numpy(),
        )
    )

    available_order_driver_pairs = set(
        map(
            tuple,
            costs[
                ["orderid", "driver_id"]
            ]
            .astype(int)
            .to_numpy(),
        )
    )

    prepared: dict[float, pd.DataFrame] = {}

    for lambda_value, assignments in anchor_assignments.items():

        key = _lambda_key(lambda_value)

        if assignments is None or assignments.empty:
            raise ValueError(
                f"The lambda={key:.3f} anchor is empty."
            )

        missing_columns = (
            required_columns
            - set(assignments.columns)
        )

        if missing_columns:
            raise ValueError(
                f"The lambda={key:.3f} anchor is missing columns: "
                f"{sorted(missing_columns)}"
            )

        anchor = assignments.copy()

        anchor["orderid"] = (
            anchor["orderid"].astype(int)
        )

        anchor["day_index"] = (
            anchor["day_index"].astype(int)
        )

        anchor["driver_id"] = (
            anchor["driver_id"].astype(int)
        )

        if len(anchor) != len(expected_orders):
            raise ValueError(
                f"The lambda={key:.3f} anchor contains "
                f"{len(anchor):,} rows; expected "
                f"{len(expected_orders):,}."
            )

        if anchor["orderid"].duplicated().any():
            duplicate_orders = (
                anchor.loc[
                    anchor["orderid"].duplicated(
                        keep=False
                    ),
                    "orderid",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"The lambda={key:.3f} anchor assigns some orders "
                f"more than once. Examples: {duplicate_orders[:10]}"
            )

        anchor_orders = set(
            anchor["orderid"]
        )

        if anchor_orders != expected_orders:
            missing_orders = sorted(
                expected_orders - anchor_orders
            )

            unexpected_orders = sorted(
                anchor_orders - expected_orders
            )

            raise ValueError(
                f"The lambda={key:.3f} anchor does not match the "
                "experiment's orders. "
                f"Missing examples: {missing_orders[:10]}; "
                f"unexpected examples: {unexpected_orders[:10]}."
            )

        anchor_order_day_pairs = set(
            map(
                tuple,
                anchor[
                    ["orderid", "day_index"]
                ]
                .to_numpy(),
            )
        )

        invalid_day_pairs = (
            anchor_order_day_pairs
            - feasible_order_day_pairs
        )

        if invalid_day_pairs:
            raise ValueError(
                f"The lambda={key:.3f} anchor contains "
                f"{len(invalid_day_pairs)} infeasible order-day "
                "assignments."
            )

        anchor_order_driver_pairs = set(
            map(
                tuple,
                anchor[
                    ["orderid", "driver_id"]
                ]
                .to_numpy(),
            )
        )

        missing_cost_pairs = (
            anchor_order_driver_pairs
            - available_order_driver_pairs
        )

        if missing_cost_pairs:
            raise ValueError(
                f"The lambda={key:.3f} anchor uses "
                f"{len(missing_cost_pairs)} order-driver pairs "
                "that are not present in the reduced candidate graph."
            )

        prepared[key] = anchor

        print(
            f"Validated warm-start anchor for lambda={key:.3f}: "
            f"{len(anchor):,} assignments.",
            flush=True,
        )

    return prepared

def run_lambda_experiment(
    lambda_values: list[float],
    time_limit_seconds: int | None = 900,
    relative_gap: float = 0.01,
    resume: bool = True,
    solver_name: str = "cbc",
    solver_threads: int | None = None,
    paths: LambdaExperimentPaths = FULL_LAMBDA_PATHS,
    orders: pd.DataFrame | None = None,
    drivers: pd.DataFrame | None = None,
    feasibility: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
    manifest_metadata: dict | None = None,
    anchor_assignments: dict[float, pd.DataFrame] | None = None,
    anchor_source_labels: dict[float, str] | None = None,
    relative_gap_by_lambda: dict[float, float] | None = None,
    equity_measure_type: str = "maximum",
) -> None:
    """
    Run a lambda sensitivity experiment.

    The function can run either:

    1. the standard full-instance experiment, using the default
       processed datasets and FULL_LAMBDA_PATHS; or

    2. a separate experiment, such as the reduced candidate-graph
       model, by supplying:
           - orders
           - drivers
           - feasibility
           - costs
           - an alternative LambdaExperimentPaths object.

    Parameters
    ----------
    lambda_values:
        Lambda values to solve. Lambda zero is automatically added
        when it is not supplied.

    time_limit_seconds:
        Solver time limit for each lambda value.

    relative_gap:
        Relative MIP-gap target supplied to the solver.

    resume:
        When True, previously stored solutions in the selected
        experiment folder are skipped.

    solver_name:
        Either "cbc" or "highs".

    solver_threads:
        Optional number of solver threads.

    paths:
        Output paths for this experiment. Defaults to the original
        full-instance lambda experiment paths.

    orders, drivers, feasibility, costs:
        Optional preloaded experiment inputs. Either all four must be
        supplied or none of them must be supplied.

    manifest_metadata:
        Optional additional fields to save in the experiment manifest,
        such as reduced-instance details and candidate-graph settings.
    """

    
    # VALIDATE AND NORMALISE THE SOLVER CONFIGURATION
    

    solver_name = solver_name.strip().lower()

    if solver_name not in {"cbc", "highs", "gurobi"}:
        raise ValueError(
            "solver_name must be either 'cbc', 'highs' or 'gurobi'."
        )

    if (
            time_limit_seconds is not None
            and time_limit_seconds <= 0
        ):
            raise ValueError(
                "time_limit_seconds must be positive "
                "when supplied."
            )

    if relative_gap < 0:
        raise ValueError(
            "relative_gap cannot be negative."
        )

    if solver_threads is not None and solver_threads <= 0:
        raise ValueError(
            "solver_threads must be positive when supplied."
        )
    
    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
    )

    
    # CREATE THE SELECTED EXPERIMENT OUTPUT DIRECTORIES
    

    create_result_directories(paths)

    
    # LOAD OR VALIDATE THE EXPERIMENT INPUTS
    

    supplied_inputs = [
        orders,
        drivers,
        feasibility,
        costs,
    ]

    if all(value is None for value in supplied_inputs):

        # Default behaviour: run the original full-instance experiment.
        orders, drivers, feasibility, costs = (
            load_stage_2_inputs()
        )

    elif any(value is None for value in supplied_inputs):

        raise ValueError(
            "Either provide all four experiment inputs "
            "(orders, drivers, feasibility and costs), "
            "or provide none of them."
        )

    # These assertions help static type checkers and confirm that
    # all four datasets are now available.
    assert orders is not None
    assert drivers is not None
    assert feasibility is not None
    assert costs is not None

    if orders.empty:
        raise ValueError(
            "The orders dataset is empty."
        )

    if drivers.empty:
        raise ValueError(
            "The drivers dataset is empty."
        )

    if feasibility.empty:
        raise ValueError(
            "The feasibility dataset is empty."
        )

    if costs.empty:
        raise ValueError(
            "The assignment-cost dataset is empty."
        )

    # Work with local copies so the caller's DataFrames are not
    # modified by the experiment.
    orders = orders.copy()
    drivers = drivers.copy()
    feasibility = feasibility.copy()
    costs = costs.copy()

    
    # PREPARE VALIDATED ANCHOR WARM STARTS
    

    prepared_anchors = prepare_anchor_warm_starts(
        anchor_assignments=anchor_assignments,
        orders=orders,
        feasibility=feasibility,
        costs=costs,
    )

    prepared_anchor_labels = {}

    if anchor_source_labels:
        prepared_anchor_labels = {
            _lambda_key(lambda_value): str(label)
            for lambda_value, label
            in anchor_source_labels.items()
        }

    
    # PREPARE OPTIONAL PER-LAMBDA GAP TARGETS
    

    gap_overrides: dict[float, float] = {}

    if relative_gap_by_lambda:

        for lambda_value, gap_value in (
            relative_gap_by_lambda.items()
        ):

            lambda_key = _lambda_key(
                lambda_value
            )

            gap_value = float(gap_value)

            if gap_value < 0 or gap_value > 1:
                raise ValueError(
                    "Every relative-gap override must lie "
                    f"between 0 and 1. Invalid setting: "
                    f"lambda={lambda_key:.3f}, gap={gap_value}."
                )

            gap_overrides[lambda_key] = (
                gap_value
            )


    def gap_for_lambda(
        lambda_value: float,
    ) -> float:
        """
        Return the configured gap for one lambda value.
        """

        return gap_overrides.get(
            _lambda_key(lambda_value),
            relative_gap,
        )

    
    # NORMALISE THE REQUESTED LAMBDA GRID
    

    lambda_values = sorted(
        set(float(value) for value in lambda_values)
    )

    if not lambda_values:
        raise ValueError(
            "At least one lambda value must be supplied."
        )

    invalid_lambda_values = [
        value
        for value in lambda_values
        if value < 0 or value > 1
    ]

    if invalid_lambda_values:
        raise ValueError(
            "All lambda values must lie between 0 and 1. "
            f"Invalid values: {invalid_lambda_values}"
        )

    if 0.0 not in lambda_values:
        lambda_values.insert(0, 0.0)

    
    # IDENTIFY PREVIOUSLY COMPLETED SOLUTIONS
    

    completed_tags: set[str] = set()

    if resume and paths.summary_path.exists():

        existing_summary = pd.read_csv(
            paths.summary_path
        )

        if (
            not existing_summary.empty
            and "lambda_tag" in existing_summary.columns
        ):
            usable_completed_rows = existing_summary.loc[
                existing_summary.apply(
                    _stored_summary_row_is_usable,
                    axis=1,
                )
            ]

            completed_tags = set(
                usable_completed_rows["lambda_tag"]
                .dropna()
                .astype(str)
            )

    
    # CREATE OR LOAD THE NORMALISATION REFERENCES
    

    if paths.manifest_path.exists():

        with open(
            paths.manifest_path,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if "cost_reference" not in manifest:
            raise KeyError(
                "The existing experiment manifest does not contain "
                "'cost_reference'."
            )

        if equity_measure_type == "gini":
            equity_reference = 1.0

        elif "equity_reference" in manifest:
            equity_reference = float(
                manifest["equity_reference"]
            )

        elif "workload_reference" in manifest:
            # Support manifests using the original workload reference field.
            equity_reference = float(
                manifest["workload_reference"]
            )

        else:
            raise KeyError(
                "The existing experiment manifest does not contain "
                "'equity_reference' or legacy 'workload_reference'."
            )

        workload_reference = equity_reference

        cost_reference = float(
            manifest["cost_reference"]
        )

        if cost_reference <= 0:
            raise ValueError(
                "The stored cost_reference must be positive."
            )

        if equity_reference <= 0:
            raise ValueError(
                "The stored equity_reference must be positive."
            )

        print(
            "\nUsing existing experiment normalisation references:"
            f"\n  Cost reference: {cost_reference:,.2f}"
            f"\n  Equity reference: {equity_reference:,.2f}"
            f"\n  Manifest: {paths.manifest_path}"
        )

    else:

        print(
            "\nSolving efficiency-only baseline: lambda = 0.000 "
            f"using {solver_name.upper()}"
        )

        baseline_tag = lambda_tag(0.0)

        baseline_anchor = (
            prepared_anchors.get(
                _lambda_key(0.0)
            )
        )

        baseline_gap = gap_for_lambda(
            0.0
        )

        if baseline_anchor is not None:
            baseline_warm_start_source = (
                "full_instance_anchor_lambda_0.000"
            )
        else:
            baseline_warm_start_source = "none"

        print(
            "  Warm-start source: "
            f"{baseline_warm_start_source}"
            f"\n  Relative-gap target: "
            f"{100 * baseline_gap:.1f}%",
            flush=True,
        )

        baseline_assignments, baseline_info = (
            solve_assignment_model(
                orders=orders,
                drivers=drivers,
                feasibility_matrix=feasibility,
                assignment_cost_matrix=costs,
                lambda_equity=0.0,
                time_limit_seconds=time_limit_seconds,
                objective_mode="normalized",
                cost_reference=1.0,
                relative_gap=baseline_gap,
                log_path=(
                    paths.logs_dir
                    / f"{baseline_tag}_{solver_name}.log"
                ),
                initial_assignments=baseline_anchor,
                solver_name=solver_name,
                solver_threads=solver_threads,
                equity_measure_type=equity_measure_type,
                equity_reference=1.0,
                workload_reference=1.0,
            )
        )

        driver_ids = list(
            range(1, len(drivers.reset_index(drop=True)) + 1)
        )

        baseline_workloads = workload_series_for_all_drivers(
            assignments=baseline_assignments,
            driver_ids=driver_ids,
        )

        baseline_equity_metrics = compute_workload_equity_metrics(
            baseline_workloads
        )

        if equity_measure_type == "gini":
            equity_reference = 1.0
        else:
            equity_reference = equity_metric_value_for_measure(
                metrics=baseline_equity_metrics,
                equity_measure_type=equity_measure_type,
            )

            if equity_reference <= 0:
                equity_reference = 1.0

        baseline_info["equity_measure_type"] = equity_measure_type
        baseline_info["equity_reference"] = equity_reference
        baseline_info["workload_reference"] = equity_reference

        baseline_info["warm_start_source"] = (
            baseline_warm_start_source
        )

        baseline_info["warm_start_lambda"] = (
            0.0
            if baseline_anchor is not None
            else None
        )

        if baseline_assignments.empty:
            raise RuntimeError(
                "The efficiency-only baseline produced no usable "
                "assignment."
            )

        cost_reference = float(
            baseline_assignments[
                "estimated_assignment_time"
            ].sum()
        )

        workload_reference = equity_reference

        if cost_reference <= 0:
            raise RuntimeError(
                "The efficiency baseline produced a non-positive "
                "cost reference."
            )

        if equity_reference <= 0:
            raise RuntimeError(
                "The efficiency baseline produced a non-positive "
                "equity reference."
            )

        # Replace the temporary reference values used during the
        # baseline solve with the final experiment references.
        baseline_info["cost_reference"] = (
            cost_reference
        )

        baseline_info["equity_reference"] = (
            equity_reference
        )

        baseline_info["workload_reference"] = (
            equity_reference
        )

        baseline_info["equity_measure_type"] = (
            equity_measure_type
        )

        # At lambda zero, the normalised baseline objective is
        # approximately one. This replaces the objective value produced
        # when temporary references of one were used.
        baseline_normalized_cost = (
            float(
                baseline_info[
                    "total_assignment_time"
                ]
            )
            / cost_reference
        )

        baseline_info["objective_value"] = (
            baseline_normalized_cost
            + 0.000001 * baseline_normalized_cost
        )

        save_solution(
            assignments=baseline_assignments,
            drivers=drivers,
            orders=orders,
            feasibility=feasibility,
            model_info=baseline_info,
            paths=paths,
        )

        manifest = {
            "objective_mode": "normalized",
            "equity_measure_type": equity_measure_type,
            "cost_reference": cost_reference,
            "equity_reference": equity_reference,
            "workload_reference": equity_reference,
            "time_limit_seconds": time_limit_seconds,
            "relative_gap_target": relative_gap,
            "lambda_values_requested": lambda_values,
            "solver_name": solver_name,
            "solver_threads": solver_threads,
            "n_orders": int(len(orders)),
            "n_drivers": int(len(drivers)),
            "n_feasibility_rows": int(
                len(feasibility)
            ),
            "n_cost_arcs": int(len(costs)),
            "anchor_warm_start_lambdas": sorted(
                float(value)
                for value in prepared_anchors.keys()
            ),
            "relative_gap_by_lambda": {
                f"{lambda_value:.3f}": float(gap_value)
                for lambda_value, gap_value
                in sorted(gap_overrides.items())
            },
        }

        if manifest_metadata:
            manifest.update(
                manifest_metadata
            )

        with open(
            paths.manifest_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )

        completed_tags.add(
            baseline_tag
        )

        print(
            "\nEfficiency-only baseline saved:"
            f"\n  Total estimated time: {cost_reference:,.2f}"
            f"\n  Selected equity reference "
            f"({equity_measure_type}): {equity_reference:,.2f}",
            flush=True,
        )

    
    # UPDATE THE MANIFEST FOR THE CURRENT RUN
    

    # This ensures that a resumed experiment records the latest
    # requested lambda grid and any supplied metadata without changing
    # the fixed normalisation references.
    with open(
        paths.manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        current_manifest = json.load(file)

    current_manifest.update({
        "objective_mode": "normalized",
        "equity_measure_type": equity_measure_type,
        "cost_reference": cost_reference,
        "equity_reference": equity_reference,
        "workload_reference": equity_reference,
        "time_limit_seconds": time_limit_seconds,
        "relative_gap_target": relative_gap,
        "lambda_values_requested": lambda_values,
        "solver_name": solver_name,
        "solver_threads": solver_threads,
        "n_orders": int(len(orders)),
        "n_drivers": int(len(drivers)),
        "n_feasibility_rows": int(
            len(feasibility)
        ),
        "n_cost_arcs": int(len(costs)),
        "anchor_warm_start_lambdas": sorted(
            float(value)
            for value in prepared_anchors.keys()
        ),
        "relative_gap_by_lambda": {
            f"{lambda_value:.3f}": float(gap_value)
            for lambda_value, gap_value
            in sorted(gap_overrides.items())
        },
    })

    if manifest_metadata:
        current_manifest.update(
            manifest_metadata
        )

    with open(
        paths.manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            current_manifest,
            file,
            indent=2,
        )

    
    # SOLVE THE REQUESTED LAMBDA VALUES
    

    previous_assignments: pd.DataFrame | None = None
    previous_solution_tag: str | None = None

    for lambda_value in lambda_values:

        tag = lambda_tag(lambda_value)

        if resume and tag in completed_tags:

            print(
                f"Skipping completed solution: {tag}"
            )

            assignment_path = (
                paths.assignments_dir
                / f"{tag}.csv"
            )

            if assignment_path.exists():
                previous_assignments = pd.read_csv(
                    assignment_path
                )
                previous_solution_tag = tag
            else:
                print(
                    "Warning: the summary marks this solution as "
                    "completed, but its assignment file was not found: "
                    f"{assignment_path}"
                )

            continue

        lambda_key = _lambda_key(
            lambda_value
        )

        matching_anchor = (
            prepared_anchors.get(
                lambda_key
            )
        )

        if matching_anchor is not None:

            initial_assignments_for_solve = (
                matching_anchor
            )

            warm_start_source = prepared_anchor_labels.get(
                lambda_key,
                (
                    "full_instance_anchor_"
                    f"lambda_{lambda_key:.3f}"
                ),
            )

            warm_start_lambda = lambda_key

        elif previous_assignments is not None:

            initial_assignments_for_solve = (
                previous_assignments
            )

            warm_start_source = (
                "previous_reduced_solution:"
                f"{previous_solution_tag}"
            )

            warm_start_lambda = None

        else:

            initial_assignments_for_solve = None
            warm_start_source = "none"
            warm_start_lambda = None

        current_gap = gap_for_lambda(
            lambda_value
        )

        print(
            f"\nSolving lambda = {lambda_value:.3f} "
            f"using {solver_name.upper()}"
            f"\n  Warm-start source: {warm_start_source}"
            f"\n  Relative-gap target: "
            f"{100 * current_gap:.1f}%",
            flush=True,
        )

        assignments, model_info = (
            solve_assignment_model(
                orders=orders,
                drivers=drivers,
                feasibility_matrix=feasibility,
                assignment_cost_matrix=costs,
                lambda_equity=lambda_value,
                time_limit_seconds=time_limit_seconds,
                objective_mode="normalized",
                cost_reference=cost_reference,
                workload_reference=equity_reference,
                equity_reference=equity_reference,
                equity_measure_type=equity_measure_type,
                relative_gap=current_gap,
                log_path=(
                    paths.logs_dir
                    / f"{tag}_{solver_name}.log"
                ),
                initial_assignments=(initial_assignments_for_solve),
                solver_name=solver_name,
                solver_threads=solver_threads,
            )
        )

        model_info["warm_start_source"] = (
            warm_start_source
        )

        model_info["warm_start_lambda"] = (
            warm_start_lambda
        )

        if assignments.empty:
            print(
                f"No usable solution stored for {tag}. "
                f"Solver status: {model_info.get('status', 'Unknown')}"
            )
            continue

        result = save_solution(
            assignments=assignments,
            drivers=drivers,
            orders=orders,
            feasibility=feasibility,
            model_info=model_info,
            paths=paths,
        )

        # The current assignment becomes the warm start for the next
        # lambda value. CBC uses this directly. Other solvers may ignore
        # it depending on their PuLP interface.
        previous_assignments = (
            assignments.copy()
        )

        previous_solution_tag = tag

        completed_tags.add(tag)

        equity_label = EQUITY_MEASURE_LABELS.get(
            equity_measure_type,
            equity_measure_type,
        )

        equity_value = model_info.get(
            "equity_penalty_value",
            model_info.get("max_driver_workload"),
        )

        print(
            f"Saved {tag}: "
            f"solver={solver_name.upper()}, "
            f"cost={model_info['total_assignment_time']:.2f}, "
            f"{equity_label}={equity_value:.2f}, "
            f"status={model_info['status']}",
            flush=True,
        )

    
    # FINAL EXPERIMENT MESSAGE
    

    print(
        "\nLambda experiment completed."
        f"\nSummary saved to: {paths.summary_path}"
        f"\nManifest saved to: {paths.manifest_path}"
    )