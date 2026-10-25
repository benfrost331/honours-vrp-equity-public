from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.assignment_costs import build_route_lookup
from src.config import (
    DATA_PROCESSED_DIR,
    STAGE3_EXACT_ROUTE_LIMIT,
    STAGE3_REPAIRED_RESULTS_DIR,
    STAGE3_REPAIR_CANDIDATE_DRIVERS,
    STAGE3_REPAIR_FINAL_EXACT_ROUTE_LIMIT,
    STAGE3_REPAIR_MAX_ITERATIONS,
    STAGE3_REPAIR_MAX_TARGETS_PER_ORDER,
    STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
    STAGE3_REPRESENTATIVE_SOLUTIONS,
    STAGE3_ROUTE_EVAL_CACHE_MAX_SIZE,
    BASE_DIR,
)
from src.stage_3_pipeline import (
    create_driver_route_workload_summary,
    create_stage_3_solution_summary,
    get_assignment_path,
    load_solution_metadata,
    stage_3_solution_dir,
)
from src.routing_engine import (
    build_route_for_driver_day,
    prepare_assignments_for_routing,
    prepare_driver_table,
)


ASSIGNMENT_OUTPUT_COLUMNS = [
    "lambda_tag",
    "lambda_equity",
    "orderid",
    "day_index",
    "weekday",
    "driver_id",
    "driver_name",
    "order_locationid",
    "driver_locationid",
    "jobtime",
    "outbound_time",
    "return_time",
    "estimated_assignment_time",
]

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


def _route_id(
    model_scope: str,
    lambda_tag: str,
    day_index: int,
    driver_id: int,
) -> str:
    return (
        f"{model_scope}_{lambda_tag}"
        f"_d{int(day_index):02d}"
        f"_k{int(driver_id):03d}"
    )

def stage_3_repaired_solution_dir(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
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

def _load_existing_stage_3_routes(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Load existing unrepaired Stage 3 routing outputs if they exist.

    This avoids rerouting the full assignment at the beginning of the
    repair step. The repair layer should start from the already-routed
    Stage 3 solution, then reroute only routes affected by accepted
    repair moves.
    """

    source_dir = stage_3_solution_dir(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    route_summaries_path = source_dir / "route_summaries.csv"
    route_stops_path = source_dir / "route_stops.csv"
    driver_workloads_path = source_dir / "driver_route_workloads.csv"

    required_paths = [
        route_summaries_path,
        route_stops_path,
        driver_workloads_path,
    ]

    if not all(path.exists() for path in required_paths):
        return None

    route_summaries = pd.read_csv(
        route_summaries_path
    )

    route_stops = pd.read_csv(
        route_stops_path
    )

    driver_workloads = pd.read_csv(
        driver_workloads_path
    )

    if "equity_measure_type" not in route_summaries.columns:
        route_summaries["equity_measure_type"] = equity_measure_type

    if "equity_measure_type" not in route_stops.columns:
        route_stops["equity_measure_type"] = equity_measure_type

    return (
        route_summaries,
        route_stops,
        driver_workloads,
    )

def _group_key(
    day_index: int,
    weekday: str,
    driver_id: int,
    driver_name: str,
) -> tuple[int, str, int, str]:
    return (
        int(day_index),
        str(weekday),
        int(driver_id),
        str(driver_name),
    )


def _route_eval_cache_key(
    key: tuple[int, str, int, str],
    group: pd.DataFrame,
    exact_route_limit: int,
) -> tuple:
    order_ids = tuple(
        sorted(
            group["orderid"]
            .astype(int)
            .tolist()
        )
    )

    return (
        int(key[0]),
        str(key[1]),
        int(key[2]),
        order_ids,
        int(exact_route_limit),
    )


def _evaluate_group_cached(
    key: tuple[int, str, int, str],
    group: pd.DataFrame,
    route_lookup: dict,
    route_eval_cache: dict[
        tuple,
        tuple[dict[str, Any], pd.DataFrame],
    ],
    exact_route_limit: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cache_key = _route_eval_cache_key(
        key=key,
        group=group,
        exact_route_limit=exact_route_limit,
    )

    cached = route_eval_cache.get(
        cache_key
    )

    if cached is not None:
        summary, stops = cached

        return (
            summary.copy(),
            stops.copy(deep=True),
        )

    summary, stops = build_route_for_driver_day(
        group=group,
        route_lookup=route_lookup,
        exact_route_limit=exact_route_limit,
    )

    if len(route_eval_cache) < STAGE3_ROUTE_EVAL_CACHE_MAX_SIZE:
        route_eval_cache[cache_key] = (
            summary.copy(),
            stops.copy(deep=True),
        )

    return (
        summary,
        stops,
    )


def _insert_or_replace_column(
    dataframe: pd.DataFrame,
    position: int,
    column: str,
    value,
) -> pd.DataFrame:
    """
    Insert a metadata column if absent; otherwise replace its value.

    This prevents pandas 'cannot insert column, already exists' errors
    when route-stop tables are passed through multiple routing layers.
    """

    result = dataframe.copy()

    if column in result.columns:
        result[column] = value
    else:
        result.insert(
            position,
            column,
            value,
        )

    return result


def _route_assignments(
    routing_assignments: pd.DataFrame,
    drivers: pd.DataFrame,
    route_lookup: dict,
    model_scope: str,
    lambda_tag: str,
    lambda_equity: float,
    equity_measure_type: str = "maximum",
    exact_route_limit: int = STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
    route_eval_cache: dict[
        tuple,
        tuple[dict[str, Any], pd.DataFrame],
    ] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Route every non-empty driver-day group in an assignment dataframe.
    """

    if route_eval_cache is None:
        route_eval_cache = {}

    route_rows = []
    stop_tables = []

    for key, group in routing_assignments.groupby(
        [
            "day_index",
            "weekday",
            "driver_id",
            "driver_name",
        ],
        sort=True,
    ):

        day_index, weekday, driver_id, _driver_name = key

        route_key = _group_key(
            day_index=day_index,
            weekday=weekday,
            driver_id=driver_id,
            driver_name=_driver_name,
        )

        route_summary, route_stops = _evaluate_group_cached(
            key=route_key,
            group=group.copy(),
            route_lookup=route_lookup,
            route_eval_cache=route_eval_cache,
            exact_route_limit=exact_route_limit,
        )

        route_identifier = _route_id(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            day_index=day_index,
            driver_id=driver_id,
        )

        route_summary.update({
            "model_scope": model_scope,
            "lambda_tag": lambda_tag,
            "lambda_equity": lambda_equity,
            "equity_measure_type": equity_measure_type,
            "route_id": route_identifier,
        })

        route_stops = route_stops.copy()

        route_stops = _insert_or_replace_column(
            route_stops,
            0,
            "route_id",
            route_identifier,
        )

        route_stops = _insert_or_replace_column(
            route_stops,
            0,
            "lambda_tag",
            lambda_tag,
        )

        route_stops = _insert_or_replace_column(
            route_stops,
            2,
            "equity_measure_type",
            equity_measure_type,
        )

        route_stops = _insert_or_replace_column(
            route_stops,
            0,
            "model_scope",
            model_scope,
        )

        route_rows.append(route_summary)
        stop_tables.append(route_stops)

    route_summaries = pd.DataFrame(route_rows)

    route_stops = (
        pd.concat(
            stop_tables,
            ignore_index=True,
        )
        if stop_tables
        else pd.DataFrame()
    )

    driver_workloads = create_driver_route_workload_summary(
        route_summaries=route_summaries,
        drivers=drivers,
    )

    return (
        route_summaries,
        route_stops,
        driver_workloads,
    )


def _route_summary_by_key(
    route_summaries: pd.DataFrame,
) -> dict[tuple[int, str, int, str], dict]:
    """
    Create a lookup from a driver-day key to its route summary row.
    """

    lookup = {}

    for row in route_summaries.to_dict("records"):

        lookup[
            _group_key(
                row["day_index"],
                row["weekday"],
                row["driver_id"],
                row["driver_name"],
            )
        ] = row

    return lookup


def _groups_by_key(
    routing_assignments: pd.DataFrame,
) -> dict[tuple[int, str, int, str], pd.DataFrame]:
    """
    Split the current assignment into driver-day groups.
    """

    return {
        _group_key(*key): group.copy().reset_index(drop=True)
        for key, group
        in routing_assignments.groupby(
            [
                "day_index",
                "weekday",
                "driver_id",
                "driver_name",
            ],
            sort=True,
        )
    }


def _patch_assignment_row(
    row: pd.Series,
    target_day_index: int,
    target_weekday: str,
    target_driver_id: int,
    drivers_prepared: pd.DataFrame,
    assignment_costs: pd.DataFrame,
) -> pd.Series:
    """
    Return a copy of one enriched assignment row reassigned to a new
    driver-day pair.

    The order time-window fields are retained. The driver fields and
    base-order-base proxy assignment cost are updated for the new
    driver.
    """

    order_id = int(row["orderid"])
    target_driver_id = int(target_driver_id)

    driver_row = drivers_prepared.loc[
        drivers_prepared["driver_id"]
        .astype(int)
        .eq(target_driver_id)
    ].iloc[0]

    cost_row = assignment_costs.loc[
        assignment_costs["orderid"]
        .astype(int)
        .eq(order_id)
        & assignment_costs["driver_id"]
        .astype(int)
        .eq(target_driver_id)
    ].iloc[0]

    updated = row.copy()

    updated["day_index"] = int(target_day_index)
    updated["weekday"] = str(target_weekday)
    updated["driver_id"] = target_driver_id
    updated["driver_name"] = str(driver_row["driver.name"])
    updated["driver_locationid"] = int(
        cost_row["driver_locationid"]
    )
    updated["order_locationid"] = int(
        cost_row["locationid"]
    )
    updated["jobtime"] = float(
        cost_row["jobtime"]
    )
    updated["outbound_time"] = float(
        cost_row["outbound_time"]
    )
    updated["return_time"] = float(
        cost_row["return_time"]
    )
    updated["estimated_assignment_time"] = float(
        cost_row["estimated_assignment_time"]
    )
    updated["shiftstart"] = float(
        driver_row["shiftstart"]
    )
    updated["shiftend"] = float(
        driver_row["shiftend"]
    )
    updated["shift_length"] = float(
        driver_row["shift_length"]
    )

    return updated


def _affected_score(
    source_summary: dict | None,
    target_summary: dict | None,
) -> tuple[float, int, float, float]:
    """
    Lexicographic score for the affected source and target routes.

    Lower is better:
    1. total violation minutes;
    2. number of infeasible affected routes;
    3. total affected route duration;
    4. maximum affected route duration.
    """

    summaries = [
        summary
        for summary in [
            source_summary,
            target_summary,
        ]
        if summary is not None
    ]

    total_violation = sum(
        float(summary["total_violation_minutes"])
        for summary in summaries
    )

    infeasible_count = sum(
        0
        if bool(summary["route_feasible"])
        else 1
        for summary in summaries
    )

    total_duration = sum(
        float(summary["route_duration"])
        for summary in summaries
    )

    maximum_duration = max(
        [
            float(summary["route_duration"])
            for summary in summaries
        ]
        or [0.0]
    )

    return (
        total_violation,
        infeasible_count,
        total_duration,
        maximum_duration,
    )


def _empty_target_candidates(
    order_id: int,
    source_key: tuple[int, str, int, str],
    used_keys: set[tuple[int, str, int, str]],
    feasibility: pd.DataFrame,
    assignment_costs: pd.DataFrame,
    drivers_prepared: pd.DataFrame,
) -> pd.DataFrame:
    """
    Candidate target driver-days for one order.

    This minimal repair layer only considers currently unused
    driver-day pairs. This is conservative: it avoids making another
    existing route harder while still allowing overloaded routes to be
    relieved.
    """

    order_costs = assignment_costs.loc[
        assignment_costs["orderid"]
        .astype(int)
        .eq(int(order_id))
        & assignment_costs["estimated_assignment_time"].notna()
    ].copy()

    order_costs = (
        order_costs
        .sort_values("estimated_assignment_time")
        .head(STAGE3_REPAIR_CANDIDATE_DRIVERS)
    )


    feasible_days = feasibility.loc[
        feasibility["orderid"]
        .astype(int)
        .eq(int(order_id))
        & feasibility["is_feasible"].astype(bool),
        [
            "day_index",
            "weekday",
        ],
    ].drop_duplicates()

    rows = []

    for _, cost_row in order_costs.iterrows():

        driver_id = int(cost_row["driver_id"])

        driver_name = str(
            drivers_prepared.loc[
                drivers_prepared["driver_id"]
                .astype(int)
                .eq(driver_id),
                "driver.name",
            ].iloc[0]
        )

        for _, day_row in feasible_days.iterrows():

            target_key = _group_key(
                day_index=day_row["day_index"],
                weekday=day_row["weekday"],
                driver_id=driver_id,
                driver_name=driver_name,
            )

            if target_key == source_key:
                continue

            if target_key in used_keys:
                continue

            rows.append({
                "orderid": int(order_id),
                "target_day_index": int(day_row["day_index"]),
                "target_weekday": str(day_row["weekday"]),
                "target_driver_id": driver_id,
                "target_driver_name": driver_name,
                "estimated_assignment_time": float(
                    cost_row["estimated_assignment_time"]
                ),
                "target_key": target_key,
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("estimated_assignment_time")
        .head(STAGE3_REPAIR_MAX_TARGETS_PER_ORDER)
        .reset_index(drop=True)
    )


def _find_best_empty_day_repair_move(
    routing_assignments: pd.DataFrame,
    route_summaries: pd.DataFrame,
    feasibility: pd.DataFrame,
    assignment_costs: pd.DataFrame,
    drivers_prepared: pd.DataFrame,
    route_lookup: dict,
    route_eval_cache: dict[
        tuple,
        tuple[dict[str, Any], pd.DataFrame],
    ],
) -> dict | None:
    """
    Find the best single-order move from the worst infeasible route
    to an unused feasible driver-day pair.
    """

    infeasible = route_summaries.loc[
        ~route_summaries["route_feasible"].astype(bool)
    ].copy()

    if infeasible.empty:
        return None

    infeasible = infeasible.sort_values(
        [
            "total_violation_minutes",
            "route_duration",
        ],
        ascending=False,
    )

    source_route = infeasible.iloc[0]

    source_key = _group_key(
        source_route["day_index"],
        source_route["weekday"],
        source_route["driver_id"],
        source_route["driver_name"],
    )

    groups = _groups_by_key(
        routing_assignments
    )

    route_lookup_by_key = _route_summary_by_key(
        route_summaries
    )

    used_keys = set(groups.keys())

    source_group = groups[source_key]
    old_source_summary = route_lookup_by_key[source_key]
    old_score = _affected_score(
        old_source_summary,
        None,
    )

    best_move = None
    best_score = None

    for order_id in source_group["orderid"].astype(int).tolist():

        order_row = source_group.loc[
            source_group["orderid"]
            .astype(int)
            .eq(order_id)
        ].iloc[0]

        candidates = _empty_target_candidates(
            order_id=order_id,
            source_key=source_key,
            used_keys=used_keys,
            feasibility=feasibility,
            assignment_costs=assignment_costs,
            drivers_prepared=drivers_prepared,
        )

        new_source_group = source_group.loc[
            ~source_group["orderid"]
            .astype(int)
            .eq(order_id)
        ].copy()

        new_source_summary = None

        if not new_source_group.empty:
            new_source_summary, _ = _evaluate_group_cached(
                key=source_key,
                group=new_source_group,
                route_lookup=route_lookup,
                route_eval_cache=route_eval_cache,
                exact_route_limit=STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
            )


        for _, candidate in candidates.iterrows():

            moved_row = _patch_assignment_row(
                row=order_row,
                target_day_index=int(
                    candidate["target_day_index"]
                ),
                target_weekday=str(
                    candidate["target_weekday"]
                ),
                target_driver_id=int(
                    candidate["target_driver_id"]
                ),
                drivers_prepared=drivers_prepared,
                assignment_costs=assignment_costs,
            )

            target_group = pd.DataFrame(
                [moved_row]
            )

            new_target_summary, _ = _evaluate_group_cached(
                key=candidate["target_key"],
                group=target_group,
                route_lookup=route_lookup,
                route_eval_cache=route_eval_cache,
                exact_route_limit=STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
            )

            new_score = _affected_score(
                new_source_summary,
                new_target_summary,
            )

            if new_score >= old_score:
                continue

            if (
                best_score is None
                or new_score < best_score
            ):

                best_score = new_score

                best_move = {
                    "orderid": int(order_id),
                    "source_key": source_key,
                    "source_route_id": source_route["route_id"],
                    "target_key": candidate["target_key"],
                    "target_day_index": int(
                        candidate["target_day_index"]
                    ),
                    "target_weekday": str(
                        candidate["target_weekday"]
                    ),
                    "target_driver_id": int(
                        candidate["target_driver_id"]
                    ),
                    "target_driver_name": str(
                        candidate["target_driver_name"]
                    ),
                    "old_score": old_score,
                    "new_score": new_score,
                    "moved_row": moved_row,
                }

                if (
                    new_score[0] <= 1e-6
                    and new_score[1] == 0
                ):
                    return best_move

    return best_move


def _apply_move(
    routing_assignments: pd.DataFrame,
    move: dict,
) -> pd.DataFrame:
    """
    Apply a selected repair move.
    """

    repaired = routing_assignments.copy()

    order_id = int(move["orderid"])

    mask = (
        repaired["orderid"]
        .astype(int)
        .eq(order_id)
    )

    if mask.sum() != 1:
        raise RuntimeError(
            f"Expected one row for order {order_id}, "
            f"found {mask.sum()}."
        )

    for column, value in move["moved_row"].items():

        repaired.loc[
            mask,
            column,
        ] = value

    return repaired


def _update_changed_routes(
    routing_assignments: pd.DataFrame,
    route_summaries: pd.DataFrame,
    route_stops: pd.DataFrame,
    changed_keys: list[tuple[int, str, int, str]],
    route_lookup: dict,
    model_scope: str,
    lambda_tag: str,
    lambda_equity: float,
    equity_measure_type: str,
    route_eval_cache: dict[
        tuple,
        tuple[dict[str, Any], pd.DataFrame],
    ],
    exact_route_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-route only the source and target driver-day routes affected by
    one repair move, instead of rerouting the full solution.
    """

    changed_route_ids = []

    for (
        day_index,
        _weekday,
        driver_id,
        _driver_name,
    ) in changed_keys:
        changed_route_ids.append(
            _route_id(
                model_scope=model_scope,
                lambda_tag=lambda_tag,
                day_index=day_index,
                driver_id=driver_id,
            )
        )

    kept_summaries = route_summaries.loc[
        ~route_summaries["route_id"].isin(
            changed_route_ids
        )
    ].copy()

    kept_stops = route_stops.loc[
        ~route_stops["route_id"].isin(
            changed_route_ids
        )
    ].copy()

    groups = _groups_by_key(
        routing_assignments
    )

    new_summary_rows = []
    new_stop_tables = []

    for key in changed_keys:

        group = groups.get(
            key
        )

        if group is None or group.empty:
            continue

        day_index, weekday, driver_id, driver_name = key

        route_identifier = _route_id(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            day_index=day_index,
            driver_id=driver_id,
        )

        summary, stops = _evaluate_group_cached(
            key=key,
            group=group,
            route_lookup=route_lookup,
            route_eval_cache=route_eval_cache,
            exact_route_limit=exact_route_limit,
        )

        summary.update({
            "model_scope": model_scope,
            "lambda_tag": lambda_tag,
            "lambda_equity": lambda_equity,
            "equity_measure_type": equity_measure_type,
            "route_id": route_identifier,
        })

        stops = stops.copy()

        stops = _insert_or_replace_column(
            stops,
            0,
            "route_id",
            route_identifier,
        )

        stops = _insert_or_replace_column(
            stops,
            0,
            "lambda_tag",
            lambda_tag,
        )

        stops = _insert_or_replace_column(
            stops,
            2,
            "equity_measure_type",
            equity_measure_type,
        )

        stops = _insert_or_replace_column(
            stops,
            0,
            "model_scope",
            model_scope,
        )

        new_summary_rows.append(
            summary
        )

        new_stop_tables.append(
            stops
        )

    if new_summary_rows:

        updated_summaries = pd.concat(
            [
                kept_summaries,
                pd.DataFrame(new_summary_rows),
            ],
            ignore_index=True,
        )

    else:

        updated_summaries = kept_summaries.copy()

    if new_stop_tables:

        updated_stops = pd.concat(
            [
                kept_stops,
                *new_stop_tables,
            ],
            ignore_index=True,
        )

    else:

        updated_stops = kept_stops.copy()

    return (
        updated_summaries,
        updated_stops,
    )


def repair_stored_stage_3_solution(
    model_scope: str,
    lambda_tag: str,
    overwrite: bool = True,
    equity_measure_type: str = "maximum",
) -> pd.DataFrame:
    """
    Repair one Stage 3 routed solution.

    The original Stage 2 assignment files and unrepaired Stage 3 files
    are not overwritten. Repaired outputs are written to
    outputs/stage_3_routing_repaired/.
    """

    output_dir = stage_3_repaired_solution_dir(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    summary_path = (
        output_dir / "stage_3_solution_summary.csv"
    )

    if summary_path.exists() and not overwrite:
        return pd.read_csv(
            summary_path
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

    feasibility = pd.read_csv(
        DATA_PROCESSED_DIR / "feasibility_matrix.csv"
    )

    assignment_costs = pd.read_csv(
        DATA_PROCESSED_DIR / "assignment_cost_matrix.csv"
    )

    drivers_prepared = prepare_driver_table(
        drivers
    )

    route_lookup = build_route_lookup(
        routes
    )

    route_eval_cache: dict[
        tuple,
        tuple[dict[str, Any], pd.DataFrame],
    ] = {}

    changed_keys_for_final_polish: set[
        tuple[int, str, int, str]
    ] = set()

    metadata = load_solution_metadata(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    lambda_equity = float(
        metadata["lambda_equity"]
    )

    assignments = pd.read_csv(
        get_assignment_path(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            equity_measure_type=equity_measure_type,
        )
    )

    routing_assignments = prepare_assignments_for_routing(
        assignments=assignments,
        orders=orders,
        drivers=drivers,
    )

    existing_stage_3_routes = _load_existing_stage_3_routes(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    if existing_stage_3_routes is not None:

        print(
            "Using existing unrepaired Stage 3 routed outputs "
            "as repair starting point.",
            flush=True,
        )

        route_summaries, route_stops, driver_workloads = (
            existing_stage_3_routes
        )

    else:

        print(
            "Existing unrepaired Stage 3 routed outputs not found; "
            "routing full assignment before repair.",
            flush=True,
        )

        route_summaries, route_stops, driver_workloads = (
            _route_assignments(
                routing_assignments=routing_assignments,
                drivers=drivers,
                route_lookup=route_lookup,
                model_scope=model_scope,
                lambda_tag=lambda_tag,
                lambda_equity=lambda_equity,
                equity_measure_type=equity_measure_type,
                exact_route_limit=STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
                route_eval_cache=route_eval_cache,
            )
        )

    print(
        "Initial repaired-route search state: "
        f"{int((~route_summaries['route_feasible'].astype(bool)).sum())} "
        "infeasible routes, "
        f"{float(route_summaries['total_violation_minutes'].sum()):.4f} "
        "violation minutes.",
        flush=True,
    )

    move_rows = []

    for iteration in range(
        1,
        STAGE3_REPAIR_MAX_ITERATIONS + 1,
    ):

        infeasible_count = int(
            (
                ~route_summaries["route_feasible"]
                .astype(bool)
            ).sum()
        )

        total_violation = float(
            route_summaries[
                "total_violation_minutes"
            ].sum()
        )

        if (
            infeasible_count == 0
            or total_violation <= 1e-6
        ):
            break

        move = _find_best_empty_day_repair_move(
            routing_assignments=routing_assignments,
            route_summaries=route_summaries,
            feasibility=feasibility,
            assignment_costs=assignment_costs,
            drivers_prepared=drivers_prepared,
            route_lookup=route_lookup,
            route_eval_cache=route_eval_cache,
        )

        if move is None:

            print(
                "No improving repair move found for "
                f"{model_scope} {lambda_tag}."
            )

            break

        (
            source_day,
            source_weekday,
            source_driver_id,
            source_driver_name,
        ) = move["source_key"]

        (
            target_day,
            target_weekday,
            target_driver_id,
            target_driver_name,
        ) = move["target_key"]

        move_rows.append({
            "iteration": iteration,
            "orderid": move["orderid"],
            "source_route_id": move["source_route_id"],
            "source_day_index": source_day,
            "source_weekday": source_weekday,
            "source_driver_id": source_driver_id,
            "source_driver_name": source_driver_name,
            "target_day_index": target_day,
            "target_weekday": target_weekday,
            "target_driver_id": target_driver_id,
            "target_driver_name": target_driver_name,
            "old_affected_violation_minutes": move["old_score"][0],
            "new_affected_violation_minutes": move["new_score"][0],
            "old_affected_infeasible_routes": move["old_score"][1],
            "new_affected_infeasible_routes": move["new_score"][1],
            "old_affected_route_duration": move["old_score"][2],
            "new_affected_route_duration": move["new_score"][2],
        })

        routing_assignments = _apply_move(
            routing_assignments,
            move,
        )

        changed_keys = [
            move["source_key"],
            move["target_key"],
        ]

        changed_keys_for_final_polish.update(
            changed_keys
        )

        route_summaries, route_stops = _update_changed_routes(
            routing_assignments=routing_assignments,
            route_summaries=route_summaries,
            route_stops=route_stops,
            changed_keys=changed_keys,
            route_lookup=route_lookup,
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            lambda_equity=lambda_equity,
            equity_measure_type=equity_measure_type,
            route_eval_cache=route_eval_cache,
            exact_route_limit=STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
        )

        print(
            f"{model_scope} {lambda_tag}: "
            f"repair move {iteration} accepted; "
            "infeasible routes="
            f"{int((~route_summaries['route_feasible'].astype(bool)).sum())}, "
            "violation="
            f"{route_summaries['total_violation_minutes'].sum():.4f}"
        )

    move_log = pd.DataFrame(
        move_rows,
        columns=REPAIR_MOVE_LOG_COLUMNS,
    )

    if changed_keys_for_final_polish:

        print(
            "Final-polishing changed routes only: "
            f"{len(changed_keys_for_final_polish)} route keys.",
            flush=True,
        )

        route_summaries, route_stops = _update_changed_routes(
            routing_assignments=routing_assignments,
            route_summaries=route_summaries,
            route_stops=route_stops,
            changed_keys=sorted(changed_keys_for_final_polish),
            route_lookup=route_lookup,
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            lambda_equity=lambda_equity,
            equity_measure_type=equity_measure_type,
            route_eval_cache=route_eval_cache,
            exact_route_limit=STAGE3_REPAIR_FINAL_EXACT_ROUTE_LIMIT,
        )

    else:

        print(
            "No repair moves accepted; skipping final route polishing.",
            flush=True,
        )

    driver_workloads = create_driver_route_workload_summary(
        route_summaries=route_summaries,
        drivers=drivers,
    )

    solution_summary = create_stage_3_solution_summary(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        metadata=metadata,
        assignments=routing_assignments,
        route_summaries=route_summaries,
        driver_route_workloads=driver_workloads,
    )

    solution_summary["equity_measure_type"] = equity_measure_type

    solution_summary["repair_applied"] = True
    solution_summary["repair_strategy"] = (
        "single_order_to_empty_driver_day"
    )
    solution_summary["n_repair_moves"] = int(
        move_log.shape[0]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    routing_assignments[
        [
            column
            for column in ASSIGNMENT_OUTPUT_COLUMNS
            if column in routing_assignments.columns
        ]
    ].to_csv(
        output_dir / "repaired_assignments.csv",
        index=False,
    )

    route_summaries.to_csv(
        output_dir / "route_summaries.csv",
        index=False,
    )

    route_stops.to_csv(
        output_dir / "route_stops.csv",
        index=False,
    )

    driver_workloads.to_csv(
        output_dir / "driver_route_workloads.csv",
        index=False,
    )

    solution_summary.to_csv(
        summary_path,
        index=False,
    )

    move_log.to_csv(
        output_dir / "repair_move_log.csv",
        index=False,
    )

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_scope": model_scope,
        "lambda_tag": lambda_tag,
        "lambda_equity": lambda_equity,
        "repair_strategy": "single_order_to_empty_driver_day",
        "source_stage_3_directory": str(
            stage_3_solution_dir(
                model_scope=model_scope,
                lambda_tag=lambda_tag,
                equity_measure_type=equity_measure_type,
            ).relative_to(BASE_DIR)
        ),
        "n_repair_moves": int(
            move_log.shape[0]
        ),
        "n_infeasible_routes_after_repair": int(
            solution_summary[
                "n_infeasible_routes"
            ].iloc[0]
        ),
        "total_violation_minutes_after_repair": float(
            solution_summary[
                "total_violation_minutes"
            ].iloc[0]
        ),
        "candidate_drivers_per_order":
            STAGE3_REPAIR_CANDIDATE_DRIVERS,
        "max_targets_per_order":
            STAGE3_REPAIR_MAX_TARGETS_PER_ORDER,
        "search_exact_route_limit":
            STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT,
        "final_exact_route_limit":
            STAGE3_REPAIR_FINAL_EXACT_ROUTE_LIMIT,
        "route_eval_cache_max_size":
            STAGE3_ROUTE_EVAL_CACHE_MAX_SIZE,
        "max_iterations":
            STAGE3_REPAIR_MAX_ITERATIONS,
        "equity_measure_type": equity_measure_type,
    }

    with open(
        output_dir / "stage_3_repair_manifest.json",
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
                "n_infeasible_routes",
                "total_violation_minutes",
                "total_route_duration",
                "max_driver_route_workload",
                "cv_active_route_workload",
                "n_repair_moves",
            ]
        ].to_string(index=False)
    )

    return solution_summary


def run_stage_3_repairs(
    representative_solutions: list[tuple[str, str]] | None = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Repair all representative Stage 3 solutions and save one master
    summary.
    """

    if representative_solutions is None:
        representative_solutions = (
            STAGE3_REPRESENTATIVE_SOLUTIONS
        )

    STAGE3_REPAIRED_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for model_scope, lambda_tag in representative_solutions:

        summary = repair_stored_stage_3_solution(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            overwrite=overwrite,
        )

        summaries.append(
            summary
        )

    master = pd.concat(
        summaries,
        ignore_index=True,
    )

    master.to_csv(
        STAGE3_REPAIRED_RESULTS_DIR
        / "stage_3_repaired_master_summary.csv",
        index=False,
    )

    return master