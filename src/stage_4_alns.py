from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.assignment_costs import build_route_lookup

from src.config import (
    DATA_PROCESSED_DIR,
    STAGE3_REPAIRED_RESULTS_DIR,
    STAGE4_CANDIDATE_DRIVERS,
    STAGE4_COOLING_RATE,
    STAGE4_CV_WEIGHT,
    STAGE4_DESTROY_OPERATORS,
    STAGE4_FINAL_EXACT_ROUTE_LIMIT,
    STAGE4_FINAL_POLISH_ONLY_CHANGED_ROUTES,
    STAGE4_INFEASIBLE_ROUTE_PENALTY,
    STAGE4_INITIAL_TEMPERATURE,
    STAGE4_MAX_ITERATIONS,
    STAGE4_MAX_REMOVALS,
    STAGE4_MAX_TARGETS_PER_ORDER,
    STAGE4_MAX_WORKLOAD_DETERIORATION_PCT,
    STAGE4_MAX_WORKLOAD_DETERIORATION_PENALTY,
    STAGE4_MIN_REMOVALS,
    STAGE4_RANDOM_SEED,
    STAGE4_REMOVAL_FRACTION,
    STAGE4_REPAIR_OPERATORS,
    STAGE4_REPRESENTATIVE_SOLUTIONS,
    STAGE4_RESULTS_DIR,
    STAGE4_ROUTE_EVAL_CACHE_MAX_SIZE,
    STAGE4_SEARCH_EXACT_ROUTE_LIMIT,
    STAGE4_VIOLATION_PENALTY,
    EQUITY_LAMBDA_VALUES,
    EQUITY_MEASURE_LABELS,
    EQUITY_MEASURE_TYPES,
    BASE_DIR,
)

from src.equity_measures import (
    compute_workload_equity_metrics,
    equity_metric_value_for_measure,
    validate_equity_measure_type,
)

from src.routing_engine import (
    build_route_for_driver_day,
    prepare_assignments_for_routing,
    prepare_driver_table,
)
from src.stage_3_pipeline import (
    create_driver_route_workload_summary,
    create_stage_3_solution_summary,
    load_solution_metadata,
)


RouteKey = tuple[int, str, int, str]

def stage3_repaired_solution_dir(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
):
    """
    Locate the Stage 3 repaired output used as the Stage 4 starting point.

    New full-instance equity outputs live at:
        outputs/stage_3_routing_repaired/full_instance/<measure>/<lambda>/

    Legacy/reduced outputs live at:
        outputs/stage_3_routing_repaired/<scope>/<lambda>/
    """

    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
    )

    if model_scope == "full_instance":

        equity_directory = (
            STAGE3_REPAIRED_RESULTS_DIR
            / model_scope
            / equity_measure_type
            / lambda_tag
        )

        if equity_directory.exists():
            return equity_directory

        legacy_directory = (
            STAGE3_REPAIRED_RESULTS_DIR
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
        STAGE3_REPAIRED_RESULTS_DIR
        / model_scope
        / lambda_tag
    )


def stage4_solution_dir(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
):
    """
    Standard Stage 4 output folder.

    Full-instance equity outputs are separated by equity measure.
    """

    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
    )

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


def insert_or_replace_column(
    dataframe: pd.DataFrame,
    position: int,
    column: str,
    value,
) -> pd.DataFrame:
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


def route_workload_metrics(
    driver_workloads: pd.DataFrame,
) -> dict:
    workloads = (
        driver_workloads
        .set_index("driver_id")["total_route_duration"]
        .astype(float)
    )

    return compute_workload_equity_metrics(
        workloads
    )

def route_id(
    model_scope: str,
    lambda_tag: str,
    key: RouteKey,
) -> str:
    day_index, _weekday, driver_id, _driver_name = key

    return (
        f"{model_scope}_{lambda_tag}"
        f"_d{int(day_index):02d}"
        f"_k{int(driver_id):03d}"
    )


def key_from_row(
    row: pd.Series | dict[str, Any],
) -> RouteKey:
    return (
        int(row["day_index"]),
        str(row["weekday"]),
        int(row["driver_id"]),
        str(row["driver_name"]),
    )


def mask_for_key(
    assignments: pd.DataFrame,
    key: RouteKey,
) -> pd.Series:
    day_index, weekday, driver_id, driver_name = key

    return (
        assignments["day_index"].astype(int).eq(int(day_index))
        & assignments["weekday"].astype(str).eq(str(weekday))
        & assignments["driver_id"].astype(int).eq(int(driver_id))
        & assignments["driver_name"].astype(str).eq(str(driver_name))
    )


def parse_route_sequence(
    value: object,
) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    return [
        int(part)
        for part in str(value).split("-")
        if str(part).strip()
    ]

@dataclass
class Stage4Context:
    model_scope: str
    equity_measure_type: str
    lambda_tag: str
    lambda_equity: float
    metadata: dict[str, Any]

    orders: pd.DataFrame
    drivers: pd.DataFrame
    drivers_prepared: pd.DataFrame
    routes: pd.DataFrame
    feasibility: pd.DataFrame
    assignment_costs: pd.DataFrame
    route_lookup: dict

    baseline_summary: pd.Series
    baseline_route_duration: float
    baseline_max_workload: float
    baseline_driver_workloads: pd.DataFrame
    baseline_equity_metrics: dict[str, float]
    baseline_selected_equity_value: float

    order_coordinates: dict[int, tuple[float, float]]

    candidate_targets_by_order: dict[int, pd.DataFrame]
    route_eval_cache: dict[
        tuple,
        tuple[dict[str, Any], pd.DataFrame],
    ] = field(default_factory=dict)

def build_candidate_target_cache(
    feasibility: pd.DataFrame,
    assignment_costs: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """
    Precompute feasible driver-day insertion targets for each order.

    Candidate targets are cached because order-day feasibility and
    assignment costs remain fixed throughout the ALNS search.
    """

    cache: dict[int, pd.DataFrame] = {}

    feasible = feasibility.loc[
        feasibility["is_feasible"].astype(bool)
    ][
        [
            "orderid",
            "day_index",
            "weekday",
        ]
    ].copy()

    feasible["orderid"] = feasible["orderid"].astype(int)

    costs = assignment_costs.loc[
        assignment_costs["estimated_assignment_time"].notna()
    ][
        [
            "orderid",
            "driver_id",
            "driver_name",
            "estimated_assignment_time",
        ]
    ].copy()

    costs["orderid"] = costs["orderid"].astype(int)

    for order_id, feasible_days in feasible.groupby("orderid"):

        order_costs = (
            costs.loc[
                costs["orderid"]
                .astype(int)
                .eq(int(order_id))
            ]
            .sort_values("estimated_assignment_time")
            .head(STAGE4_CANDIDATE_DRIVERS)
            .copy()
        )

        if feasible_days.empty or order_costs.empty:
            cache[int(order_id)] = pd.DataFrame()
            continue

        candidates = (
            feasible_days[
                [
                    "day_index",
                    "weekday",
                ]
            ]
            .drop_duplicates()
            .merge(
                order_costs[
                    [
                        "driver_id",
                        "driver_name",
                        "estimated_assignment_time",
                    ]
                ],
                how="cross",
            )
        )

        candidates = (
            candidates
            .sort_values(
                [
                    "estimated_assignment_time",
                    "day_index",
                    "driver_id",
                ]
            )
            .head(STAGE4_MAX_TARGETS_PER_ORDER)
            .reset_index(drop=True)
        )

        cache[int(order_id)] = candidates

    return cache

def load_stage4_context(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
) -> Stage4Context:
    
    equity_measure_type = validate_equity_measure_type(
        equity_measure_type
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

    metadata = load_solution_metadata(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    metadata["equity_measure_type"] = equity_measure_type

    lambda_equity = float(
        metadata["lambda_equity"]
    )

    stage3_dir = stage3_repaired_solution_dir(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    if not stage3_dir.exists():
        raise FileNotFoundError(
            f"Stage 3 repaired source directory not found: {stage3_dir}"
        )

    baseline_summary = pd.read_csv(
        stage3_dir / "stage_3_solution_summary.csv"
    ).iloc[0]

    baseline_driver_workloads = pd.read_csv(
        stage3_dir / "driver_route_workloads.csv"
    )

    baseline_equity_metrics = route_workload_metrics(
        baseline_driver_workloads
    )

    baseline_selected_equity_value = equity_metric_value_for_measure(
        baseline_equity_metrics,
        equity_measure_type,
    )

    order_coordinates = {
        int(row.orderid): (
            float(row.x),
            float(row.y),
        )
        for row in orders.itertuples()
    }

    candidate_targets_by_order = build_candidate_target_cache(
        feasibility=feasibility,
        assignment_costs=assignment_costs,
    )

    return Stage4Context(
        model_scope=model_scope,
        equity_measure_type=equity_measure_type,
        lambda_tag=lambda_tag,
        lambda_equity=lambda_equity,
        metadata=metadata,
        orders=orders,
        drivers=drivers,
        drivers_prepared=drivers_prepared,
        routes=routes,
        feasibility=feasibility,
        assignment_costs=assignment_costs,
        route_lookup=route_lookup,
        baseline_summary=baseline_summary,
        baseline_route_duration=float(
            baseline_summary["total_route_duration"]
        ),
        baseline_max_workload=float(
            baseline_summary["max_driver_route_workload"]
        ),
        baseline_driver_workloads=baseline_driver_workloads,
        baseline_equity_metrics=baseline_equity_metrics,
        baseline_selected_equity_value=float(
            baseline_selected_equity_value
        ),
        order_coordinates=order_coordinates,
        candidate_targets_by_order=candidate_targets_by_order,
    )

@dataclass
class Stage4Score:
    scalar_score: float
    total_violation_minutes: float
    n_infeasible_routes: int
    total_route_duration: float
    selected_equity_metric_value: float
    max_driver_route_workload: float
    workload_range: float
    workload_mad_mean: float
    workload_minmax_deviation: float
    workload_gini_coefficient: float
    cv_active_route_workload: float
    n_driver_day_routes: int
    n_active_drivers: int

    def as_tuple(self) -> tuple:
        return (
            round(self.total_violation_minutes, 8),
            int(self.n_infeasible_routes),
            round(self.scalar_score, 10),
            round(self.total_route_duration, 8),
            round(self.selected_equity_metric_value, 8),
            round(self.max_driver_route_workload, 8),
            round(self.cv_active_route_workload, 10),
        )

    def better_than(
        self,
        other: "Stage4Score",
    ) -> bool:
        return self.as_tuple() < other.as_tuple()

def route_eval_cache_key(
    key: RouteKey,
    group: pd.DataFrame,
) -> tuple:
    """
    Route evaluation depends on the driver-day key, the set of orders
    assigned to that route and the exact-route limit used.
    """

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
    )


def evaluate_driver_day_group_cached(
    key: RouteKey,
    group: pd.DataFrame,
    ctx: Stage4Context,
    exact_route_limit: int = STAGE4_SEARCH_EXACT_ROUTE_LIMIT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Evaluate a driver-day route using cached results where available.

    Caching avoids repeated route evaluations for identical candidate
    assignments during the ALNS search.
    """

    cache_key = route_eval_cache_key(
        key=key,
        group=group,
    ) + (
        int(exact_route_limit),
    )

    cached = ctx.route_eval_cache.get(
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
        route_lookup=ctx.route_lookup,
        exact_route_limit=exact_route_limit,
    )

    if len(ctx.route_eval_cache) < STAGE4_ROUTE_EVAL_CACHE_MAX_SIZE:
        ctx.route_eval_cache[cache_key] = (
            summary.copy(),
            stops.copy(deep=True),
        )

    return (
        summary,
        stops,
    )



@dataclass
class Stage4Solution:
    assignments: pd.DataFrame
    routes_by_key: dict[RouteKey, list[int]]
    summaries_by_key: dict[RouteKey, dict[str, Any]]
    stops_by_key: dict[RouteKey, pd.DataFrame]
    score: Stage4Score | None = None
    changed_route_keys: set[RouteKey] = field(default_factory=set)

    def clone(self) -> "Stage4Solution":
        return Stage4Solution(
            assignments=(
                self.assignments
                .copy(deep=True)
                .reset_index(drop=True)
            ),
            routes_by_key={
                key: sequence.copy()
                for key, sequence
                in self.routes_by_key.items()
            },
            summaries_by_key={
                key: value.copy()
                for key, value
                in self.summaries_by_key.items()
            },
            stops_by_key={
                key: value.copy(deep=True)
                for key, value
                in self.stops_by_key.items()
            },
            score=self.score,
            changed_route_keys=set(self.changed_route_keys),
        )

    def route_summaries_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(self.summaries_by_key.values())
        )

    def route_stops_df(self) -> pd.DataFrame:
        if not self.stops_by_key:
            return pd.DataFrame()

        return pd.concat(
            list(self.stops_by_key.values()),
            ignore_index=True,
        )

    def reroute_key(
        self,
        key: RouteKey,
        ctx: Stage4Context,
        exact_route_limit: int = STAGE4_SEARCH_EXACT_ROUTE_LIMIT,
        mark_changed: bool = True,
    ) -> None:
        group = self.assignments.loc[
            mask_for_key(
                self.assignments,
                key,
            )
        ].copy().reset_index(drop=True)

        if group.empty:
            self.routes_by_key.pop(key, None)
            self.summaries_by_key.pop(key, None)
            self.stops_by_key.pop(key, None)

            if mark_changed:
                self.changed_route_keys.add(key)

            self.score = None
            return

        route_summary, route_stops = evaluate_driver_day_group_cached(
            key=key,
            group=group,
            ctx=ctx,
            exact_route_limit=exact_route_limit,
        )

        route_identifier = route_id(
            model_scope=ctx.model_scope,
            lambda_tag=ctx.lambda_tag,
            key=key,
        )

        route_summary.update({
            "model_scope": ctx.model_scope,
            "equity_measure_type": ctx.equity_measure_type,
            "lambda_tag": ctx.lambda_tag,
            "lambda_equity": ctx.lambda_equity,
            "route_id": route_identifier,
        })

        route_stops = route_stops.copy()

        route_stops = insert_or_replace_column(
            route_stops,
            0,
            "route_id",
            route_identifier,
        )

        route_stops = insert_or_replace_column(
            route_stops,
            0,
            "lambda_tag",
            ctx.lambda_tag,
        )

        route_stops = insert_or_replace_column(
            route_stops,
            2,
            "equity_measure_type",
            ctx.equity_measure_type,
        )

        route_stops = insert_or_replace_column(
            route_stops,
            0,
            "model_scope",
            ctx.model_scope,
        )

        self.routes_by_key[key] = parse_route_sequence(
            route_summary["route_sequence"]
        )

        self.summaries_by_key[key] = route_summary
        self.stops_by_key[key] = route_stops

        if mark_changed:
            self.changed_route_keys.add(key)

        self.score = None

    def remove_orders(
        self,
        order_ids: list[int],
        ctx: Stage4Context,
    ) -> pd.DataFrame:
        order_ids = [
            int(order_id)
            for order_id in order_ids
        ]

        remove_mask = (
            self.assignments["orderid"]
            .astype(int)
            .isin(order_ids)
        )

        removed = (
            self.assignments
            .loc[remove_mask]
            .copy()
            .reset_index(drop=True)
        )

        affected_keys = {
            key_from_row(row)
            for row
            in removed.to_dict("records")
        }

        self.assignments = (
            self.assignments
            .loc[~remove_mask]
            .copy()
            .reset_index(drop=True)
        )

        for key in affected_keys:
            self.reroute_key(
                key=key,
                ctx=ctx,
            )

        self.score = None

        return removed


    def insert_assignment_row(
        self,
        row: pd.Series,
        ctx: Stage4Context,
    ) -> None:
        key = key_from_row(row)

        self.assignments = pd.concat(
            [
                self.assignments,
                pd.DataFrame([row.to_dict()]),
            ],
            ignore_index=True,
        )

        self.reroute_key(
            key=key,
            ctx=ctx,
        )

        self.score = None

    def driver_workloads_df(
        self,
        ctx: Stage4Context,
    ) -> pd.DataFrame:
        return create_driver_route_workload_summary(
            route_summaries=self.route_summaries_df(),
            drivers=ctx.drivers,
        )


    def evaluate_score(
        self,
        ctx: Stage4Context,
    ) -> Stage4Score:
        route_summaries = self.route_summaries_df()

        driver_workloads = self.driver_workloads_df(
            ctx
        )

        all_equity_metrics = route_workload_metrics(
            driver_workloads
        )

        selected_equity_value = equity_metric_value_for_measure(
            all_equity_metrics,
            ctx.equity_measure_type,
        )

        active = driver_workloads.loc[
            driver_workloads["is_active"].astype(bool)
        ].copy()

        total_violation = float(
            route_summaries[
                "total_violation_minutes"
            ].sum()
        )

        n_infeasible = int(
            (
                ~route_summaries[
                    "route_feasible"
                ].astype(bool)
            ).sum()
        )

        total_duration = float(
            route_summaries[
                "route_duration"
            ].sum()
        )

        if active.empty:
            max_workload = 0.0
            cv = 0.0
            n_active = 0

        else:
            workloads = active[
                "total_route_duration"
            ].astype(float)

            max_workload = float(
                workloads.max()
            )

            mean_workload = float(
                workloads.mean()
            )

            std_workload = float(
                workloads.std(ddof=1)
            )

            if pd.isna(std_workload):
                std_workload = 0.0

            cv = (
                std_workload / mean_workload
                if mean_workload > 0
                else 0.0
            )

            n_active = int(
                active.shape[0]
            )

        duration_ref = max(
            float(ctx.baseline_route_duration),
            1.0,
        )

        workload_ref = max(
            float(ctx.baseline_max_workload),
            1.0,
        )

        selected_equity_ref = max(
            float(ctx.baseline_selected_equity_value),
            1.0,
        )

        lam = float(
            ctx.lambda_equity
        )

        allowed_max_workload = (
            float(ctx.baseline_max_workload)
            * (
                1.0
                + STAGE4_MAX_WORKLOAD_DETERIORATION_PCT
            )
        )

        max_workload_excess_ratio = max(
            0.0,
            (
                max_workload
                - allowed_max_workload
            )
            / workload_ref,
        )

        scalar = (
            STAGE4_VIOLATION_PENALTY
            * total_violation
            + STAGE4_INFEASIBLE_ROUTE_PENALTY
            * n_infeasible
            + (1.0 - lam)
            * (total_duration / duration_ref)
            + lam
            * (selected_equity_value / selected_equity_ref)
            + STAGE4_CV_WEIGHT
            * cv
            + STAGE4_MAX_WORKLOAD_DETERIORATION_PENALTY
            * max_workload_excess_ratio
        )

        self.score = Stage4Score(
            scalar_score=float(scalar),
            total_violation_minutes=total_violation,
            n_infeasible_routes=n_infeasible,
            total_route_duration=total_duration,
            selected_equity_metric_value=float(
                selected_equity_value
            ),
            max_driver_route_workload=max_workload,
            workload_range=float(
                all_equity_metrics["workload_range"]
            ),
            workload_mad_mean=float(
                all_equity_metrics["workload_mad_mean"]
            ),
            workload_minmax_deviation=float(
                all_equity_metrics[
                    "workload_minmax_deviation"
                ]
            ),
            workload_gini_coefficient=float(
                all_equity_metrics[
                    "workload_gini_coefficient"
                ]
            ),
            cv_active_route_workload=float(cv),
            n_driver_day_routes=int(
                route_summaries.shape[0]
            ),
            n_active_drivers=n_active,
        )

        return self.score
    
def build_initial_solution(
    ctx: Stage4Context,
) -> Stage4Solution:
    
    stage3_dir = stage3_repaired_solution_dir(
        model_scope=ctx.model_scope,
        lambda_tag=ctx.lambda_tag,
        equity_measure_type=ctx.equity_measure_type,
    )

    assignments = pd.read_csv(
        stage3_dir / "repaired_assignments.csv"
    )

    route_summaries = pd.read_csv(
        stage3_dir / "route_summaries.csv"
    )

    route_stops = pd.read_csv(
        stage3_dir / "route_stops.csv"
    )

    assignments["equity_measure_type"] = ctx.equity_measure_type

    if "equity_measure_type" not in route_summaries.columns:
        route_summaries["equity_measure_type"] = (
            ctx.equity_measure_type
        )

    if "equity_measure_type" not in route_stops.columns:
        route_stops["equity_measure_type"] = (
            ctx.equity_measure_type
        )

    routing_assignments = prepare_assignments_for_routing(
        assignments=assignments,
        orders=ctx.orders,
        drivers=ctx.drivers,
    )

    routes_by_key = {}
    summaries_by_key = {}
    stops_by_key = {}

    for row in route_summaries.to_dict("records"):
        key = key_from_row(row)

        routes_by_key[key] = parse_route_sequence(
            row["route_sequence"]
        )

        summaries_by_key[key] = row.copy()

        stops_by_key[key] = (
            route_stops.loc[
                route_stops["route_id"]
                .astype(str)
                .eq(str(row["route_id"]))
            ]
            .copy()
            .reset_index(drop=True)
        )

    solution = Stage4Solution(
        assignments=routing_assignments,
        routes_by_key=routes_by_key,
        summaries_by_key=summaries_by_key,
        stops_by_key=stops_by_key,
        changed_route_keys=set(),
    )

    solution.evaluate_score(
        ctx
    )

    return solution

def patch_assignment_row(
    row: pd.Series,
    target_day_index: int,
    target_weekday: str,
    target_driver_id: int,
    ctx: Stage4Context,
) -> pd.Series:
    order_id = int(
        row["orderid"]
    )

    target_driver_id = int(
        target_driver_id
    )

    driver_row = ctx.drivers_prepared.loc[
        ctx.drivers_prepared["driver_id"]
        .astype(int)
        .eq(target_driver_id)
    ].iloc[0]

    cost_row = ctx.assignment_costs.loc[
        ctx.assignment_costs["orderid"]
        .astype(int)
        .eq(order_id)
        & ctx.assignment_costs["driver_id"]
        .astype(int)
        .eq(target_driver_id)
    ].iloc[0]

    updated = row.copy()

    updated["day_index"] = int(
        target_day_index
    )

    updated["weekday"] = str(
        target_weekday
    )

    updated["driver_id"] = target_driver_id

    updated["driver_name"] = str(
        driver_row["driver.name"]
    )

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

def candidate_targets_for_order(
    row: pd.Series,
    ctx: Stage4Context,
) -> pd.DataFrame:
    order_id = int(
        row["orderid"]
    )

    targets = ctx.candidate_targets_by_order.get(
        order_id,
        pd.DataFrame(),
    )

    return targets.copy()

def removal_count(
    solution: Stage4Solution,
) -> int:
    n_orders = int(
        solution.assignments[
            "orderid"
        ].nunique()
    )

    return max(
        STAGE4_MIN_REMOVALS,
        min(
            STAGE4_MAX_REMOVALS,
            int(
                math.ceil(
                    n_orders
                    * STAGE4_REMOVAL_FRACTION
                )
            ),
        ),
    )

def select_orders_to_remove(
    solution: Stage4Solution,
    ctx: Stage4Context,
    operator_name: str,
    rng: random.Random,
) -> list[int]:
    k = removal_count(
        solution
    )

    all_orders = (
        solution.assignments["orderid"]
        .astype(int)
        .tolist()
    )

    if operator_name == "random_removal":
        return rng.sample(
            all_orders,
            k=min(k, len(all_orders)),
        )

    route_summaries = solution.route_summaries_df()

    if operator_name == "worst_route_removal":
        ranked = route_summaries.copy()

        ranked["route_badness"] = (
            1000.0
            * ranked[
                "total_violation_minutes"
            ].astype(float)
            + ranked[
                "route_duration"
            ].astype(float)
        )

        selected = []

        for row in (
            ranked
            .sort_values(
                "route_badness",
                ascending=False,
            )
            .itertuples()
        ):
            selected.extend(
                parse_route_sequence(
                    row.route_sequence
                )
            )

            if len(selected) >= k:
                break

        rng.shuffle(
            selected
        )

        return selected[:k]

    if operator_name == "high_workload_driver_removal":
        workloads = solution.driver_workloads_df(
            ctx
        )

        active = workloads.loc[
            workloads["is_active"]
            .astype(bool)
        ].copy()

        if active.empty:
            return rng.sample(
                all_orders,
                k=min(k, len(all_orders)),
            )

        driver_id = int(
            active
            .sort_values(
                "total_route_duration",
                ascending=False,
            )
            .iloc[0]["driver_id"]
        )

        driver_orders = (
            solution.assignments
            .loc[
                solution.assignments[
                    "driver_id"
                ].astype(int).eq(driver_id),
                "orderid",
            ]
            .astype(int)
            .tolist()
        )

        if len(driver_orders) >= k:
            return rng.sample(
                driver_orders,
                k=k,
            )

        remaining = [
            order
            for order in all_orders
            if order not in set(driver_orders)
        ]

        return (
            driver_orders
            + rng.sample(
                remaining,
                k=min(
                    k - len(driver_orders),
                    len(remaining),
                ),
            )
        )

    if operator_name == "related_order_removal":
        seed_order = int(
            rng.choice(
                all_orders
            )
        )

        seed_xy = ctx.order_coordinates.get(
            seed_order
        )

        if seed_xy is None:
            return rng.sample(
                all_orders,
                k=min(k, len(all_orders)),
            )

        sx, sy = seed_xy

        distances = []

        for order_id in all_orders:
            xy = ctx.order_coordinates.get(
                int(order_id)
            )

            if xy is None:
                continue

            x, y = xy

            distances.append(
                (
                    int(order_id),
                    (x - sx) ** 2
                    + (y - sy) ** 2,
                )
            )

        distances.sort(
            key=lambda item: item[1]
        )

        return [
            order_id
            for order_id, _distance
            in distances[:k]
        ]

    raise ValueError(
        f"Unknown destroy operator: {operator_name}"
    )

def local_insertion_score(
    solution: Stage4Solution,
    row: pd.Series,
    target: pd.Series,
    ctx: Stage4Context,
    repair_operator: str,
    driver_workload_lookup: dict[int, float] | None = None,
) -> tuple[tuple[float, int, float, float], pd.Series]:
    updated = patch_assignment_row(
        row=row,
        target_day_index=int(target["day_index"]),
        target_weekday=str(target["weekday"]),
        target_driver_id=int(target["driver_id"]),
        ctx=ctx,
    )

    target_key = key_from_row(
        updated
    )

    target_mask = mask_for_key(
        solution.assignments,
        target_key,
    )

    candidate_group = pd.concat(
        [
            solution.assignments
            .loc[target_mask]
            .copy(),
            pd.DataFrame(
                [updated.to_dict()]
            ),
        ],
        ignore_index=True,
    )

    new_summary, _new_stops = evaluate_driver_day_group_cached(
        key=target_key,
        group=candidate_group,
        ctx=ctx,
        exact_route_limit=STAGE4_SEARCH_EXACT_ROUTE_LIMIT,
    )

    old_summary = solution.summaries_by_key.get(
        target_key
    )

    old_violation = (
        0.0
        if old_summary is None
        else float(
            old_summary[
                "total_violation_minutes"
            ]
        )
    )

    old_duration = (
        0.0
        if old_summary is None
        else float(
            old_summary[
                "route_duration"
            ]
        )
    )

    delta_violation = (
        float(
            new_summary[
                "total_violation_minutes"
            ]
        )
        - old_violation
    )

    delta_duration = (
        float(
            new_summary[
                "route_duration"
            ]
        )
        - old_duration
    )

    new_infeasible = (
        0
        if bool(
            new_summary["route_feasible"]
        )
        else 1
    )

    if repair_operator == "workload_balancing_insertion":

        if driver_workload_lookup is None:
            target_driver_load = 0.0

        else:
            target_driver_load = float(
                driver_workload_lookup.get(
                    int(target["driver_id"]),
                    0.0,
                )
            )

        workload_tie_break = (
            0.00001
            * target_driver_load
        )

    else:
        workload_tie_break = 0.0

    score = (
        delta_violation,
        new_infeasible,
        delta_duration,
        workload_tie_break,
    )

    return score, updated

def best_insertion_for_order(
    solution: Stage4Solution,
    row: pd.Series,
    ctx: Stage4Context,
    repair_operator: str,
) -> tuple[tuple[float, int, float, float], pd.Series] | None:
    targets = candidate_targets_for_order(
        row=row,
        ctx=ctx,
    )

    scored = []

    workload_lookup = driver_workload_lookup(
        solution,
        ctx,
    )

    for _, target in targets.iterrows():
        try:
            score, updated = local_insertion_score(
                solution=solution,
                row=row,
                target=target,
                ctx=ctx,
                repair_operator=repair_operator,
                driver_workload_lookup=workload_lookup,
            )

            scored.append(
                (
                    score,
                    updated,
                )
            )

        except Exception:
            continue

    if not scored:
        return None

    scored.sort(
        key=lambda item: item[0]
    )

    return scored[0]

def driver_workload_lookup(
    solution: Stage4Solution,
    ctx: Stage4Context,
) -> dict[int, float]:
    workloads = solution.driver_workloads_df(
        ctx
    )

    return {
        int(row.driver_id): float(row.total_route_duration)
        for row in workloads.itertuples()
    }

def repair_removed_orders(
    partial_solution: Stage4Solution,
    removed_rows: pd.DataFrame,
    ctx: Stage4Context,
    repair_operator: str,
    rng: random.Random,
) -> Stage4Solution | None:
    working = partial_solution.clone()

    remaining = (
        removed_rows
        .copy()
        .reset_index(drop=True)
    )

    while not remaining.empty:
        candidates = []

        workload_lookup = driver_workload_lookup(
            working,
            ctx,
        )

        for row_index, row in remaining.iterrows():
            targets = candidate_targets_for_order(
                row=row,
                ctx=ctx,
            )

            scored = []

            for _, target in targets.iterrows():
                try:
                    score, updated = local_insertion_score(
                        solution=working,
                        row=row,
                        target=target,
                        ctx=ctx,
                        repair_operator=repair_operator,
                        driver_workload_lookup=workload_lookup,
                    )

                    scored.append(
                        (
                            score,
                            updated,
                        )
                    )

                except Exception:
                    continue

            if not scored:
                continue

            scored.sort(
                key=lambda item: item[0]
            )

            best_score, best_updated = scored[0]

            if len(scored) > 1:
                second_score = scored[1][0]
            else:
                second_score = best_score

            regret = (
                second_score[2]
                - best_score[2]
            )

            candidates.append({
                "row_index": row_index,
                "best_score": best_score,
                "regret": regret,
                "updated": best_updated,
            })

        if not candidates:
            return None

        if repair_operator == "regret_2_insertion":
            chosen = max(
                candidates,
                key=lambda record: (
                    record["regret"],
                    -record["best_score"][2],
                ),
            )

        else:
            chosen = min(
                candidates,
                key=lambda record: record["best_score"],
            )

        working.insert_assignment_row(
            row=chosen["updated"],
            ctx=ctx,
        )

        remaining = (
            remaining
            .drop(
                index=int(chosen["row_index"])
            )
            .reset_index(drop=True)
        )

    working.evaluate_score(
        ctx
    )

    return working

def accept_candidate(
    current: Stage4Score,
    candidate: Stage4Score,
    temperature: float,
    rng: random.Random,
) -> bool:
    if candidate.better_than(
        current
    ):
        return True

    delta = (
        candidate.scalar_score
        - current.scalar_score
    )

    if temperature <= 1e-12:
        return False

    probability = (
        math.exp(
            -delta / temperature
        )
        if delta > 0
        else 1.0
    )

    return rng.random() < probability

def choose_weighted(
    rng: random.Random,
    weights: dict[str, float],
) -> str:
    names = list(
        weights.keys()
    )

    values = [
        max(
            0.001,
            float(weights[name]),
        )
        for name in names
    ]

    return rng.choices(
        names,
        weights=values,
        k=1,
    )[0]

def update_operator_weight(
    weights: dict[str, float],
    operator_name: str,
    reward: float,
) -> None:
    weights[operator_name] = (
        0.90
        * weights[operator_name]
        + 0.10
        * max(
            0.1,
            reward,
        )
    )

def run_alns_search(
    ctx: Stage4Context,
    max_iterations: int = STAGE4_MAX_ITERATIONS,
    random_seed: int = STAGE4_RANDOM_SEED,
) -> tuple[
    Stage4Solution,
    pd.DataFrame,
    pd.DataFrame,
    Stage4Solution,
]:
    rng = random.Random(
        random_seed
    )

    initial = build_initial_solution(
        ctx
    )

    current = initial.clone()
    best = initial.clone()

    destroy_weights = {
        name: 1.0
        for name in STAGE4_DESTROY_OPERATORS
    }

    repair_weights = {
        name: 1.0
        for name in STAGE4_REPAIR_OPERATORS
    }

    temperature = STAGE4_INITIAL_TEMPERATURE

    iteration_rows = []

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        destroy_operator = choose_weighted(
            rng,
            destroy_weights,
        )

        repair_operator = choose_weighted(
            rng,
            repair_weights,
        )

        candidate = current.clone()

        removed_order_ids = select_orders_to_remove(
            solution=candidate,
            ctx=ctx,
            operator_name=destroy_operator,
            rng=rng,
        )

        removed_rows = candidate.remove_orders(
            order_ids=removed_order_ids,
            ctx=ctx,
        )

        repaired_candidate = repair_removed_orders(
            partial_solution=candidate,
            removed_rows=removed_rows,
            ctx=ctx,
            repair_operator=repair_operator,
            rng=rng,
        )

        if repaired_candidate is None:
            accepted = False
            new_best = False
            improved_current = False
            reward = 0.0
            candidate_score = None

        else:
            candidate_score = repaired_candidate.evaluate_score(
                ctx
            )

            current_score = (
                current.score
                or current.evaluate_score(ctx)
            )

            accepted = accept_candidate(
                current=current_score,
                candidate=candidate_score,
                temperature=temperature,
                rng=rng,
            )

            improved_current = candidate_score.better_than(
                current_score
            )

            best_score = (
                best.score
                or best.evaluate_score(ctx)
            )

            new_best = candidate_score.better_than(
                best_score
            )

            if accepted:
                current = repaired_candidate

            if new_best:
                best = repaired_candidate.clone()

            if new_best:
                reward = 10.0
            elif improved_current:
                reward = 5.0
            elif accepted:
                reward = 1.0
            else:
                reward = 0.0

        update_operator_weight(
            destroy_weights,
            destroy_operator,
            reward,
        )

        update_operator_weight(
            repair_weights,
            repair_operator,
            reward,
        )

        current_score = (
            current.score
            or current.evaluate_score(ctx)
        )

        best_score = (
            best.score
            or best.evaluate_score(ctx)
        )

        iteration_rows.append({
            "iteration": iteration,
            "destroy_operator": destroy_operator,
            "repair_operator": repair_operator,
            "n_removed_orders": int(
                len(removed_order_ids)
            ),
            "accepted": bool(accepted),
            "improved_current": bool(
                improved_current
            ),
            "new_best": bool(new_best),
            "temperature": float(temperature),
            "candidate_scalar_score": (
                None
                if candidate_score is None
                else float(
                    candidate_score.scalar_score
                )
            ),
            "current_scalar_score": float(
                current_score.scalar_score
            ),
            "best_scalar_score": float(
                best_score.scalar_score
            ),
            "best_total_route_duration": float(
                best_score.total_route_duration
            ),
            "best_selected_equity_metric_value": float(
                best_score.selected_equity_metric_value
            ),
            "best_max_driver_route_workload": float(
                best_score.max_driver_route_workload
            ),
            "best_cv_active_route_workload": float(
                best_score.cv_active_route_workload
            ),
            "best_total_violation_minutes": float(
                best_score.total_violation_minutes
            ),
        })

        temperature *= STAGE4_COOLING_RATE

        if (
            iteration == 1
            or iteration == max_iterations
            or iteration % 10 == 0
            or new_best
        ):
            print(
                f"{ctx.model_scope} {ctx.equity_measure_type} "
                f"{ctx.lambda_tag} | "
                f"iter {iteration:04d} | "
                f"best score={best_score.scalar_score:.6f} | "
                f"duration={best_score.total_route_duration:,.1f} | "
                f"equity={best_score.selected_equity_metric_value:,.4f} | "
                f"max={best_score.max_driver_route_workload:,.1f} | "
                f"cv={best_score.cv_active_route_workload:.4f} | "
                f"violation={best_score.total_violation_minutes:.4f}"
            )

    operator_rows = []

    for name, weight in destroy_weights.items():
        operator_rows.append({
            "operator_type": "destroy",
            "operator_name": name,
            "final_weight": weight,
        })

    for name, weight in repair_weights.items():
        operator_rows.append({
            "operator_type": "repair",
            "operator_name": name,
            "final_weight": weight,
        })

    return (
        best,
        pd.DataFrame(iteration_rows),
        pd.DataFrame(operator_rows),
        initial,
    )

def count_changed_assignments(
    initial: Stage4Solution,
    best: Stage4Solution,
) -> int:
    initial_map = {
        int(row.orderid): (
            int(row.day_index),
            int(row.driver_id),
        )
        for row in initial.assignments.itertuples()
    }

    best_map = {
        int(row.orderid): (
            int(row.day_index),
            int(row.driver_id),
        )
        for row in best.assignments.itertuples()
    }

    return sum(
        1
        for order_id, value
        in initial_map.items()
        if best_map.get(order_id) != value
    )

def create_stage4_solution_summary(
    ctx: Stage4Context,
    initial: Stage4Solution,
    best: Stage4Solution,
    iteration_log: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    route_summaries = best.route_summaries_df()

    driver_workloads = best.driver_workloads_df(
        ctx
    )

    summary = create_stage_3_solution_summary(
        model_scope=ctx.model_scope,
        lambda_tag=ctx.lambda_tag,
        metadata=ctx.metadata,
        assignments=best.assignments,
        route_summaries=route_summaries,
        driver_route_workloads=driver_workloads,
    )

    baseline = ctx.baseline_summary

    best_score = (
        best.score
        or best.evaluate_score(ctx)
    )

    summary["stage"] = "stage_4_alns"
    summary["source_stage"] = "stage_3_repaired"
    summary["equity_measure_type"] = ctx.equity_measure_type

    summary["n_changed_route_keys"] = int(
        len(best.changed_route_keys)
    )

    summary["final_polish_only_changed_routes"] = bool(
        STAGE4_FINAL_POLISH_ONLY_CHANGED_ROUTES
    )

    summary["final_exact_route_limit"] = int(
        STAGE4_FINAL_EXACT_ROUTE_LIMIT
    )

    summary["stage4_selected_equity_metric_value"] = float(
        best_score.selected_equity_metric_value
    )

    summary["baseline_selected_equity_metric_value"] = float(
        ctx.baseline_selected_equity_value
    )

    summary["delta_selected_equity_metric_value"] = (
        summary["stage4_selected_equity_metric_value"]
        - summary["baseline_selected_equity_metric_value"]
    )

    summary["stage4_workload_range"] = float(
        best_score.workload_range
    )

    summary["stage4_workload_mad_mean"] = float(
        best_score.workload_mad_mean
    )

    summary["stage4_workload_minmax_deviation"] = float(
        best_score.workload_minmax_deviation
    )

    summary["stage4_workload_gini_coefficient"] = float(
        best_score.workload_gini_coefficient
    )
    summary["alns_iterations"] = int(
        iteration_log.shape[0]
    )
    summary["random_seed"] = int(
        random_seed
    )
    summary["alns_scalar_score"] = float(
        best_score.scalar_score
    )

    summary["n_changed_assignments"] = int(
        count_changed_assignments(
            initial,
            best,
        )
    )

    summary["baseline_total_route_duration"] = float(
        baseline["total_route_duration"]
    )

    summary["baseline_max_driver_route_workload"] = float(
        baseline["max_driver_route_workload"]
    )

    summary["baseline_cv_active_route_workload"] = float(
        baseline["cv_active_route_workload"]
    )

    summary["baseline_total_violation_minutes"] = float(
        baseline["total_violation_minutes"]
    )

    summary["delta_total_route_duration"] = (
        summary["total_route_duration"]
        .astype(float)
        - float(
            baseline["total_route_duration"]
        )
    )

    summary["delta_max_driver_route_workload"] = (
        summary["max_driver_route_workload"]
        .astype(float)
        - float(
            baseline[
                "max_driver_route_workload"
            ]
        )
    )

    summary["delta_cv_active_route_workload"] = (
        summary["cv_active_route_workload"]
        .astype(float)
        - float(
            baseline[
                "cv_active_route_workload"
            ]
        )
    )

    return summary

ASSIGNMENT_OUTPUT_COLUMNS = [
    "equity_measure_type",
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

def polish_solution_routes(
    solution: Stage4Solution,
    ctx: Stage4Context,
) -> Stage4Solution:
    """
    Re-evaluate the final best solution using the higher final exact-route
    threshold before writing report-facing outputs.

    Stage 4 starts from Stage 3 repaired routes, which have already been
    routed and validated. Therefore, only driver-day routes changed by the
    ALNS search need to be final-polished. Re-polishing every route is
    computationally expensive and does not change untouched routes.
    """

    polished = solution.clone()

    if STAGE4_FINAL_POLISH_ONLY_CHANGED_ROUTES:

        keys = sorted(
            polished.changed_route_keys
        )

        if not keys:
            print(
                "Final Stage 4 polishing skipped: no route keys changed.",
                flush=True,
            )

            polished.evaluate_score(
                ctx
            )

            return polished

        print(
            "Final Stage 4 polishing changed routes only: "
            f"{len(keys)} route keys.",
            flush=True,
        )

    else:

        keys = list(
            polished.routes_by_key.keys()
        )

        print(
            "Final Stage 4 polishing all routes: "
            f"{len(keys)} route keys.",
            flush=True,
        )

    for key in keys:
        polished.reroute_key(
            key=key,
            ctx=ctx,
            exact_route_limit=STAGE4_FINAL_EXACT_ROUTE_LIMIT,
            mark_changed=False,
        )

    polished.evaluate_score(
        ctx
    )

    return polished

def run_stage_4_alns_solution(
    model_scope: str,
    lambda_tag: str,
    equity_measure_type: str = "maximum",
    max_iterations: int = STAGE4_MAX_ITERATIONS,
    random_seed: int = STAGE4_RANDOM_SEED,
    overwrite: bool = True,
) -> pd.DataFrame:
    
    output_dir = stage4_solution_dir(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    summary_path = (
        output_dir
        / "stage_4_solution_summary.csv"
    )

    if summary_path.exists() and not overwrite:
        return pd.read_csv(
            summary_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ctx = load_stage4_context(
        model_scope=model_scope,
        lambda_tag=lambda_tag,
        equity_measure_type=equity_measure_type,
    )

    print(
        f"\nRunning Stage 4 ALNS for "
        f"{model_scope} {equity_measure_type} {lambda_tag} "
        f"for {max_iterations:,} iterations."
    )

    best, iteration_log, operator_weights, initial = (
        run_alns_search(
            ctx=ctx,
            max_iterations=max_iterations,
            random_seed=random_seed,
        )
    )

    best = polish_solution_routes(
        solution=best,
        ctx=ctx,
    )

    route_summaries = best.route_summaries_df()
    route_stops = best.route_stops_df()
    driver_workloads = best.driver_workloads_df(
        ctx
    )

    solution_summary = create_stage4_solution_summary(
        ctx=ctx,
        initial=initial,
        best=best,
        iteration_log=iteration_log,
        random_seed=random_seed,
    )

    best.assignments[
        [
            column
            for column in ASSIGNMENT_OUTPUT_COLUMNS
            if column in best.assignments.columns
        ]
    ].to_csv(
        output_dir / "stage_4_assignments.csv",
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

    iteration_log.to_csv(
        output_dir / "alns_iteration_log.csv",
        index=False,
    )

    operator_weights.to_csv(
        output_dir / "alns_operator_weights.csv",
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
        "stage": "stage_4_alns",
        "model_scope": model_scope,
        "equity_measure_type": equity_measure_type,
        "lambda_tag": lambda_tag,
        "lambda_equity": ctx.lambda_equity,
        "source_stage_3_repaired_directory": str(
            stage3_repaired_solution_dir(
                model_scope=model_scope,
                lambda_tag=lambda_tag,
                equity_measure_type=equity_measure_type,
            ).relative_to(BASE_DIR)
        ),
        "max_iterations": int(
            max_iterations
        ),
        "random_seed": int(
            random_seed
        ),
        "final_polish_only_changed_routes":
            STAGE4_FINAL_POLISH_ONLY_CHANGED_ROUTES,
        "final_exact_route_limit":
            STAGE4_FINAL_EXACT_ROUTE_LIMIT,
        "destroy_operators": STAGE4_DESTROY_OPERATORS,
        "repair_operators": STAGE4_REPAIR_OPERATORS,
        "removal_fraction": STAGE4_REMOVAL_FRACTION,
        "min_removals": STAGE4_MIN_REMOVALS,
        "max_removals": STAGE4_MAX_REMOVALS,
        "candidate_drivers": STAGE4_CANDIDATE_DRIVERS,
        "max_targets_per_order": STAGE4_MAX_TARGETS_PER_ORDER,
        "search_exact_route_limit": STAGE4_SEARCH_EXACT_ROUTE_LIMIT,
        "final_exact_route_limit": STAGE4_FINAL_EXACT_ROUTE_LIMIT,
        "route_eval_cache_max_size": STAGE4_ROUTE_EVAL_CACHE_MAX_SIZE,
        "max_workload_deterioration_pct": (
            STAGE4_MAX_WORKLOAD_DETERIORATION_PCT
        ),
        "max_workload_deterioration_penalty": (
            STAGE4_MAX_WORKLOAD_DETERIORATION_PENALTY
        ),
    }

    with open(
        output_dir / "stage_4_manifest.json",
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
                column
                for column in [
                    "model_scope",
                    "equity_measure_type",
                    "lambda_tag",
                    "n_changed_route_keys",
                    "final_polish_only_changed_routes",
                    "n_changed_assignments",
                    "n_infeasible_routes",
                    "total_route_duration",
                    "delta_total_route_duration",
                    "stage4_selected_equity_metric_value",
                    "delta_selected_equity_metric_value",
                    "max_driver_route_workload",
                    "delta_max_driver_route_workload",
                    "cv_active_route_workload",
                ]
                if column in solution_summary.columns
            ]
        ].to_string(index=False)
    )

    return solution_summary

def run_stage_4_alns(
    representative_solutions: list[tuple[str, str]] | None = None,
    max_iterations: int = STAGE4_MAX_ITERATIONS,
    random_seed: int = STAGE4_RANDOM_SEED,
    overwrite: bool = True,
) -> pd.DataFrame:
    if representative_solutions is None:
        representative_solutions = (
            STAGE4_REPRESENTATIVE_SOLUTIONS
        )

    STAGE4_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for index, solution_key in enumerate(
        representative_solutions
    ):

        if len(solution_key) == 2:
            model_scope, lambda_tag = solution_key
            equity_measure_type = "maximum"

        elif len(solution_key) == 3:
            model_scope, equity_measure_type, lambda_tag = solution_key

        else:
            raise ValueError(
                "Stage 4 representative solution keys must be "
                "(model_scope, lambda_tag) or "
                "(model_scope, equity_measure_type, lambda_tag)."
            )

        summary = run_stage_4_alns_solution(
            model_scope=model_scope,
            lambda_tag=lambda_tag,
            equity_measure_type=equity_measure_type,
            max_iterations=max_iterations,
            random_seed=random_seed + index,
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
        STAGE4_RESULTS_DIR
        / "stage_4_master_summary.csv",
        index=False,
    )

    print(
        "\nStage 4 ALNS completed."
    )

    print(
        master[
            [
                column
                for column in [
                    "model_scope",
                    "equity_measure_type",
                    "lambda_tag",
                    "n_changed_assignments",
                    "n_infeasible_routes",
                    "total_route_duration",
                    "delta_total_route_duration",
                    "stage4_selected_equity_metric_value",
                    "delta_selected_equity_metric_value",
                    "max_driver_route_workload",
                    "delta_max_driver_route_workload",
                    "cv_active_route_workload",
                    "delta_cv_active_route_workload",
                ]
                if column in master.columns
            ]
        ].to_string(index=False)
    )

    return master


