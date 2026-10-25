from __future__ import annotations

from itertools import permutations
from typing import Iterable

import numpy as np
import pandas as pd

from src.assignment_costs import (
    build_route_lookup,
    get_travel_time,
)
from src.config import (
    DRIVER_SHIFT_END,
    DRIVER_SHIFT_START,
    ORDER_SERVICE_END,
    ORDER_SERVICE_START,
    STAGE3_EXACT_ROUTE_LIMIT,
)


MISSING_ARC_PENALTY = 1_000_000.0


def prepare_driver_table(
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the internal driver_id used by the optimisation model.
    """

    drivers_prepared = (
        drivers
        .copy()
        .reset_index(drop=True)
    )

    drivers_prepared["driver_id"] = (
        drivers_prepared.index + 1
    )

    return drivers_prepared


def prepare_assignments_for_routing(
    assignments: pd.DataFrame,
    orders: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich stored assignment rows with order time-window and
    driver shift-window information required for route scheduling.
    """

    assignments = assignments.copy()
    orders = orders.copy()
    drivers = prepare_driver_table(
        drivers
    )

    assignments["orderid"] = (
        assignments["orderid"].astype(int)
    )

    assignments["driver_id"] = (
        assignments["driver_id"].astype(int)
    )

    order_columns = [
        "orderid",
        "timefrom",
        "timeto",
    ]

    enriched = assignments.merge(
        orders[order_columns],
        on="orderid",
        how="left",
    )

    driver_columns = [
        "driver_id",
        "shiftstart",
        "shiftend",
        "shift_length",
    ]

    enriched = enriched.merge(
        drivers[driver_columns],
        on="driver_id",
        how="left",
    )

    enriched["timefrom"] = (
        enriched["timefrom"]
        .fillna(ORDER_SERVICE_START)
        .astype(float)
    )

    enriched["timeto"] = (
        enriched["timeto"]
        .fillna(ORDER_SERVICE_END)
        .astype(float)
    )

    enriched["shiftstart"] = (
        enriched["shiftstart"]
        .fillna(DRIVER_SHIFT_START)
        .astype(float)
    )

    enriched["shiftend"] = (
        enriched["shiftend"]
        .fillna(DRIVER_SHIFT_END)
        .astype(float)
    )

    enriched["shift_length"] = (
        enriched["shift_length"]
        .fillna(
            enriched["shiftend"]
            - enriched["shiftstart"]
        )
        .astype(float)
    )

    return enriched


def route_score(
    route_summary: dict,
) -> tuple[float, float, float, float]:
    """
    Lexicographic route score.

    Feasibility is prioritised first, then duration, then travel.
    """

    return (
        float(route_summary["missing_arc_count"]),
        float(route_summary["total_violation_minutes"]),
        float(route_summary["route_duration"]),
        float(route_summary["total_travel_time"]),
    )

def compute_route_departure_time_for_first_stop(
    shift_start: float,
    first_window_start: float,
    travel_time_to_first_stop: float,
) -> float:
    """
    Delay route departure where possible so that the driver reaches the
    first customer when the first service window opens.

    If the first-stop travel time is longer than the gap between shift start
    and the first service-window opening time, the driver still departs at
    shift start.

    This only removes avoidable waiting before the first stop. Waiting later
    in the route is still calculated normally.
    """

    shift_start = float(shift_start)
    first_window_start = float(first_window_start)
    travel_time_to_first_stop = float(travel_time_to_first_stop)

    if (
        pd.isna(shift_start)
        or pd.isna(first_window_start)
        or pd.isna(travel_time_to_first_stop)
    ):
        return shift_start

    latest_safe_departure = (
        first_window_start
        - travel_time_to_first_stop
    )

    return max(
        shift_start,
        latest_safe_departure,
    )

def evaluate_sequence(
    sequence: list[int],
    group: pd.DataFrame,
    route_lookup: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate a directed driver-day route sequence.

    The route starts and ends at the driver's base. For the first stop,
    the scheduler delays departure where possible so that the driver reaches
    the first customer when its service window opens. If the first-stop travel
    time is too long for that, the driver departs at shift start. Later waiting,
    time-window violations and shift-end violations are still calculated
    normally.
    """

    group = group.copy()

    if group.empty:
        raise ValueError(
            "Cannot evaluate an empty driver-day group."
        )

    driver_id = int(
        group["driver_id"].iloc[0]
    )

    driver_name = str(
        group["driver_name"].iloc[0]
    )

    day_index = int(
        group["day_index"].iloc[0]
    )

    weekday = str(
        group["weekday"].iloc[0]
    )

    driver_base = int(
        group["driver_locationid"].iloc[0]
    )

    shift_start = float(
        group["shiftstart"].iloc[0]
    )

    shift_end = float(
        group["shiftend"].iloc[0]
    )

    order_lookup = {
        int(row.orderid): row
        for row in group.itertuples()
    }

    current_location = driver_base
    route_departure_time = shift_start
    current_time = route_departure_time

    total_travel_time = 0.0
    total_service_time = 0.0
    total_waiting_time = 0.0
    time_window_violation = 0.0
    missing_arc_count = 0

    stop_rows = []

    for stop_number, order_id in enumerate(
        sequence,
        start=1,
    ):

        row = order_lookup[int(order_id)]

        order_location = int(
            row.order_locationid
        )

        travel_time = get_travel_time(
            route_lookup=route_lookup,
            location_from=current_location,
            location_to=order_location,
        )

        missing_arc = bool(
            pd.isna(travel_time)
        )

        if missing_arc:
            missing_arc_count += 1
            travel_time = MISSING_ARC_PENALTY

        travel_time = float(
            travel_time
        )

        window_start = float(
            row.timefrom
        )

        window_end = float(
            row.timeto
        )

        if stop_number == 1:
            route_departure_time = (
                compute_route_departure_time_for_first_stop(
                    shift_start=shift_start,
                    first_window_start=window_start,
                    travel_time_to_first_stop=travel_time,
                )
            )

            current_time = route_departure_time

        arrival_time = (
            current_time + travel_time
        )

        waiting_time = max(
            0.0,
            window_start - arrival_time,
        )

        service_start_time = (
            arrival_time + waiting_time
        )

        late_start_violation = max(
            0.0,
            service_start_time - window_end,
        )

        service_time = float(
            row.jobtime
        )

        service_end_time = (
            service_start_time + service_time
        )

        departure_time = service_end_time

        total_travel_time += travel_time
        total_waiting_time += waiting_time
        total_service_time += service_time
        time_window_violation += late_start_violation

        stop_rows.append({
            "driver_id": driver_id,
            "driver_name": driver_name,
            "day_index": day_index,
            "weekday": weekday,
            "stop_number": stop_number,
            "orderid": int(order_id),
            "locationid": order_location,
            "previous_locationid": current_location,
            "travel_time_from_previous": travel_time,
            "arrival_time": arrival_time,
            "window_start": window_start,
            "window_end": window_end,
            "waiting_time": waiting_time,
            "service_start_time": service_start_time,
            "service_time": service_time,
            "service_end_time": service_end_time,
            "departure_time": departure_time,
            "late_start_violation": late_start_violation,
            "missing_inbound_arc": missing_arc,
        })

        current_location = order_location
        current_time = departure_time

    return_travel_time = get_travel_time(
        route_lookup=route_lookup,
        location_from=current_location,
        location_to=driver_base,
    )

    missing_return_arc = bool(
        pd.isna(return_travel_time)
    )

    if missing_return_arc:
        missing_arc_count += 1
        return_travel_time = MISSING_ARC_PENALTY

    return_travel_time = float(
        return_travel_time
    )

    total_travel_time += return_travel_time

    return_arrival_time = (
        current_time + return_travel_time
    )

    route_duration = (
        return_arrival_time - route_departure_time
    )

    shift_violation = max(
        0.0,
        return_arrival_time - shift_end,
    )

    total_violation_minutes = (
        time_window_violation
        + shift_violation
        + missing_arc_count * MISSING_ARC_PENALTY
    )

    feasible = (
        missing_arc_count == 0
        and total_violation_minutes <= 1e-6
    )

    proxy_assignment_time = float(
        group["estimated_assignment_time"].sum()
    )

    summary = {
        "driver_id": driver_id,
        "driver_name": driver_name,
        "day_index": day_index,
        "weekday": weekday,
        "n_orders": int(len(sequence)),
        "driver_base_locationid": driver_base,
        "shift_start": shift_start,
        "shift_end": shift_end,
        "shift_length": float(
            shift_end - shift_start
        ),
        "departure_time": route_departure_time,
        "return_arrival_time": return_arrival_time,
        "total_travel_time": total_travel_time,
        "total_service_time": total_service_time,
        "total_waiting_time": total_waiting_time,
        "route_duration": route_duration,
        "proxy_assignment_time": proxy_assignment_time,
        "time_window_violation": time_window_violation,
        "shift_violation": shift_violation,
        "total_violation_minutes": total_violation_minutes,
        "missing_arc_count": int(missing_arc_count),
        "missing_return_arc": missing_return_arc,
        "route_feasible": bool(feasible),
        "route_sequence": "-".join(
            str(order_id)
            for order_id in sequence
        ),
    }

    stops = pd.DataFrame(
        stop_rows
    )

    return summary, stops


def best_exact_route(
    order_ids: list[int],
    group: pd.DataFrame,
    route_lookup: dict,
) -> tuple[list[int], dict, pd.DataFrame]:
    """
    Enumerate all permutations for small driver-day groups.
    """

    best_sequence = None
    best_summary = None
    best_stops = None
    best_score = None

    for candidate in permutations(order_ids):

        candidate_sequence = list(
            candidate
        )

        summary, stops = evaluate_sequence(
            sequence=candidate_sequence,
            group=group,
            route_lookup=route_lookup,
        )

        score = route_score(
            summary
        )

        if (
            best_score is None
            or score < best_score
        ):
            best_sequence = candidate_sequence
            best_summary = summary
            best_stops = stops
            best_score = score

    assert best_sequence is not None
    assert best_summary is not None
    assert best_stops is not None

    return (
        best_sequence,
        best_summary,
        best_stops,
    )


def cheapest_insertion_route(
    order_ids: list[int],
    group: pd.DataFrame,
    route_lookup: dict,
) -> tuple[list[int], dict, pd.DataFrame]:
    """
    Construct an initial route using directed cheapest insertion.

    Candidate insertions are evaluated by full schedule recalculation.
    """

    route: list[int] = []
    remaining = set(
        int(order_id)
        for order_id in order_ids
    )

    best_summary = None
    best_stops = None

    while remaining:

        best_candidate_route = None
        best_candidate_summary = None
        best_candidate_stops = None
        best_candidate_score = None
        best_order = None

        for order_id in sorted(remaining):

            for position in range(
                len(route) + 1
            ):

                candidate_route = (
                    route[:position]
                    + [order_id]
                    + route[position:]
                )

                summary, stops = evaluate_sequence(
                    sequence=candidate_route,
                    group=group,
                    route_lookup=route_lookup,
                )

                score = route_score(
                    summary
                )

                if (
                    best_candidate_score is None
                    or score < best_candidate_score
                ):
                    best_candidate_score = score
                    best_candidate_route = candidate_route
                    best_candidate_summary = summary
                    best_candidate_stops = stops
                    best_order = order_id

        assert best_candidate_route is not None
        assert best_candidate_summary is not None
        assert best_candidate_stops is not None
        assert best_order is not None

        route = best_candidate_route
        remaining.remove(
            best_order
        )

        best_summary = best_candidate_summary
        best_stops = best_candidate_stops

    assert best_summary is not None
    assert best_stops is not None

    return route, best_summary, best_stops


def generate_local_search_neighbours(
    sequence: list[int],
) -> Iterable[list[int]]:
    """
    Generate relocate, swap and directed segment-reversal neighbours.
    """

    n = len(sequence)

    seen: set[tuple[int, ...]] = set()

    # Relocate one order.
    for i in range(n):

        reduced = (
            sequence[:i]
            + sequence[i + 1:]
        )

        for j in range(
            len(reduced) + 1
        ):

            candidate = (
                reduced[:j]
                + [sequence[i]]
                + reduced[j:]
            )

            key = tuple(candidate)

            if (
                candidate != sequence
                and key not in seen
            ):
                seen.add(key)
                yield candidate

    # Swap two orders.
    for i in range(n):
        for j in range(i + 1, n):

            candidate = sequence.copy()
            candidate[i], candidate[j] = (
                candidate[j],
                candidate[i],
            )

            key = tuple(candidate)

            if key not in seen:
                seen.add(key)
                yield candidate

    # Directed 2-opt-style segment reversal.
    # Because costs are asymmetric, the full route is recalculated.
    for i in range(n):
        for j in range(i + 2, n + 1):

            candidate = (
                sequence[:i]
                + list(reversed(sequence[i:j]))
                + sequence[j:]
            )

            key = tuple(candidate)

            if (
                candidate != sequence
                and key not in seen
            ):
                seen.add(key)
                yield candidate


def improve_route_local_search(
    sequence: list[int],
    group: pd.DataFrame,
    route_lookup: dict,
    max_iterations: int = 50,
) -> tuple[list[int], dict, pd.DataFrame]:
    """
    Improve a route using greedy best-improvement local search.
    """

    current_sequence = sequence.copy()

    current_summary, current_stops = evaluate_sequence(
        sequence=current_sequence,
        group=group,
        route_lookup=route_lookup,
    )

    current_score = route_score(
        current_summary
    )

    for _ in range(max_iterations):

        best_sequence = current_sequence
        best_summary = current_summary
        best_stops = current_stops
        best_score = current_score

        for neighbour in generate_local_search_neighbours(
            current_sequence
        ):

            summary, stops = evaluate_sequence(
                sequence=neighbour,
                group=group,
                route_lookup=route_lookup,
            )

            score = route_score(
                summary
            )

            if score < best_score:
                best_sequence = neighbour
                best_summary = summary
                best_stops = stops
                best_score = score

        if best_sequence == current_sequence:
            break

        current_sequence = best_sequence
        current_summary = best_summary
        current_stops = best_stops
        current_score = best_score

    return (
        current_sequence,
        current_summary,
        current_stops,
    )


def build_route_for_driver_day(
    group: pd.DataFrame,
    route_lookup: dict,
    exact_route_limit: int = STAGE3_EXACT_ROUTE_LIMIT,
) -> tuple[dict, pd.DataFrame]:
    """
    Build and validate one driver-day route.

    For small groups, exact enumeration is used. Larger groups use
    cheapest insertion followed by local search.
    """

    group = (
        group
        .copy()
        .sort_values("orderid")
        .reset_index(drop=True)
    )

    order_ids = (
        group["orderid"]
        .astype(int)
        .tolist()
    )

    if not order_ids:
        raise ValueError(
            "Cannot route a driver-day group with no orders."
        )

    if len(order_ids) <= exact_route_limit:

        sequence, summary, stops = best_exact_route(
            order_ids=order_ids,
            group=group,
            route_lookup=route_lookup,
        )

        method = "exact_enumeration"

    else:

        sequence, summary, stops = cheapest_insertion_route(
            order_ids=order_ids,
            group=group,
            route_lookup=route_lookup,
        )

        sequence, summary, stops = improve_route_local_search(
            sequence=sequence,
            group=group,
            route_lookup=route_lookup,
        )

        method = "cheapest_insertion_local_search"

    summary["construction_method"] = method

    stops = stops.copy()
    stops["construction_method"] = method

    return summary, stops