from src.config import OUTPUT_DIR

# WRITE DOCUMENTED ASSUMPTIONS

def write_assumptions_file() -> None:
    """
    Save the main Stage 1 modelling and preprocessing assumptions.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assumptions_text = """
# Stage 1 Documented Assumptions

## Data structure assumptions

1. `orders.csv`, `drivers.csv` and `routes.csv` are the three raw input datasets.
2. `orders.orderId` uniquely identifies each customer order.
3. `orders.locationid` links each order to a location node.
4. `drivers.locationid` identifies the base location of each driver.
5. `routes.location_fromId` and `routes.location_toId` define directed origin-destination travel arcs.
6. `routes.travel_time` is measured in minutes.
7. `routes.travel_distance` is measured in miles.
8. `orders.JobTime` is measured in minutes.

## Planning horizon assumptions

1. The planning horizon contains 16 working days.
2. The working-day sequence is assumed to be:
   Tue, Wed, Thu, Fri, Mon, repeated across the available planning horizon.
3. Weekends are excluded from the optimisation horizon.

## Time-window assumptions

1. Order service windows are represented by `TimeFrom` and `TimeTo`.
2. Driver shift windows are represented by `ShiftStart` and `ShiftEnd`.
3. The initial feasibility matrix only handles weekday and Friday feasibility.
4. Detailed time-window and route-duration feasibility will be handled in later optimisation stages.

## Friday restriction assumptions

1. `NoFriday = 1`, `TRUE`, or `Yes` means the order may not be scheduled on Friday.
2. `NoFriday = 0`, `FALSE`, or `No` means the order may be scheduled on Friday.
3. Friday-restricted orders are marked infeasible for Friday planning days.

## Routing assumptions

1. The routes matrix is treated as directed.
2. Travel from location i to location j is not assumed to equal travel from j to i.
3. No vehicle capacity constraint is included at this stage.
4. The initial cost focus is travel time plus service time.

## Workload-equity assumptions

1. Driver workload is evaluated over the full planning horizon.
2. Stage 2 measures workload using estimated assignment time, incorporating service duration and directed travel-time estimates.
3. Subsequent routing stages evaluate workloads using the constructed daily routes.
"""

    path = OUTPUT_DIR / "stage_1_assumptions.md"

    with open(path, "w", encoding="utf-8") as file:
        file.write(assumptions_text.strip())

    print(f"Saved assumptions file to: {path}")