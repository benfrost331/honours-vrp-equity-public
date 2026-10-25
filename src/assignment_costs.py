import pandas as pd
import numpy as np

# BUILD ROUTE LOOKUP DICTIONARIES

def build_route_lookup(routes: pd.DataFrame) -> dict:
    """
    Build a route lookup dictionary keyed by:
    (location_fromid, location_toid)

    The values contain travel_time and travel_distance.
    """

    route_lookup = {}

    for _, row in routes.iterrows():

        key = (
            int(row["location_fromid"]),
            int(row["location_toid"])
        )

        route_lookup[key] = {
            "travel_time": float(row["travel_time"]),
            "travel_distance": float(row["travel_distance"])
        }

    return route_lookup

# LOOKUP DIRECTED TRAVEL TIME

def get_travel_time(
    route_lookup: dict,
    location_from: int,
    location_to: int
) -> float:
    """
    Return directed travel time from location_from to location_to.

    If origin and destination are the same, travel time is zero.
    """

    if int(location_from) == int(location_to):
        return 0.0

    key = (int(location_from), int(location_to))

    if key not in route_lookup:
        return np.nan

    return route_lookup[key]["travel_time"]

# CREATE DRIVER-ORDER ASSIGNMENT COST MATRIX

def build_assignment_cost_matrix(
    orders: pd.DataFrame,
    drivers: pd.DataFrame,
    routes: pd.DataFrame
) -> pd.DataFrame:
    """
    Build an approximate assignment cost matrix.

    Each row is one order-driver combination.

    Estimated assignment time is:
        driver base -> order location
      + order job time
      + order location -> driver base

    This is not yet a sequenced route cost. It is a first-stage
    proxy for driver-order assignment effort.
    """

    route_lookup = build_route_lookup(routes)

    cost_rows = []

    for _, order in orders.iterrows():

        order_id = int(order["orderid"])
        order_location = int(order["locationid"])
        job_time = float(order["jobtime"])

        for driver_index, driver in drivers.reset_index(drop=True).iterrows():

            driver_id = driver_index + 1
            driver_name = driver["driver.name"]
            driver_location = int(driver["locationid"])

            outbound_time = get_travel_time(
                route_lookup=route_lookup,
                location_from=driver_location,
                location_to=order_location
            )

            return_time = get_travel_time(
                route_lookup=route_lookup,
                location_from=order_location,
                location_to=driver_location
            )

            if pd.isna(outbound_time) or pd.isna(return_time):
                estimated_assignment_time = np.nan
            else:
                estimated_assignment_time = outbound_time + job_time + return_time

            cost_rows.append({
                "orderid": order_id,
                "locationid": order_location,
                "driver_id": driver_id,
                "driver_name": driver_name,
                "driver_locationid": driver_location,
                "jobtime": job_time,
                "outbound_time": outbound_time,
                "return_time": return_time,
                "estimated_assignment_time": estimated_assignment_time
            })

    cost_matrix = pd.DataFrame(cost_rows)

    return cost_matrix