import pandas as pd



# CREATE PLANNING CALENDAR


def create_planning_calendar() -> pd.DataFrame:
    """
    Create the 16-day working calendar for the project.

    The planning horizon starts on Tuesday and contains 16 working days.
    """

    weekday_pattern = [
        "Tue", "Wed", "Thu", "Fri", "Mon",
        "Tue", "Wed", "Thu", "Fri", "Mon",
        "Tue", "Wed", "Thu", "Fri", "Mon",
        "Tue"
    ]

    calendar = pd.DataFrame({
        "day_index": range(1, 17),
        "weekday": weekday_pattern,
        "is_working_day": True
    })

    calendar["is_friday"] = calendar["weekday"] == "Fri"

    return calendar



# BUILD ORDER-DAY FEASIBILITY MATRIX


def build_feasibility_matrix(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Build the long-format order-day feasibility matrix.

    Each row corresponds to one order-day combination.
    """

    feasibility_rows = []
    calendar = create_planning_calendar()

    for _, order in orders.iterrows():

        order_id = order["orderid"]

        # NoFriday is interpreted as:
        # 1 / TRUE / Yes  = order cannot be delivered on Friday
        # 0 / FALSE / No  = order can be delivered on Friday
        no_friday_value = str(order["nofriday"]).strip().lower()

        no_friday = no_friday_value in ["1", "true", "t", "yes", "y"]

        for _, day in calendar.iterrows():

            is_feasible = True

            if day["is_friday"] and no_friday:
                is_feasible = False

            feasibility_rows.append({
                "orderid": order_id,
                "day_index": day["day_index"],
                "weekday": day["weekday"],
                "is_friday": day["is_friday"],
                "no_friday": no_friday,
                "is_feasible": is_feasible
            })

    feasibility_matrix = pd.DataFrame(feasibility_rows)

    return feasibility_matrix