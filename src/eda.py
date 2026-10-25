import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR



# PLOT ORDER AND DRIVER LOCATIONS


def plot_order_driver_locations(
    orders: pd.DataFrame,
    drivers: pd.DataFrame
) -> None:
    """
    Plot order locations against driver base locations.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        orders["x"],
        orders["y"],
        alpha=0.45,
        s=12,
        label="Orders"
    )

    plt.scatter(
        drivers["x"],
        drivers["y"],
        marker="^",
        s=70,
        label="Driver base locations"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Order Locations vs Driver Base Locations")
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "order_driver_locations.png", dpi=300)
    plt.close()



# PLOT JOB TIME DISTRIBUTION


def plot_job_time_distribution(orders: pd.DataFrame) -> None:
    """
    Plot the distribution of order job times.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        orders["jobtime"],
        bins=30,
        edgecolor="black"
    )

    plt.xlabel("Job time")
    plt.ylabel("Number of orders")
    plt.title("Distribution of Job Times")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "job_time_distribution.png", dpi=300)
    plt.close()



# PLOT FRIDAY RESTRICTION COUNTS


def plot_friday_restriction_counts(orders: pd.DataFrame) -> None:
    """
    Plot number of orders with and without Friday restrictions.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    orders_plot = orders.copy()

    orders_plot["no_friday_flag"] = (
        orders_plot["nofriday"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
    )

    counts = (
        orders_plot["no_friday_flag"]
        .value_counts()
        .rename(index={False: "Friday allowed", True: "Friday not allowed"})
    )

    plt.figure(figsize=(7, 5))

    counts.plot(kind="bar")

    plt.xlabel("Friday delivery status")
    plt.ylabel("Number of orders")
    plt.title("Orders With and Without Friday Restrictions")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "friday_restriction_counts.png", dpi=300)
    plt.close()



# PLOT TRAVEL TIME VS DISTANCE


def plot_travel_time_vs_distance(routes: pd.DataFrame) -> None:
    """
    Plot travel time against travel distance.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        routes["travel_distance"],
        routes["travel_time"],
        alpha=0.25,
        s=8
    )

    plt.xlabel("Travel distance")
    plt.ylabel("Travel time")
    plt.title("Travel Time vs Travel Distance")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "travel_time_vs_distance.png", dpi=300)
    plt.close()



# RUN ALL INITIAL EDA FIGURES


def generate_initial_eda_figures(
    orders: pd.DataFrame,
    drivers: pd.DataFrame,
    routes: pd.DataFrame
) -> None:
    """
    Generate all initial Stage 1 EDA figures.
    """

    plot_order_driver_locations(orders, drivers)
    plot_job_time_distribution(orders)
    plot_friday_restriction_counts(orders)
    plot_travel_time_vs_distance(routes)