import pandas as pd
import matplotlib.pyplot as plt
from src.config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR

# CREATE DRIVER WORKLOAD SUMMARY

def create_driver_workload_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise total assigned workload by driver.
    """

    summary = (
        assignments
        .groupby(["driver_id", "driver_name"], as_index=False)
        .agg(
            n_orders=("orderid", "count"),
            total_job_time=("jobtime", "sum"),
            total_outbound_time=("outbound_time", "sum"),
            total_return_time=("return_time", "sum"),
            total_estimated_assignment_time=("estimated_assignment_time", "sum")
        )
    )

    return summary

# CREATE DAY WORKLOAD SUMMARY

def create_day_workload_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise total assigned workload by planning day.
    """

    summary = (
        assignments
        .groupby(["day_index", "weekday"], as_index=False)
        .agg(
            n_orders=("orderid", "count"),
            total_job_time=("jobtime", "sum"),
            total_outbound_time=("outbound_time", "sum"),
            total_return_time=("return_time", "sum"),
            total_estimated_assignment_time=("estimated_assignment_time", "sum")
        )
        .sort_values("day_index")
    )

    return summary

# CREATE ASSIGNMENT SUMMARY

def create_assignment_summary(
    assignments: pd.DataFrame,
    model_info: dict
) -> pd.DataFrame:
    """
    Create a one-row summary of the Stage 2 assignment model result.
    """

    driver_workload = create_driver_workload_summary(assignments)

    total_workload = assignments["estimated_assignment_time"].sum()
    mean_driver_workload = driver_workload["total_estimated_assignment_time"].mean()
    max_driver_workload = driver_workload["total_estimated_assignment_time"].max()
    min_driver_workload = driver_workload["total_estimated_assignment_time"].min()
    workload_range = max_driver_workload - min_driver_workload
    workload_std = driver_workload["total_estimated_assignment_time"].std()

    summary = pd.DataFrame({
        "status": [model_info["status"]],
        "lambda_equity": [model_info["lambda_equity"]],
        "objective_value": [model_info["objective_value"]],
        "n_assignments": [model_info["n_assignments"]],
        "n_drivers_used": [(driver_workload["n_orders"] > 0).sum()],
        "total_estimated_assignment_time": [total_workload],
        "mean_driver_workload": [mean_driver_workload],
        "max_driver_workload": [max_driver_workload],
        "min_driver_workload": [min_driver_workload],
        "workload_range": [workload_range],
        "workload_std": [workload_std]
    })

    return summary

# PLOT DRIVER WORKLOAD

def plot_driver_workload(driver_summary: pd.DataFrame) -> None:
    """
    Plot total estimated assignment time by driver.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = driver_summary.sort_values(
        "total_estimated_assignment_time",
        ascending=False
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        plot_data["driver_id"].astype(str),
        plot_data["total_estimated_assignment_time"]
    )

    plt.xlabel("Driver ID")
    plt.ylabel("Total estimated assignment time")
    plt.title("Stage 2 Total Workload by Driver")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "stage_2_driver_workload.png", dpi=300)
    plt.close()

# PLOT DAY WORKLOAD

def plot_day_workload(day_summary: pd.DataFrame) -> None:
    """
    Plot total estimated assignment time by planning day.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = (
        day_summary["day_index"].astype(str)
        + " - "
        + day_summary["weekday"].astype(str)
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        labels,
        day_summary["total_estimated_assignment_time"]
    )

    plt.xlabel("Planning day")
    plt.ylabel("Total estimated assignment time")
    plt.title("Stage 2 Total Workload by Day")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "stage_2_day_workload.png", dpi=300)
    plt.close()

# PLOT COST VS WORKLOAD DISTRIBUTION

def plot_cost_vs_order_count(driver_summary: pd.DataFrame) -> None:
    """
    Plot driver order count against workload.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        driver_summary["n_orders"],
        driver_summary["total_estimated_assignment_time"],
        s=45,
        alpha=0.75
    )

    plt.xlabel("Number of assigned orders")
    plt.ylabel("Total estimated assignment time")
    plt.title("Stage 2 Driver Order Count vs Workload")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "stage_2_cost_vs_workload.png", dpi=300)
    plt.close()

# WRITE STAGE 2 MODEL NOTES

def write_stage_2_model_notes(model_info: dict) -> None:
    """
    Write a documented note explaining the Stage 2 model.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text = f"""
# Stage 2 Model Notes

## Purpose

Stage 2 implements a baseline order-day-driver assignment model.

The model assigns every order to exactly one feasible planning day and one driver. It does not yet perform full route sequencing. Instead, it approximates assignment cost using travel from the driver's base location to the order location, the order job time, and return travel from the order location to the driver's base.

## Decision variable

`x[i, d, k] = 1` if order `i` is assigned to planning day `d` and driver `k`.

## Main constraints

1. Each order is assigned exactly once.
2. Orders may only be assigned to feasible days from the Stage 1 feasibility matrix.
3. Friday-restricted orders are not assigned to Fridays.
4. The total estimated workload assigned to each driver on each day may not exceed that driver's shift length.
5. The model tracks the maximum total driver workload over the planning horizon.

## Objective

The model minimises:

`total estimated assignment time + lambda_equity * maximum driver workload`

where `lambda_equity = {model_info["lambda_equity"]}`.

## Model status

Solver status: `{model_info["status"]}`

Objective value: `{model_info["objective_value"]}`

Total estimated assignment time: `{model_info["total_assignment_time"]}`

Maximum driver workload: `{model_info["max_driver_workload"]}`

Number of decision variables: `{model_info["n_variables"]}`

Number of assignments: `{model_info["n_assignments"]}`

## Interpretation

This model is a feasibility and workload-allocation bridge between Stage 1 preprocessing and later full route construction. It should not be interpreted as the final routing solution because order sequence, inter-order travel and daily route structure are not yet modelled.

"""

    path = OUTPUT_DIR / "stage_2_model_notes.md"

    with open(path, "w", encoding="utf-8") as file:
        file.write(text.strip())

    print(f"Saved Stage 2 model notes to: {path}")

# SAVE ALL STAGE 2 OUTPUTS

def save_stage_2_outputs(
    assignments: pd.DataFrame,
    model_info: dict
) -> None:
    """
    Save Stage 2 assignment outputs, summaries, figures and model notes.
    """

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    driver_summary = create_driver_workload_summary(assignments)
    day_summary = create_day_workload_summary(assignments)
    assignment_summary = create_assignment_summary(assignments, model_info)

    driver_summary.to_csv(
        TABLES_DIR / "stage_2_driver_workload_summary.csv",
        index=False
    )

    day_summary.to_csv(
        TABLES_DIR / "stage_2_day_workload_summary.csv",
        index=False
    )

    assignment_summary.to_csv(
        TABLES_DIR / "stage_2_assignment_summary.csv",
        index=False
    )

    plot_driver_workload(driver_summary)
    plot_day_workload(day_summary)
    plot_cost_vs_order_count(driver_summary)

    write_stage_2_model_notes(model_info)