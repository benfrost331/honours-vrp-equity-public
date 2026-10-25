import pandas as pd

from src.config import DATA_PROCESSED_DIR
from src.assignment_costs import build_assignment_cost_matrix
from src.assignment_model import solve_assignment_model
from src.assignment_outputs import save_stage_2_outputs



# RUN STAGE 2 PIPELINE


def run_stage_2_pipeline(
    lambda_equity: float = 0.10,
    time_limit_seconds: int = 300
) -> None:
    """
    Run the full Stage 2 assignment pipeline.
    """

    print("Loading Stage 1 processed data...")

    orders = pd.read_csv(DATA_PROCESSED_DIR / "orders_clean.csv")
    drivers = pd.read_csv(DATA_PROCESSED_DIR / "drivers_clean.csv")
    routes = pd.read_csv(DATA_PROCESSED_DIR / "routes_clean.csv")
    feasibility_matrix = pd.read_csv(DATA_PROCESSED_DIR / "feasibility_matrix.csv")

    print("Building assignment cost matrix...")

    assignment_cost_matrix = build_assignment_cost_matrix(
        orders=orders,
        drivers=drivers,
        routes=routes
    )

    assignment_cost_matrix.to_csv(
        DATA_PROCESSED_DIR / "assignment_cost_matrix.csv",
        index=False
    )

    print(f"Assignment cost matrix rows: {len(assignment_cost_matrix)}")

    missing_costs = assignment_cost_matrix["estimated_assignment_time"].isna().sum()

    print(f"Missing assignment costs: {missing_costs}")

    if missing_costs > 0:
        print(
            "Warning: Some driver-order combinations have missing travel times. "
            "These combinations will be excluded from the optimisation model."
        )

    print("Solving Stage 2 assignment model...")

    assignments, model_info = solve_assignment_model(
        orders=orders,
        drivers=drivers,
        feasibility_matrix=feasibility_matrix,
        assignment_cost_matrix=assignment_cost_matrix,
        lambda_equity=lambda_equity,
        time_limit_seconds=time_limit_seconds
    )

    assignments.to_csv(
        DATA_PROCESSED_DIR / "stage_2_assignments.csv",
        index=False
    )

    print("Saving Stage 2 outputs...")

    save_stage_2_outputs(
        assignments=assignments,
        model_info=model_info
    )

    print("\nStage 2 pipeline completed.")
    print(model_info)