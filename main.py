import pandas as pd

from src.load_data import load_all_data
from src.preprocess import clean_orders, clean_drivers, clean_routes, save_processed_data
from src.validation import check_missing_values, check_duplicate_rows, check_routes_symmetry
from src.feasibility import build_feasibility_matrix
from src.eda import generate_initial_eda_figures
from src.documentation import write_assumptions_file
from src.config import DATA_PROCESSED_DIR, TABLES_DIR



# RUN STAGE 1 PIPELINE


def main() -> None:
    """
    Run the full Stage 1 pipeline:
    - load raw data
    - validate raw data
    - clean data
    - save processed data
    - build order-day feasibility matrix
    - save validation summaries
    - generate initial EDA figures
    - write documented assumptions
    """

    print("Loading raw data...")
    orders_raw, drivers_raw, routes_raw = load_all_data()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    
    # VALIDATE RAW DATA
    

    print("Running validation checks...")

    missing_summary = pd.concat(
        [
            check_missing_values(orders_raw, "orders"),
            check_missing_values(drivers_raw, "drivers"),
            check_missing_values(routes_raw, "routes")
        ],
        ignore_index=True
    )

    duplicate_summary = pd.DataFrame(
        [
            check_duplicate_rows(orders_raw, "orders"),
            check_duplicate_rows(drivers_raw, "drivers"),
            check_duplicate_rows(routes_raw, "routes")
        ]
    )

    routes_symmetry_summary = check_routes_symmetry(routes_raw)

    missing_summary.to_csv(
        TABLES_DIR / "missing_value_summary.csv",
        index=False
    )

    duplicate_summary.to_csv(
        TABLES_DIR / "duplicate_row_summary.csv",
        index=False
    )

    routes_symmetry_summary.to_csv(
        TABLES_DIR / "routes_symmetry_summary.csv",
        index=False
    )

    print("Saved validation summaries.")

    
    # CLEAN DATA
    

    print("Cleaning datasets...")

    orders_clean = clean_orders(orders_raw)
    drivers_clean = clean_drivers(drivers_raw)
    routes_clean = clean_routes(routes_raw)

    save_processed_data(
        orders_clean=orders_clean,
        drivers_clean=drivers_clean,
        routes_clean=routes_clean
    )

    print("Saved cleaned datasets.")

    
    # BUILD FEASIBILITY MATRIX
    

    print("Building feasibility matrix...")

    feasibility_matrix = build_feasibility_matrix(orders_clean)

    feasibility_matrix.to_csv(
        DATA_PROCESSED_DIR / "feasibility_matrix.csv",
        index=False
    )

    feasibility_summary = (
        feasibility_matrix
        .groupby(["weekday", "is_feasible"])
        .size()
        .reset_index(name="n_order_day_combinations")
    )

    feasibility_summary.to_csv(
        TABLES_DIR / "feasibility_summary.csv",
        index=False
    )

    print("Saved feasibility matrix and feasibility summary.")

    
    # GENERATE INITIAL EDA FIGURES
    

    print("Generating initial EDA figures...")

    generate_initial_eda_figures(
        orders=orders_clean,
        drivers=drivers_clean,
        routes=routes_clean
    )

    print("Saved initial EDA figures.")

    
    # WRITE DOCUMENTED ASSUMPTIONS
    

    print("Writing documented assumptions...")

    write_assumptions_file()

    print("\nStage 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()