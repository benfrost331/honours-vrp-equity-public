import pandas as pd
from src.config import DATA_PROCESSED_DIR



# CLEAN ORDERS


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the orders dataset.
    """

    orders_clean = orders.copy()

    # Standardise column names.
    orders_clean.columns = (
        orders_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Remove exact duplicate rows.
    orders_clean = orders_clean.drop_duplicates()

    return orders_clean



# CLEAN DRIVERS


def clean_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the drivers dataset.
    """

    drivers_clean = drivers.copy()

    drivers_clean.columns = (
        drivers_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    drivers_clean = drivers_clean.drop_duplicates()

    return drivers_clean



# CLEAN ROUTES


def clean_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the routes / travel matrix dataset.
    """

    routes_clean = routes.copy()

    routes_clean.columns = (
        routes_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    routes_clean = routes_clean.drop_duplicates()

    return routes_clean



# SAVE PROCESSED DATA


def save_processed_data(
    orders_clean: pd.DataFrame,
    drivers_clean: pd.DataFrame,
    routes_clean: pd.DataFrame
) -> None:
    """
    Save cleaned datasets to the processed data folder.
    """

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    orders_clean.to_csv(DATA_PROCESSED_DIR / "orders_clean.csv", index=False)
    drivers_clean.to_csv(DATA_PROCESSED_DIR / "drivers_clean.csv", index=False)
    routes_clean.to_csv(DATA_PROCESSED_DIR / "routes_clean.csv", index=False)