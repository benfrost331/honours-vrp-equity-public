import pandas as pd
from src.config import DATA_RAW_DIR



# LOAD RAW INPUT DATA


def load_orders(filename: str = "orders.csv") -> pd.DataFrame:
    """
    Load the raw orders dataset.
    """
    path = DATA_RAW_DIR / filename
    return pd.read_csv(path)


def load_drivers(filename: str = "drivers.csv") -> pd.DataFrame:
    """
    Load the raw drivers dataset.
    """
    path = DATA_RAW_DIR / filename
    return pd.read_csv(path)


def load_routes(filename: str = "routes.csv") -> pd.DataFrame:
    """
    Load the raw routes / travel matrix dataset.
    """
    path = DATA_RAW_DIR / filename
    return pd.read_csv(path)


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all core datasets required for the VRP project.
    """
    orders = load_orders()
    drivers = load_drivers()
    routes = load_routes()

    return orders, drivers, routes