from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    LAMBDA_ASSIGNMENTS_DIR,
    LAMBDA_DAY_DIR,
    LAMBDA_DRIVER_DIR,
    LAMBDA_SUMMARY_PATH,
)

from src.experiment_paths import make_equity_experiment_paths


# GENERIC CACHED CSV LOADER


@st.cache_data(show_spinner=False)
def _read_csv_cached(
    path_string: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    """
    Read and cache a CSV file.

    modified_time_ns is deliberately included in the cache key.
    Replacing the underlying CSV therefore invalidates the cache
    automatically on the next Streamlit rerun.
    """

    del modified_time_ns

    return pd.read_csv(path_string)


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Validate a CSV path and load it through Streamlit's cache.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required dashboard file was not found: {path}"
        )

    return _read_csv_cached(
        path_string=str(path),
        modified_time_ns=path.stat().st_mtime_ns,
    )


def load_equity_lambda_summary(
    equity_measure_type: str,
) -> pd.DataFrame:
    paths = make_equity_experiment_paths(
        equity_measure_type=equity_measure_type,
        model_scope="full_instance",
    )

    return load_lambda_summary(
        summary_path=paths.summary_path,
    )


def load_equity_assignments(
    equity_measure_type: str,
    lambda_tag: str,
) -> pd.DataFrame:
    paths = make_equity_experiment_paths(
        equity_measure_type=equity_measure_type,
        model_scope="full_instance",
    )

    return load_assignments(
        lambda_tag=lambda_tag,
        assignments_dir=paths.assignments_dir,
    )


def load_equity_driver_workloads(
    equity_measure_type: str,
    lambda_tag: str,
) -> pd.DataFrame:
    paths = make_equity_experiment_paths(
        equity_measure_type=equity_measure_type,
        model_scope="full_instance",
    )

    return load_driver_workloads(
        lambda_tag=lambda_tag,
        driver_workloads_dir=paths.driver_workloads_dir,
    )


def load_equity_day_workloads(
    equity_measure_type: str,
    lambda_tag: str,
) -> pd.DataFrame:
    paths = make_equity_experiment_paths(
        equity_measure_type=equity_measure_type,
        model_scope="full_instance",
    )

    return load_day_workloads(
        lambda_tag=lambda_tag,
        day_workloads_dir=paths.day_workloads_dir,
    )


# DASHBOARD DATASETS


def load_lambda_summary(
    summary_path: Path = LAMBDA_SUMMARY_PATH,
) -> pd.DataFrame:
    """
    Load and sort a solution-level lambda summary.

    The default path preserves compatibility with the existing
    full-instance dashboard.
    """

    summary = _load_csv(summary_path)

    return (
        summary
        .sort_values("lambda_equity")
        .reset_index(drop=True)
    )


def load_assignments(
    lambda_tag: str,
    assignments_dir: Path = LAMBDA_ASSIGNMENTS_DIR,
) -> pd.DataFrame:
    """
    Load detailed order-day-driver assignments for one solution.
    """

    path = (
        assignments_dir
        / f"{lambda_tag}.csv"
    )

    return _load_csv(path)


def load_driver_workloads(
    lambda_tag: str,
    driver_workloads_dir: Path = LAMBDA_DRIVER_DIR,
) -> pd.DataFrame:
    """
    Load driver workload metrics for one stored solution.
    """

    path = (
        driver_workloads_dir
        / f"{lambda_tag}.csv"
    )

    return _load_csv(path)


def load_day_workloads(
    lambda_tag: str,
    day_workloads_dir: Path = LAMBDA_DAY_DIR,
) -> pd.DataFrame:
    """
    Load day-level workload metrics for one stored solution.
    """

    path = (
        day_workloads_dir
        / f"{lambda_tag}.csv"
    )

    return _load_csv(path)