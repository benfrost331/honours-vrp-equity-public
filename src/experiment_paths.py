from dataclasses import dataclass
from pathlib import Path

from src.config import (
    EQUITY_RESULTS_DIR,
    LAMBDA_RESULTS_DIR,
    REDUCED_LAMBDA_RESULTS_DIR,
)


@dataclass(frozen=True)
class LambdaExperimentPaths:
    """
    File locations for one lambda experiment.
    """

    results_dir: Path
    assignments_dir: Path
    driver_workloads_dir: Path
    day_workloads_dir: Path
    logs_dir: Path
    summary_path: Path
    manifest_path: Path


def make_lambda_experiment_paths(
    results_dir: Path,
) -> LambdaExperimentPaths:
    """
    Construct the standard output paths beneath a result folder.
    """

    return LambdaExperimentPaths(
        results_dir=results_dir,
        assignments_dir=results_dir / "assignments",
        driver_workloads_dir=results_dir / "driver_workloads",
        day_workloads_dir=results_dir / "day_workloads",
        logs_dir=results_dir / "logs",
        summary_path=results_dir / "lambda_solution_summary.csv",
        manifest_path=results_dir / "experiment_manifest.json",
    )


def make_equity_experiment_paths(
    equity_measure_type: str,
    model_scope: str = "full_instance",
) -> LambdaExperimentPaths:
    """
    Standard output path for a lambda sweep under one equity measure.
    """

    return make_lambda_experiment_paths(
        EQUITY_RESULTS_DIR
        / model_scope
        / equity_measure_type
    )


FULL_LAMBDA_PATHS = make_lambda_experiment_paths(
    LAMBDA_RESULTS_DIR
)

REDUCED_LAMBDA_PATHS = make_lambda_experiment_paths(
    REDUCED_LAMBDA_RESULTS_DIR
)