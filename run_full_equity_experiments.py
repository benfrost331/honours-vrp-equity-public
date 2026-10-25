from __future__ import annotations

import argparse

from src.config import (
    DEFAULT_EQUITY_RELATIVE_GAP,
    DEFAULT_EQUITY_SOLVER_NAME,
    DEFAULT_EQUITY_SOLVER_THREADS,
    DEFAULT_EQUITY_TIME_LIMIT_SECONDS,
    EQUITY_LAMBDA_VALUES,
    EQUITY_MEASURE_TYPES,
)
from src.experiment_paths import make_equity_experiment_paths
from src.lambda_experiment import run_lambda_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-instance lambda experiments for one or more "
            "workload-equity measures."
        )
    )

    parser.add_argument(
        "--equity-measures",
        nargs="+",
        default=EQUITY_MEASURE_TYPES,
        choices=EQUITY_MEASURE_TYPES,
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_EQUITY_RELATIVE_GAP,
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_EQUITY_TIME_LIMIT_SECONDS,
    )

    parser.add_argument(
        "--solver",
        type=str,
        default=DEFAULT_EQUITY_SOLVER_NAME,
        choices=["cbc", "highs", "gurobi"],
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_EQUITY_SOLVER_THREADS,
    )

    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional subset of lambda values to run, e.g. "
            "--lambda-values 0 0.1 0.2"
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for equity_measure_type in args.equity_measures:

        print(
            "\n"
            + "=" * 80
            + f"\nRunning full-instance equity experiment: "
            f"{equity_measure_type}"
            + "\n"
            + "=" * 80
        )

        paths = make_equity_experiment_paths(
            equity_measure_type=equity_measure_type,
            model_scope="full_instance",
        )

        run_lambda_experiment(
            lambda_values=(
                args.lambda_values
                if args.lambda_values is not None
                else EQUITY_LAMBDA_VALUES
            ),
            time_limit_seconds=args.time_limit,
            relative_gap=args.gap,
            resume=not args.no_resume,
            solver_name=args.solver,
            solver_threads=args.threads,
            paths=paths,
            equity_measure_type=equity_measure_type,
            manifest_metadata={
                "model_scope": "full_instance",
                "experiment_family": "multi_equity_measure",
            },
        )