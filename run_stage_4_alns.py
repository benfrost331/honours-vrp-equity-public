from __future__ import annotations

import argparse

import pandas as pd

from src.config import (
    EQUITY_LAMBDA_VALUES,
    EQUITY_MEASURE_TYPES,
    STAGE4_MAX_ITERATIONS,
    STAGE4_RANDOM_SEED,
    STAGE4_RESULTS_DIR,
)
from src.stage_4_alns import run_stage_4_alns_solution


def lambda_tag(lambda_value: float) -> str:
    return f"lambda_{lambda_value:.3f}".replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 4 ALNS routing improvement for selected "
            "equity-measure Stage 3 repaired solutions."
        )
    )

    parser.add_argument(
        "--equity-measures",
        nargs="+",
        default=EQUITY_MEASURE_TYPES,
        choices=EQUITY_MEASURE_TYPES,
        help="Equity measures to run.",
    )

    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=EQUITY_LAMBDA_VALUES,
        help="Lambda values to run, e.g. --lambda-values 0.3 0.6",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=STAGE4_MAX_ITERATIONS,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=STAGE4_RANDOM_SEED,
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("\nStage 4 ALNS run configuration")
    print("Equity measures:", args.equity_measures)
    print("Lambda values:", args.lambda_values)
    print("Max iterations:", args.max_iterations)
    print("Overwrite:", not args.no_overwrite)

    STAGE4_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    run_index = 0

    for equity_measure_type in args.equity_measures:
        for lambda_value in args.lambda_values:

            tag = lambda_tag(
                lambda_value
            )

            print(
                f"\nRunning Stage 4: "
                f"{equity_measure_type} {tag}",
                flush=True,
            )

            summary = run_stage_4_alns_solution(
                model_scope="full_instance",
                equity_measure_type=equity_measure_type,
                lambda_tag=tag,
                max_iterations=args.max_iterations,
                random_seed=args.random_seed + run_index,
                overwrite=not args.no_overwrite,
            )

            summaries.append(
                summary
            )

            run_index += 1

    if summaries:

        master = pd.concat(
            summaries,
            ignore_index=True,
        )

        master.to_csv(
            STAGE4_RESULTS_DIR / "stage_4_master_summary.csv",
            index=False,
        )

        print("\nStage 4 ALNS selected run completed.")
        print(
            master[
                [
                    column
                    for column in [
                        "model_scope",
                        "equity_measure_type",
                        "lambda_tag",
                        "n_changed_assignments",
                        "n_infeasible_routes",
                        "total_route_duration",
                        "delta_total_route_duration",
                        "stage4_selected_equity_metric_value",
                        "delta_selected_equity_metric_value",
                        "max_driver_route_workload",
                        "delta_max_driver_route_workload",
                        "cv_active_route_workload",
                    ]
                    if column in master.columns
                ]
            ].to_string(index=False)
        )