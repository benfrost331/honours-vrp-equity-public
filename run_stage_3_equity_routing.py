from __future__ import annotations

import argparse

from src.config import EQUITY_LAMBDA_VALUES, EQUITY_MEASURE_TYPES
from src.stage_3_pipeline import route_stored_assignment_solution


def lambda_tag(lambda_value: float) -> str:
    return f"lambda_{lambda_value:.3f}".replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 3 routing for equity-measure Stage 2 solutions."
    )

    parser.add_argument(
        "--equity-measures",
        nargs="+",
        default=EQUITY_MEASURE_TYPES,
        choices=EQUITY_MEASURE_TYPES,
        help="Equity measures to route.",
    )

    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=EQUITY_LAMBDA_VALUES,
        help="Lambda values to route, e.g. --lambda-values 0 0.1",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing Stage 3 routing outputs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\nStage 3 equity routing run configuration")
    print("Equity measures:", args.equity_measures)
    print("Lambda values:", args.lambda_values)
    print("Overwrite:", not args.no_overwrite)

    for equity_measure_type in args.equity_measures:
        for lambda_value in args.lambda_values:
            tag = lambda_tag(lambda_value)

            print(
                f"\nRouting Stage 3 solution: "
                f"{equity_measure_type} {tag}",
                flush=True,
            )

            route_stored_assignment_solution(
                model_scope="full_instance",
                lambda_tag=tag,
                equity_measure_type=equity_measure_type,
                overwrite=not args.no_overwrite,
            )