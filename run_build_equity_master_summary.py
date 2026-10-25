from pathlib import Path

import pandas as pd

from src.config import EQUITY_RESULTS_DIR, EQUITY_MEASURE_TYPES


if __name__ == "__main__":

    rows = []

    for equity_measure_type in EQUITY_MEASURE_TYPES:

        summary_path = (
            EQUITY_RESULTS_DIR
            / "full_instance"
            / equity_measure_type
            / "lambda_solution_summary.csv"
        )

        if not summary_path.exists():
            print(f"Missing: {summary_path}")
            continue

        summary = pd.read_csv(summary_path)
        summary["equity_measure_type"] = equity_measure_type

        rows.append(summary)

    if not rows:
        raise RuntimeError(
            "No equity experiment summaries were found."
        )

    master = pd.concat(
        rows,
        ignore_index=True,
    )

    output_path = (
        EQUITY_RESULTS_DIR
        / "full_instance"
        / "equity_master_summary.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")
    print(
        master[
            [
                "equity_measure_type",
                "lambda_equity",
                "status",
                "total_estimated_assignment_time",
                "equity_penalty_metric_value",
                "max_driver_workload",
                "workload_range",
                "workload_mad_mean",
                "workload_minmax_deviation",
                "workload_gini_coefficient",
                "solve_runtime_seconds",
            ]
        ].to_string(index=False)
    )