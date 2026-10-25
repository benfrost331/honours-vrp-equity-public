from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import STAGE4_RESULTS_DIR


if __name__ == "__main__":

    rows = []

    for path in sorted(
        (
            STAGE4_RESULTS_DIR
            / "full_instance"
        ).glob("*/lambda_*/stage_4_solution_summary.csv")
    ):
        rows.append(
            pd.read_csv(path)
        )

    if not rows:
        raise RuntimeError(
            "No Stage 4 solution summaries were found."
        )

    master = pd.concat(
        rows,
        ignore_index=True,
    )

    if {
        "equity_measure_type",
        "lambda_equity",
    }.issubset(master.columns):
        master = master.sort_values(
            [
                "equity_measure_type",
                "lambda_equity",
            ]
        ).reset_index(drop=True)

    output_path = (
        STAGE4_RESULTS_DIR
        / "stage_4_master_summary.csv"
    )

    master.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {len(master)}")