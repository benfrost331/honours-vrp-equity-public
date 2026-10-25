from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_REDUCED_DIR,
    REDUCED_CANDIDATE_COST_PATH,
    REDUCED_CANDIDATE_K,
    REDUCED_CANDIDATE_MANIFEST_PATH,
    REDUCED_CANDIDATE_MAP_PATH,
)
from src.experiment_paths import FULL_LAMBDA_PATHS
from src.lambda_experiment import lambda_tag


ANCHOR_LAMBDAS = (
    0.0,
    0.1,
    0.2,
    0.3,
)


def _load_anchor_assignments(
    n_orders: int,
) -> dict[float, pd.DataFrame]:
    """
    Load validated full-instance assignments used to check that the
    reduced candidate graph retains the known full solutions.
    """

    anchors: dict[float, pd.DataFrame] = {}

    for lambda_value in ANCHOR_LAMBDAS:

        tag = lambda_tag(lambda_value)

        path = (
            FULL_LAMBDA_PATHS.assignments_dir
            / f"{tag}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Required full-instance anchor was not found: {path}"
            )

        assignments = pd.read_csv(path)

        if len(assignments) != n_orders:
            raise ValueError(
                f"{path} contains {len(assignments)} assignments; "
                f"expected {n_orders}."
            )

        anchors[lambda_value] = assignments

    return anchors


def build_reduced_candidate_graph(
    candidate_k: int = REDUCED_CANDIDATE_K,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a restricted assignment-cost graph.

    All orders, drivers, planning days and capacity constraints remain
    unchanged. Each order retains its candidate_k lowest-cost drivers.

    Drivers used in the validated full-instance solutions for lambda
    0.0 to 0.3 are also retained as anchors.
    """

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")

    if (
        REDUCED_CANDIDATE_COST_PATH.exists()
        and REDUCED_CANDIDATE_MANIFEST_PATH.exists()
        and not overwrite
    ):
        with open(
            REDUCED_CANDIDATE_MANIFEST_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        costs = pd.read_csv(
            REDUCED_CANDIDATE_COST_PATH
        )

        return costs, manifest

    DATA_REDUCED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    orders = pd.read_csv(
        DATA_PROCESSED_DIR / "orders_clean.csv"
    )

    drivers = pd.read_csv(
        DATA_PROCESSED_DIR / "drivers_clean.csv"
    )

    feasibility = pd.read_csv(
        DATA_PROCESSED_DIR / "feasibility_matrix.csv"
    )

    full_costs = pd.read_csv(
        DATA_PROCESSED_DIR / "assignment_cost_matrix.csv"
    )

    full_costs = (
        full_costs
        .dropna(subset=["estimated_assignment_time"])
        .copy()
    )

    full_costs["orderid"] = (
        full_costs["orderid"].astype(int)
    )

    full_costs["driver_id"] = (
        full_costs["driver_id"].astype(int)
    )

    full_costs = (
        full_costs
        .sort_values(
            [
                "orderid",
                "estimated_assignment_time",
                "driver_id",
            ]
        )
        .reset_index(drop=True)
    )

    full_costs["candidate_rank"] = (
        full_costs
        .groupby("orderid")
        .cumcount()
        + 1
    )

    top_k_pairs = (
        full_costs.loc[
            full_costs["candidate_rank"] <= candidate_k,
            ["orderid", "driver_id"],
        ]
        .drop_duplicates()
    )

    anchors = _load_anchor_assignments(
        n_orders=len(orders)
    )

    anchor_pairs = (
        pd.concat(
            [
                assignments[
                    ["orderid", "driver_id"]
                ]
                for assignments in anchors.values()
            ],
            ignore_index=True,
        )
        .astype(int)
        .drop_duplicates()
    )

    candidate_pairs = (
        pd.concat(
            [top_k_pairs, anchor_pairs],
            ignore_index=True,
        )
        .drop_duplicates(
            ["orderid", "driver_id"]
        )
    )

    reduced_costs = (
        full_costs
        .merge(
            candidate_pairs,
            on=["orderid", "driver_id"],
            how="inner",
        )
        .sort_values(
            ["orderid", "candidate_rank"]
        )
        .reset_index(drop=True)
    )

    anchor_index = pd.MultiIndex.from_frame(
        anchor_pairs[
            ["orderid", "driver_id"]
        ]
    )

    reduced_index = pd.MultiIndex.from_frame(
        reduced_costs[
            ["orderid", "driver_id"]
        ]
    )

    reduced_costs["is_top_k"] = (
        reduced_costs["candidate_rank"]
        <= candidate_k
    )

    reduced_costs["is_anchor_driver"] = (
        reduced_index.isin(anchor_index)
    )

    candidate_counts = (
        reduced_costs
        .groupby("orderid")
        .size()
    )

    missing_orders = set(
        orders["orderid"].astype(int)
    ) - set(candidate_counts.index.astype(int))

    if missing_orders:
        raise RuntimeError(
            f"{len(missing_orders)} orders have no candidate drivers."
        )

    feasible_day_counts = (
        feasibility.loc[
            feasibility["is_feasible"].astype(bool)
        ]
        .groupby("orderid")
        .size()
    )

    full_candidate_counts = (
        full_costs
        .groupby("orderid")
        .size()
    )

    full_variable_count = int(
        (
            full_candidate_counts
            * feasible_day_counts
        ).sum()
    )

    reduced_variable_count = int(
        (
            candidate_counts
            * feasible_day_counts
        ).sum()
    )

    reduced_pair_index = set(
        map(
            tuple,
            reduced_costs[
                ["orderid", "driver_id"]
            ]
            .astype(int)
            .to_numpy(),
        )
    )

    anchor_coverage: dict[str, dict] = {}

    for lambda_value, assignments in anchors.items():

        assignment_pairs = set(
            map(
                tuple,
                assignments[
                    ["orderid", "driver_id"]
                ]
                .astype(int)
                .to_numpy(),
            )
        )

        missing_pairs = (
            assignment_pairs
            - reduced_pair_index
        )

        anchor_coverage[str(lambda_value)] = {
            "assignments": int(len(assignments)),
            "missing_pairs": int(len(missing_pairs)),
            "fully_retained": len(missing_pairs) == 0,
        }

        if missing_pairs:
            raise RuntimeError(
                f"The lambda={lambda_value} full solution is not "
                "fully retained in the reduced candidate graph."
            )

    reduced_costs.to_csv(
        REDUCED_CANDIDATE_COST_PATH,
        index=False,
    )

    reduced_costs[
        [
            "orderid",
            "driver_id",
            "driver_name",
            "driver_locationid",
            "estimated_assignment_time",
            "candidate_rank",
            "is_top_k",
            "is_anchor_driver",
        ]
    ].to_csv(
        REDUCED_CANDIDATE_MAP_PATH,
        index=False,
    )

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "instance_type": "restricted_candidate_graph",
        "candidate_rule": "lowest_estimated_assignment_time",
        "candidate_k": int(candidate_k),
        "anchor_lambdas": list(ANCHOR_LAMBDAS),
        "n_orders": int(len(orders)),
        "n_drivers": int(len(drivers)),
        "planning_days": int(
            feasibility["day_index"].nunique()
        ),
        "full_order_driver_arcs": int(
            len(full_costs)
        ),
        "reduced_order_driver_arcs": int(
            len(reduced_costs)
        ),
        "full_binary_variables": full_variable_count,
        "reduced_binary_variables": reduced_variable_count,
        "binary_variable_reduction_pct": float(
            100
            * (
                1
                - reduced_variable_count
                / full_variable_count
            )
        ),
        "minimum_candidates_per_order": int(
            candidate_counts.min()
        ),
        "maximum_candidates_per_order": int(
            candidate_counts.max()
        ),
        "mean_candidates_per_order": float(
            candidate_counts.mean()
        ),
        "anchor_coverage": anchor_coverage,
    }

    with open(
        REDUCED_CANDIDATE_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        "\nReduced candidate graph created:"
        f"\n  Orders retained: {len(orders):,}"
        f"\n  Drivers retained: {len(drivers):,}"
        f"\n  Candidate arcs: {len(reduced_costs):,}"
        f"\n  Binary variables: {reduced_variable_count:,}"
        f"\n  Variable reduction: "
        f"{manifest['binary_variable_reduction_pct']:.2f}%"
    )

    return reduced_costs, manifest