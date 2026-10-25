"""Derive fleet-wide shift-capacity utilisation across all 55 VRP configurations.

Run from the root of the honours-vrp-equity repository, or supply --repo-root.
Uses the canonical Stage 2 summary and the Stage 4 summary, which retains
its corresponding repaired Stage 3 baseline duration for each configuration.
No third-party Python dependencies are required.
"""

import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/fleet_capacity_utilisation_55.csv"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    s2 = read_csv(root / "outputs/equity_experiments/full_instance/equity_master_summary.csv")
    s4 = read_csv(root / "outputs/stage_4_alns/stage_4_master_summary.csv")
    key = lambda row: (row["equity_measure_type"], row["lambda_tag"])
    by_stage4 = {key(row): row for row in s4}
    if len(s2) != 55 or len(s4) != 55 or len(by_stage4) != 55:
        raise ValueError("Expected 55 unique configurations in both canonical summaries.")

    drivers, days, shift_minutes = 86, 16, 600
    available_minutes = drivers * days * shift_minutes
    fixed_service_minutes = 124714.0
    names = {
        "maximum": "Maximum Workload",
        "minmax_deviation": "Min-Max Deviation",
        "mad_mean": "Mean Absolute Deviation",
        "range": "Workload Range",
        "gini": "Normalised Gini",
    }
    results = []
    for r2 in s2:
        r4 = by_stage4.get(key(r2))
        if r4 is None:
            raise ValueError(f"No Stage 4 match for {key(r2)!r}")
        total_2 = float(r2["total_estimated_assignment_time"])
        total_3 = float(r4["baseline_total_route_duration"])
        total_4 = float(r4["total_route_duration"])
        if not (total_4 <= total_3 + 0.00001):
            raise ValueError(f"Stage 4 is worse than its retained Stage 3 baseline: {key(r2)!r}")
        if not (r2["solution_valid"].lower() == "true"):
            raise ValueError(f"Stored Stage 2 assignment not independently validated: {key(r2)!r}")
        results.append({
            "equity_measure": names[r2["equity_measure_type"]],
            "lambda": float(r2["lambda_equity"]),
            "stage2_solver_status": r2["status"],
            "stage2_independently_valid": r2["solution_valid"],
            "stage2_active_drivers": int(r2["n_drivers_used"]),
            "stage4_active_drivers": int(r4["n_active_drivers"]),
            "stage2_estimated_minutes": round(total_2, 3),
            "stage3_repaired_routed_minutes": round(total_3, 3),
            "stage4_final_routed_minutes": round(total_4, 3),
            "stage2_estimated_capacity_utilisation_pct": round(100 * total_2 / available_minutes, 4),
            "stage3_routed_capacity_utilisation_pct": round(100 * total_3 / available_minutes, 4),
            "stage4_routed_capacity_utilisation_pct": round(100 * total_4 / available_minutes, 4),
            "service_only_capacity_share_pct": round(100 * fixed_service_minutes / available_minutes, 4),
            "available_capacity_minutes": available_minutes,
        })
    results.sort(key=lambda r: (r["lambda"], r["equity_measure"]))
    out = args.output if args.output.is_absolute() else root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} configuration comparisons to {out}")
    print(f"Common shift-capacity denominator: {available_minutes:,} minutes ({available_minutes / 60:,.0f} hours)")


if __name__ == "__main__":
    main()
