import pandas as pd



# BASIC DATASET STRUCTURE CHECKS


def check_missing_values(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Return a missing value summary for a dataframe.
    """
    missing_summary = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )

    missing_summary["dataset"] = name
    missing_summary["missing_pct"] = missing_summary["missing_count"] / len(df)

    return missing_summary[["dataset", "column", "missing_count", "missing_pct"]]


def check_duplicate_rows(df: pd.DataFrame, name: str) -> dict:
    """
    Return duplicate row count for a dataframe.
    """
    return {
        "dataset": name,
        "n_rows": len(df),
        "n_duplicate_rows": df.duplicated().sum()
    }


def check_routes_symmetry(routes: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether the routes matrix is symmetric.

    This is important because the proposal EDA found that the matrix
    is asymmetric, meaning that travel from i to j cannot be assumed
    equal to travel from j to i.
    """

    routes_reversed = (
        routes.rename(
            columns={
                "location_fromId": "location_toId",
                "location_toId": "location_fromId",
                "travel_time": "travel_time_rev",
                "travel_distance": "travel_distance_rev"
            }
        )
    )

    symmetry_check = routes.merge(
        routes_reversed,
        on=["location_fromId", "location_toId"],
        how="inner"
    )

    symmetry_check["time_diff"] = (
        symmetry_check["travel_time"] - symmetry_check["travel_time_rev"]
    )

    symmetry_check["distance_diff"] = (
        symmetry_check["travel_distance"] - symmetry_check["travel_distance_rev"]
    )

    summary = pd.DataFrame({
        "n_pairs": [len(symmetry_check)],
        "all_times_symmetric": [(symmetry_check["time_diff"] == 0).all()],
        "all_distances_symmetric": [(symmetry_check["distance_diff"] == 0).all()],
        "max_abs_time_diff": [symmetry_check["time_diff"].abs().max()],
        "max_abs_distance_diff": [symmetry_check["distance_diff"].abs().max()]
    })

    return summary