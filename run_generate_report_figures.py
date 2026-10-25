from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator, FuncFormatter

def thousands_k_formatter(x, pos):
    return f"{x/1000:.0f}k"

def annotate_lambda_points(ax, df, xcol, ycol, label_specs):
    for _, row in df.iterrows():
        lam = round(float(row["lambda_equity"]), 1)

        if lam not in label_specs:
            continue

        spec = label_specs[lam]
        dx, dy = spec["xytext"]

        ax.annotate(
            rf"$\lambda={lam:.1f}$",
            (row[xcol], row[ycol]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=spec.get("ha", "left"),
            va=spec.get("va", "bottom"),
            fontsize=7.5,
            color="#555555",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=0.15,
            ),
        )



# PATHS


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

STAGE2_BASELINE_WORKLOAD_PATH = (
    OUTPUT_DIR
    / "equity_experiments"
    / "full_instance"
    / "maximum"
    / "driver_workloads"
    / "lambda_0_000.csv"
)

STAGE2_MASTER_SUMMARY_PATH = (
    OUTPUT_DIR
    / "equity_experiments"
    / "full_instance"
    / "equity_master_summary.csv"
)



# REPORT FIGURE STYLE


BAR_BLUE = "#377EB8"
EDGE_COLOUR = "#222222"
GRID_COLOUR = "#D9D9D9"

MEASURE_COLOURS = {
    "maximum": "#377EB8",
    "minmax_deviation": "#E69F00",
    "mad_mean": "#4DAF4A",
    "range": "#D55E5E",
    "gini": "#8E6BBE",
}

MEASURE_LABELS = {
    "maximum": "Maximum Workload",
    "minmax_deviation": "Min-Max Deviation",
    "mad_mean": "Mean Absolute Deviation",
    "range": "Range",
    "gini": "Gini",
}

EQUITY_METRIC_COLUMNS = {
    "maximum": "max_driver_workload",
    "minmax_deviation": "workload_minmax_deviation",
    "mad_mean": "workload_mad_mean",
    "range": "workload_range",
    "gini": "workload_gini_coefficient",
}

FIGURE_WIDTH = 7.0
FIGURE_HEIGHT = 4.6


def apply_report_style(ax):
    """
    Apply a common visual style to report figures.
    Designed to resemble the existing EDA figures in Section 3.
    """

    ax.set_facecolor("white")

    ax.grid(
        True,
        which="major",
        axis="both",
        color=GRID_COLOUR,
        linewidth=0.8,
        alpha=0.7,
    )

    ax.set_axisbelow(True)

    # Keep the figure visually light, similar to the existing EDA plots.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#BFBFBF")
    ax.spines["bottom"].set_color("#BFBFBF")

    ax.tick_params(
        axis="both",
        labelsize=10,
        color="#BFBFBF",
    )


# HELPERS:


def thousands_k_formatter(x, pos):
    return f"{x/1000:.0f}k"


# FIGURE: Section 6.1.1


def plot_stage2_baseline_workloads():
    df = pd.read_csv(STAGE2_BASELINE_WORKLOAD_PATH)

    # Highest workload first so concentration is immediately visible.
    df = (
        df.sort_values(
            "total_estimated_assignment_time",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    df["driver_rank"] = range(1, len(df) + 1)

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
    )

    active_mean = (
        df.loc[
            df["is_active"],
            "total_estimated_assignment_time"
        ]
        .mean()
    )

    ax.axhline(
        active_mean,
        linestyle="--",
        linewidth=1.0,
        color="#666666",
    )

    ax.text(
        85,
        active_mean + 120,
        f"Active-Driver Mean = {active_mean:.0f} min",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    ax.bar(
        df["driver_rank"],
        df["total_estimated_assignment_time"],
        width=0.82,
        color=BAR_BLUE,
        edgecolor=EDGE_COLOUR,
        linewidth=0.6,
    )

    apply_report_style(ax)

    ax.set_xlabel(
        "Driver Rank (Highest to Lowest Workload)",
        fontsize=11,
    )

    ax.set_ylabel(
        "Estimated Assignment Workload (minutes)",
        fontsize=11,
    )

    # Keep the entire 86-driver distribution visible.
    ax.set_xlim(0.3, len(df) + 0.7)

    # Horizon capacity = 16 days x 600 minutes.
    ax.set_ylim(0, 9800)

    ax.set_xticks(
        [1, 10, 20, 30, 40, 50, 60, 70, 80, 86]
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # PNG is useful for quick viewing.
    fig.savefig(
        FIGURE_DIR / "stage2_baseline_driver_workloads.png",
        dpi=300,
        bbox_inches="tight",
    )

    # PDF is preferable for the final LaTeX report because it remains vector.
    fig.savefig(
        FIGURE_DIR / "stage2_baseline_driver_workloads.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        FIGURE_DIR / "stage2_baseline_driver_workloads.png"
    )
    print(
        "Saved:",
        FIGURE_DIR / "stage2_baseline_driver_workloads.pdf"
    )


# FIGURE: Section 6.1.2


def plot_stage2_efficiency_equity_tradeoff_refined():
    df = pd.read_csv(STAGE2_MASTER_SUMMARY_PATH)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.6, 4.8),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={
            "width_ratios": [4.8, 1.4],
            "wspace": 0.05,
        },
    )

    for ax in (ax_left, ax_right):
        apply_report_style(ax)

    for measure, metric_column in EQUITY_METRIC_COLUMNS.items():
        subset = (
            df.loc[df["equity_measure_type"] == measure]
            .sort_values("lambda_equity")
            .copy()
        )

        baseline_cost = subset.iloc[0]["total_estimated_assignment_time"]
        baseline_equity = subset.iloc[0][metric_column]

        subset["efficiency_cost_increase_pct_plot"] = (
            100
            * (
                subset["total_estimated_assignment_time"]
                - baseline_cost
            )
            / baseline_cost
        )

        subset["equity_remaining_pct"] = (
            100
            * subset[metric_column]
            / baseline_equity
        )

        subset["proven_flag"] = (
            subset["proven_optimal"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        colour = MEASURE_COLOURS[measure]
        label = MEASURE_LABELS[measure]

        
        # Main decision region: 0--10.5% increase in assignment time
        

        main_region = subset.loc[
            subset["efficiency_cost_increase_pct_plot"] <= 10.5
        ].copy()

        ax_left.plot(
            main_region["efficiency_cost_increase_pct_plot"],
            main_region["equity_remaining_pct"],
            color=colour,
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            label=label,
            zorder=2,
        )


        
        # Extreme region: show isolated points only
        

        extreme_region = subset.loc[
            subset["efficiency_cost_increase_pct_plot"] >= 44.0
        ].copy()

        if not extreme_region.empty:
            ax_right.scatter(
                extreme_region["efficiency_cost_increase_pct_plot"],
                extreme_region["equity_remaining_pct"],
                s=30,
                facecolors=colour,
                edgecolors=colour,
                linewidths=1.0,
                zorder=2,
            )

            
            # Hollow markers identify feasible incumbents without the
            # requested optimisation certificate
            

            uncertified_main = main_region.loc[
                ~main_region["proven_flag"]
            ]

            if not uncertified_main.empty:
                ax_left.scatter(
                    uncertified_main["efficiency_cost_increase_pct_plot"],
                    uncertified_main["equity_remaining_pct"],
                    s=38,
                    facecolors="white",
                    edgecolors=colour,
                    linewidths=1.3,
                    zorder=4,
                )


            uncertified_extreme = extreme_region.loc[
                ~extreme_region["proven_flag"]
            ]

            if not uncertified_extreme.empty:
                ax_right.scatter(
                    uncertified_extreme["efficiency_cost_increase_pct_plot"],
                    uncertified_extreme["equity_remaining_pct"],
                    s=38,
                    facecolors="white",
                    edgecolors=colour,
                    linewidths=1.3,
                    zorder=4,
                )

    # Label λ = 0 once, since all measures share the same origin.
    if measure == "gini":
        ax_left.annotate(
            r"$\lambda=0$",
            xy=(0, 100),
            xytext=(5, -11),
            textcoords="offset points",
            fontsize=8,
            color="#666666",
        )


    # Label Maximum Workload λ = 1 in the main region.
    if measure == "maximum":
        last_row = subset.iloc[-1]

        ax_left.annotate(
            r"$\lambda=1$",
            xy=(
                last_row["efficiency_cost_increase_pct_plot"],
                last_row["equity_remaining_pct"],
            ),
            xytext=(5, -1),
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )


    # Label the two extreme λ = 1 endpoints.
    if measure in {"range", "minmax_deviation"}:
        last_row = subset.iloc[-1]

        ax_right.annotate(
            r"$\lambda=1$",
            xy=(
                last_row["efficiency_cost_increase_pct_plot"],
                last_row["equity_remaining_pct"],
            ),
            xytext=(4, -11),
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )    

    # Axis limits: focus on the practical region while still retaining the extreme points
    ax_left.set_xlim(-0.3, 10.5)
    ax_right.set_xlim(44.0, 58.5)

    ax_left.set_ylim(-2, 105)

    ax_left.set_ylabel(
        "Equity Measure Remaining (% of Baseline)",
        fontsize=11,
    )

    fig.supxlabel(
        "Increase in Estimated Assignment Time (%)",
        fontsize=11,
    )

    # Clean up the shared axis appearance
    ax_right.spines["left"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    ax_right.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    # Broken-axis markers
    d = 0.012  # size of diagonal break marks in axes coordinates
    kwargs = dict(
        transform=ax_left.transAxes,
        color="#666666",
        clip_on=False,
        linewidth=1.0,
    )
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Legend
    handles, labels = ax_left.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_right.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        fontsize=8.5,
        loc="right",
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR / "stage2_efficiency_equity_tradeoff_refined.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR / "stage2_efficiency_equity_tradeoff_refined.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# FIGURE: Section 6.3.1


def plot_stage2_stage3_maximum_tradeoff():
    stage2 = pd.read_csv(STAGE2_MASTER_SUMMARY_PATH)

    stage2 = (
        stage2.loc[
            stage2["equity_measure_type"] == "maximum"
        ]
        .sort_values("lambda_equity")
        .copy()
    )

    stage3_rows = []

    for _, row in stage2.iterrows():
        lambda_tag = row["lambda_tag"]

        path = (
            OUTPUT_DIR
            / "stage_3_routing_repaired"
            / "full_instance"
            / "maximum"
            / lambda_tag
            / "stage_3_solution_summary.csv"
        )

        summary = pd.read_csv(path)

        stage3_rows.append(summary.iloc[0])

    stage3 = pd.DataFrame(stage3_rows)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(9.2, 4.4),
        constrained_layout=True,
    )

    apply_report_style(ax1)
    apply_report_style(ax2)

    ax1.plot(
        stage2["total_estimated_assignment_time"],
        stage2["max_driver_workload"],
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color=BAR_BLUE,
    )

    ax1.set_xlabel(
        "Estimated Assignment Time (minutes)",
        fontsize=10,
    )

    ax1.set_ylabel(
        "Maximum Driver Workload (minutes)",
        fontsize=10,
    )

    ax1.set_title(
        "Stage 2: Assignment",
        fontsize=11,
    )

    ax2.plot(
        stage3["total_route_duration"],
        stage3["max_driver_route_workload"],
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color=BAR_BLUE,
    )

    ax2.set_xlabel(
        "Total Routed Duration (minutes)",
        fontsize=10,
    )

    ax2.set_ylabel(
        "Maximum Routed Workload (minutes)",
        fontsize=10,
    )

    ax2.set_title(
        "Stage 3: Routed",
        fontsize=11,
    )

    # Reduce x-axis tick crowding and shorten labels
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1.xaxis.set_major_formatter(FuncFormatter(thousands_k_formatter))
    ax2.xaxis.set_major_formatter(FuncFormatter(thousands_k_formatter))

    ax1.tick_params(axis="x", labelsize=9)
    ax2.tick_params(axis="x", labelsize=9)

    # Give labels at the ends a little room
    ax1.margins(x=0.06, y=0.08)
    ax2.margins(x=0.06, y=0.08)

    stage2_label_specs = {
        0.0: {"xytext": (6, 6),  "ha": "left", "va": "bottom"},
        0.1: {"xytext": (6, 6),  "ha": "left", "va": "bottom"},
        0.4: {"xytext": (6, 6),  "ha": "left", "va": "bottom"},
        0.7: {"xytext": (8, 6),  "ha": "left", "va": "bottom"},
        1.0: {"xytext": (8, -10), "ha": "left", "va": "top"},
    }

    stage3_label_specs = {
        0.0: {"xytext": (6, 6),   "ha": "left", "va": "bottom"},
        0.1: {"xytext": (6, 6),   "ha": "left", "va": "bottom"},
        0.4: {"xytext": (6, 6),   "ha": "left", "va": "bottom"},
        0.7: {"xytext": (8, 6),   "ha": "left", "va": "bottom"},
        1.0: {"xytext": (8, -10), "ha": "left", "va": "top"},
    }

    annotate_lambda_points(
        ax1,
        stage2,
        "total_estimated_assignment_time",
        "max_driver_workload",
        stage2_label_specs,
    )

    annotate_lambda_points(
        ax2,
        stage3,
        "total_route_duration",
        "max_driver_route_workload",
        stage3_label_specs,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR
        / "stage2_stage3_maximum_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR
        / "stage2_stage3_maximum_tradeoff.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# FIGURE: Section 6.3.2


def plot_stage3_proxy_vs_routed_ratio():
    stage2 = pd.read_csv(STAGE2_MASTER_SUMMARY_PATH)

    measure_order = [
        "maximum",
        "minmax_deviation",
        "mad_mean",
        "range",
        "gini",
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5),
        constrained_layout=True,
    )

    apply_report_style(ax)

    for measure in measure_order:
        subset = (
            stage2.loc[
                stage2["equity_measure_type"] == measure
            ]
            .sort_values("lambda_equity")
            .copy()
        )

        routed_percentages = []

        for _, row in subset.iterrows():
            lambda_tag = row["lambda_tag"]

            stage3_path = (
                OUTPUT_DIR
                / "stage_3_routing_repaired"
                / "full_instance"
                / measure
                / lambda_tag
                / "stage_3_solution_summary.csv"
            )

            stage3 = pd.read_csv(stage3_path).iloc[0]

            routed_percentages.append(
                100
                * float(stage3["total_route_duration"])
                / float(row["total_estimated_assignment_time"])
            )

        ax.plot(
            subset["lambda_equity"],
            routed_percentages,
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=MEASURE_COLOURS[measure],
            label=MEASURE_LABELS[measure],
        )

    ax.set_xlabel(
        r"Equity Weight $\lambda$",
        fontsize=11,
    )

    ax.set_ylabel(
        "Routed Duration (% of Assignment Proxy)",
        fontsize=11,
    )

    ax.set_xticks(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    ax.set_ylim(88, 97)

    ax.axhline(
        100,
        linewidth=0.8,
        linestyle="--",
        color="#777777",
    )

    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR / "stage3_proxy_vs_routed_ratio.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR / "stage3_proxy_vs_routed_ratio.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# FIGURE: Section 6.4.1


def plot_stage4_route_duration_improvement():

    summary_path = (
        OUTPUT_DIR
        / "stage_4_alns"
        / "stage_4_master_summary.csv"
    )

    results = pd.read_csv(summary_path)

    results["route_duration_reduction_pct"] = (
        100
        * (
            results["baseline_total_route_duration"]
            - results["total_route_duration"]
        )
        / results["baseline_total_route_duration"]
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5),
        constrained_layout=True,
    )

    apply_report_style(ax)

    measure_order = [
        "maximum",
        "minmax_deviation",
        "mad_mean",
        "range",
        "gini",
    ]

    for measure in measure_order:

        subset = (
            results.loc[
                results["equity_measure_type"] == measure
            ]
            .sort_values("lambda_equity")
            .copy()
        )

        ax.plot(
            subset["lambda_equity"],
            subset["route_duration_reduction_pct"],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=MEASURE_COLOURS[measure],
            label=MEASURE_LABELS[measure],
        )

    ax.axhline(
        0,
        linewidth=0.9,
        color="#666666",
    )

    ax.set_xlabel(
        r"Equity Weight $\lambda$",
        fontsize=11,
    )

    ax.set_ylabel(
        "Reduction in Routed Duration (%)",
        fontsize=11,
    )

    ax.set_xticks(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    ax.set_ylim(
        -0.15,
        4.2,
    )

    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper right",
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR
        / "stage4_route_duration_improvement.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR
        / "stage4_route_duration_improvement.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# FIGURE: Section 6.4.3


def plot_stage4_assignment_changes():

    summary_path = (
        OUTPUT_DIR
        / "stage_4_alns"
        / "stage_4_master_summary.csv"
    )

    results = pd.read_csv(summary_path)

    n_orders = 1367

    results["changed_assignments_pct"] = (
        100
        * results["n_changed_assignments"]
        / n_orders
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5),
        constrained_layout=True,
    )

    apply_report_style(ax)

    measure_order = [
        "maximum",
        "minmax_deviation",
        "mad_mean",
        "range",
        "gini",
    ]

    for measure in measure_order:

        subset = (
            results.loc[
                results["equity_measure_type"] == measure
            ]
            .sort_values("lambda_equity")
            .copy()
        )

        ax.plot(
            subset["lambda_equity"],
            subset["changed_assignments_pct"],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=MEASURE_COLOURS[measure],
            label=MEASURE_LABELS[measure],
        )

    ax.set_xlabel(
        r"Equity Weight $\lambda$",
        fontsize=11,
    )

    ax.set_ylabel(
        "Orders Reassigned from Stage 3 (%)",
        fontsize=11,
    )

    ax.set_xticks(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    ax.set_ylim(
        -0.5,
        30,
    )

    ax.axhline(
        0,
        linewidth=0.8,
        color="#666666",
    )

    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper right",
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR / "stage4_assignment_changes.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR / "stage4_assignment_changes.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)



# FIGURE: Section 6.2 


def plot_stage2_equity_workload_profiles():

    workload_dir = (
        OUTPUT_DIR
        / "equity_experiments"
        / "full_instance"
    )

    measure_order = [
        "maximum",
        "minmax_deviation",
        "mad_mean",
        "range",
        "gini",
    ]

    
    # Load and rank driver workloads
    

    def load_ranked_workloads(measure, lambda_tag):

        path = (
            workload_dir
            / measure
            / "driver_workloads"
            / f"{lambda_tag}.csv"
        )

        df = pd.read_csv(path)

        if len(df) != 86:
            raise ValueError(
                f"Expected 86 drivers in {path}, found {len(df)}"
            )

        return (
            df["total_estimated_assignment_time"]
            .sort_values(ascending=False)
            .to_numpy(dtype=float)
        )

    # Common efficiency-oriented baseline.
    baseline = load_ranked_workloads(
        "maximum",
        "lambda_0_000",
    )

    # Five certified equity-adjusted configurations.
    adjusted = {
        measure: load_ranked_workloads(
            measure,
            "lambda_0_700",
        )
        for measure in measure_order
    }

    ranks = range(1, 87)

    
    # Figure layout
    

    fig, (ax_full, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.05, 1.0],
        },
    )

    for ax in (ax_full, ax_zoom):

        apply_report_style(ax)

        ax.set_xlim(0.5, 86.5)

        ax.set_ylabel(
            "Estimated Driver Workload (mins)",
            fontsize=11,
        )

    
    # Panel A: Baseline versus five equity-adjusted solutions
    

    ax_full.fill_between(
        ranks,
        baseline,
        color="#303841",
        alpha=0.08,
    )

    ax_full.plot(
        ranks,
        baseline,
        color="#303841",
        linewidth=2.2,
        label=r"Efficiency Baseline ($\lambda=0$)",
        zorder=4,
    )

    for measure in measure_order:

        colour = MEASURE_COLOURS[measure]

        ax_full.plot(
            ranks,
            adjusted[measure],
            color=colour,
            linewidth=1.5,
            label=MEASURE_LABELS[measure],
            zorder=3,
        )

        # Add the same measure to the expanded lower panel.
        ax_zoom.plot(
            ranks,
            adjusted[measure],
            color=colour,
            linewidth=1.9,
        )

    ax_full.set_title(
        r"Full Workload Range: Baseline vs. $\lambda=0.7$",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )

    ax_full.set_ylim(-150, 9600)

    ax_full.set_yticks(
        [0, 2000, 4000, 6000, 8000]
    )

    ax_full.tick_params(
        axis="x",
        labelbottom=False,
    )

    # Highlight the maximum baseline driver workload.
    ax_full.annotate(
        f"{baseline[0]:,.0f} min",
        xy=(1, baseline[0]),
        xytext=(14, 8650),
        fontsize=9,
        color="#303841",
        arrowprops=dict(
            arrowstyle="-",
            color="#666666",
            lw=0.9,
        ),
    )

    # Highlight the unused drivers in the baseline.
    n_unused = int((baseline <= 1e-8).sum())

    ax_full.annotate(
        f"{n_unused} Drivers\nUnused",
        xy=(81, 0),
        xytext=(72, 3700),
        fontsize=9,
        ha="center",
        color="#303841",
        arrowprops=dict(
            arrowstyle="->",
            color="#666666",
            lw=0.9,
        ),
    )

    
    # Panel B: Expanded Equity-Adjusted Workload Distributions
    

    ax_zoom.set_title(
        "Expanded View: Five Equity-Adjusted Distributions",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )

    ax_zoom.set_ylim(1080, 2400)

    ax_zoom.set_yticks(
        [1200, 1400, 1600, 1800, 2000, 2200, 2400]
    )

    ax_zoom.set_xticks(
        [1, 10, 20, 30, 40, 50, 60, 70, 80, 86]
    )

    ax_zoom.set_xlabel(
        "Driver Rank (Highest to Lowest within Each Solution)",
        fontsize=11,
    )

    
    # Shared Legend
    

    handles, labels = ax_full.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.015),
        frameon=False,
        fontsize=9,
        handlelength=2.5,
        columnspacing=1.4,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.94,
        bottom=0.19,
        hspace=0.27,
    )

    
    # Export Report-Quality Figures
    

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in ("png", "pdf"):

        path = (
            FIGURE_DIR
            / f"stage2_workload_comparison_lambda07.{extension}"
        )

        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )

        print("Saved:", path)

    plt.close(fig)




# FIGURE: Section 6.4.4 - ALNS Search Trace (Gini, lambda = 0.1)


def plot_stage4_alns_search_trace_gini_lambda01():

    run_dir = (
        OUTPUT_DIR
        / "stage_4_alns"
        / "full_instance"
        / "gini"
        / "lambda_0_100"
    )

    trace = pd.read_csv(
        run_dir / "alns_iteration_log.csv"
    ).sort_values("iteration").reset_index(drop=True)

    summary = pd.read_csv(
        run_dir / "stage_4_solution_summary.csv"
    ).iloc[0]

    
    # Extract Recorded Search Results
    

    iterations = trace["iteration"].to_numpy(dtype=int)

    best_duration = trace[
        "best_total_route_duration"
    ].to_numpy(dtype=float)

    current_score = trace[
        "current_scalar_score"
    ].to_numpy(dtype=float)

    best_score = trace[
        "best_scalar_score"
    ].to_numpy(dtype=float)

    baseline_duration = float(
        summary["baseline_total_route_duration"]
    )

    polished_duration = float(
        summary["total_route_duration"]
    )

    # Accepted candidates that did not improve the current solution.
    accepted = (
        trace["accepted"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    improved_current = (
        trace["improved_current"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    non_improving = (
        accepted & ~improved_current
    ).to_numpy(dtype=bool)

    # Confirm that the canonical 100-iteration trace is complete.
    if iterations.tolist() != list(range(1, 101)):
        raise ValueError(
            "Expected a complete 100-iteration ALNS trace."
        )

    # Include the Stage 3 starting solution at iteration zero.
    plot_iterations = [0] + iterations.tolist()

    plot_duration = (
        [baseline_duration] + best_duration.tolist()
    )

    colour_best = MEASURE_COLOURS["gini"]
    colour_current = BAR_BLUE
    colour_baseline = "#777777"
    colour_markers = "#B45309"

    
    # Create the two-panel figure
    

    fig, (ax_duration, ax_score) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.7),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.05, 1.0],
        },
    )

    for ax in (ax_duration, ax_score):
        apply_report_style(ax)
        ax.set_xlim(0, 100)

    
    # PANEL A: Best retained routed duration
    

    ax_duration.axhline(
        baseline_duration,
        color=colour_baseline,
        linestyle="--",
        linewidth=1.2,
        label="Stage 3 Starting Duration",
        zorder=2,
    )

    ax_duration.step(
        plot_iterations,
        plot_duration,
        where="post",
        color=colour_best,
        linewidth=2.0,
        label="Best Retained Routed Duration",
        zorder=3,
    )

    # Highlight the final best solution within the ALNS search.
    ax_duration.scatter(
        iterations[-1],
        best_duration[-1],
        color=colour_best,
        s=30,
        zorder=5,
    )

    ax_duration.annotate(
        f"Iteration 100:\n{best_duration[-1]:,.0f} min",
        xy=(100, best_duration[-1]),
        xytext=(-10, 17),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=colour_best,
    )

    duration_span = (
        baseline_duration - best_duration[-1]
    )

    ax_duration.set_ylim(
        best_duration[-1] - 0.12 * duration_span,
        baseline_duration + 0.12 * duration_span,
    )

    ax_duration.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:,.0f}")
    )

    ax_duration.set_ylabel(
        "Total Routed Duration (minutes)",
        fontsize=11,
    )

    ax_duration.set_title(
        "Best Retained Routed Duration",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )

    ax_duration.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper right",
    )

    
    # PANEL B: Current vs. Best Scalar Evaluation Score
    

    ax_score.plot(
        iterations,
        current_score,
        color=colour_current,
        linewidth=1.5,
        label="Current Solution Score",
        zorder=2,
    )

    ax_score.plot(
        iterations,
        best_score,
        color=colour_best,
        linewidth=2.0,
        label="Best Retained Score",
        zorder=3,
    )

    # Highlight periods when the current solution is worse than the best retained solution.
    ax_score.fill_between(
        iterations,
        best_score,
        current_score,
        where=(current_score > best_score),
        color=colour_current,
        alpha=0.13,
        linewidth=0,
        zorder=1,
    )

    # Hollow markers indicate accepted candidates that did not improve the current solution.
    ax_score.scatter(
        iterations[non_improving],
        current_score[non_improving],
        s=24,
        facecolors="white",
        edgecolors=colour_markers,
        linewidths=1.1,
        label="Accepted Without Current Improvement",
        zorder=5,
    )

    score_min = min(
        current_score.min(),
        best_score.min(),
    )

    score_max = max(
        current_score.max(),
        best_score.max(),
    )

    score_span = score_max - score_min

    ax_score.set_ylim(
        score_min - 0.12 * score_span,
        score_max + 0.20 * score_span,
    )

    ax_score.set_title(
        "Current vs. Best Evaluation Score",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )

    ax_score.set_xlabel(
        "ALNS Iteration",
        fontsize=11,
    )

    ax_score.set_ylabel(
        "Scalar Evaluation Score",
        fontsize=11,
    )

    ax_score.set_xticks(
        [0, 20, 40, 60, 80, 100]
    )

    ax_score.legend(
        frameon=False,
        fontsize=8,
        loc="upper right",
    )

    
    # Layout & Export
    

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.94,
        bottom=0.10,
        hspace=0.28,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = "stage4_alns_search_trace_gini_lambda01"

    fig.savefig(
        FIGURE_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIGURE_DIR / f"{filename}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Saved:", FIGURE_DIR / f"{filename}.png")
    print("Saved:", FIGURE_DIR / f"{filename}.pdf")

    print(
        f"Final search duration: {best_duration[-1]:,.2f} min"
    )
    print(
        f"Final polished duration: {polished_duration:,.2f} min"
    )
    print(
        f"Additional polishing saving: "
        f"{best_duration[-1] - polished_duration:.2f} min"
    )



# RUN


if __name__ == "__main__":

    plot_stage2_baseline_workloads()
    plot_stage2_equity_workload_profiles()
    plot_stage2_efficiency_equity_tradeoff_refined()
    plot_stage2_stage3_maximum_tradeoff()
    plot_stage3_proxy_vs_routed_ratio()
    plot_stage4_route_duration_improvement()
    plot_stage4_assignment_changes()
    plot_stage4_alns_search_trace_gini_lambda01()

