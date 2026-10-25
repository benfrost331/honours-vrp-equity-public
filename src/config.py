from pathlib import Path

# PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

# CORE PROBLEM PARAMETERS

PLANNING_DAYS = 16
ORDER_SERVICE_START = 540   # 09:00 
ORDER_SERVICE_END = 1020    # 17:00 
DRIVER_SHIFT_START = 480    # 08:00 
DRIVER_SHIFT_END = 1080     # 18:00 
DRIVER_SHIFT_LENGTH = DRIVER_SHIFT_END - DRIVER_SHIFT_START
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
FRIDAY_DAY_NAME = "Fri"

# LAMBDA EXPERIMENT PATHS

LAMBDA_RESULTS_DIR = OUTPUT_DIR / "lambda_experiments"
LAMBDA_ASSIGNMENTS_DIR = LAMBDA_RESULTS_DIR / "assignments"
LAMBDA_DRIVER_DIR = LAMBDA_RESULTS_DIR / "driver_workloads"
LAMBDA_DAY_DIR = LAMBDA_RESULTS_DIR / "day_workloads"
LAMBDA_LOG_DIR = LAMBDA_RESULTS_DIR / "logs"
LAMBDA_SUMMARY_PATH = (
    LAMBDA_RESULTS_DIR / "lambda_solution_summary.csv"
)
LAMBDA_MANIFEST_PATH = (
    LAMBDA_RESULTS_DIR / "experiment_manifest.json"
)

# STAGE 3 ROUTING OUTPUTS

STAGE3_RESULTS_DIR = (
    OUTPUT_DIR / "stage_3_routing"
)
STAGE3_REPRESENTATIVE_SOLUTIONS = [
    ("full_instance", "lambda_0_000"),
    ("full_instance", "lambda_0_100"),
    ("full_instance", "lambda_0_300"),
    ("reduced_candidate_graph", "lambda_0_600"),
]
STAGE3_EXACT_ROUTE_LIMIT = 8

# STAGE 3 REPAIR OUTPUTS

STAGE3_REPAIRED_RESULTS_DIR = (
    OUTPUT_DIR / "stage_3_routing_repaired"
)
STAGE3_REPAIR_MAX_ITERATIONS = 20
STAGE3_REPAIR_CANDIDATE_DRIVERS = 8
STAGE3_REPAIR_SEARCH_EXACT_ROUTE_LIMIT = 5
STAGE3_REPAIR_FINAL_EXACT_ROUTE_LIMIT = 8
STAGE3_REPAIR_MAX_TARGETS_PER_ORDER = 12
STAGE3_ROUTE_EVAL_CACHE_MAX_SIZE = 50_000

# REDUCED CANDIDATE-GRAPH PARAMETERS

REDUCED_CANDIDATE_K = 20
DATA_REDUCED_DIR = (
    BASE_DIR / "data" / "reduced_candidate_graph"
)
REDUCED_CANDIDATE_COST_PATH = (
    DATA_REDUCED_DIR
    / f"assignment_cost_matrix_k{REDUCED_CANDIDATE_K}.csv"
)
REDUCED_CANDIDATE_MAP_PATH = (
    DATA_REDUCED_DIR
    / f"candidate_driver_map_k{REDUCED_CANDIDATE_K}.csv"
)
REDUCED_CANDIDATE_MANIFEST_PATH = (
    DATA_REDUCED_DIR
    / f"candidate_graph_manifest_k{REDUCED_CANDIDATE_K}.json"
)
REDUCED_LAMBDA_RESULTS_DIR = (
    OUTPUT_DIR
    / f"lambda_experiments_reduced_k{REDUCED_CANDIDATE_K}"
)

# STAGE 4 ALNS ROUTING OUTPUTS AND PARAMETERS

STAGE4_RESULTS_DIR = (
    OUTPUT_DIR / "stage_4_alns"
)
STAGE4_REPRESENTATIVE_SOLUTIONS = [
    ("full_instance", "lambda_0_000"),
    ("full_instance", "lambda_0_300"),
    ("reduced_candidate_graph", "lambda_0_600"),
]
STAGE4_RANDOM_SEED = 42
STAGE4_MAX_ITERATIONS = 100
STAGE4_REMOVAL_FRACTION = 0.005
STAGE4_MIN_REMOVALS = 2
STAGE4_MAX_REMOVALS = 6
STAGE4_CANDIDATE_DRIVERS = 6
STAGE4_MAX_TARGETS_PER_ORDER = 12
STAGE4_SEARCH_EXACT_ROUTE_LIMIT = 5
STAGE4_FINAL_EXACT_ROUTE_LIMIT = 8
STAGE4_FINAL_POLISH_ONLY_CHANGED_ROUTES = True
STAGE4_ROUTE_EVAL_CACHE_MAX_SIZE = 50_000
STAGE4_INITIAL_TEMPERATURE = 0.035
STAGE4_COOLING_RATE = 0.995
STAGE4_VIOLATION_PENALTY = 1_000.0
STAGE4_INFEASIBLE_ROUTE_PENALTY = 100_000.0
STAGE4_CV_WEIGHT = 0.05
STAGE4_MAX_WORKLOAD_DETERIORATION_PCT = 0.10
STAGE4_MAX_WORKLOAD_DETERIORATION_PENALTY = 50.0
STAGE4_DESTROY_OPERATORS = [
    "random_removal",
    "worst_route_removal",
    "high_workload_driver_removal",
    "related_order_removal",
]
STAGE4_REPAIR_OPERATORS = [
    "cheapest_insertion",
    "regret_2_insertion",
    "workload_balancing_insertion",
]

# EQUITY-MEASURE EXPERIMENTS

EQUITY_RESULTS_DIR = OUTPUT_DIR / "equity_experiments"
EQUITY_MODEL_SCOPE = "full_instance"
EQUITY_MEASURE_TYPES = [
    "maximum",
    "minmax_deviation",
    "mad_mean",
    "range",
    "gini",
]
EQUITY_MEASURE_LABELS = {
    "maximum": "Maximum Workload",
    "minmax_deviation": "Min-Max Deviation",
    "mad_mean": "Mean Absolute Deviation",
    "range": "Range",
    "gini": "Normalised Gini Workload Dispersion",
}
EQUITY_LAMBDA_VALUES = [
    round(index / 10, 1)
    for index in range(11)
]
DEFAULT_EQUITY_RELATIVE_GAP = 0.03
DEFAULT_EQUITY_TIME_LIMIT_SECONDS = 7200
DEFAULT_EQUITY_SOLVER_NAME = "gurobi"
DEFAULT_EQUITY_SOLVER_THREADS = 4
