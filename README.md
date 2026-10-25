# Decision Support Models for Multi-Period Vehicle Routing with Equity Considerations

**BSc Honours in Statistics & Data Science · University of Cape Town · 2026**  
**Authors:** Benjamin Frost & Jadene Jaftha  
**Supervisor:** Dr. Georgina Rakotonirainy

## Overview

This repository contains the Python implementation and canonical experimental results of a research project investigating the trade-off between operational efficiency and equitable driver workloads in a multi-period vehicle routing problem. The framework combines mixed-integer linear programming (MILP), explicit daily route construction, feasibility repair, and Adaptive Large Neighbourhood Search (ALNS).

Rather than identifying a single preferred schedule, the project compares allocation policies across **five workload-equity measures and eleven equity weights**. Three Streamlit applications support exploration of the resulting assignments, routes, and improvements.

| Instance Characteristic | Value |
|---|---:|
| Customer Orders | 1,367 |
| Drivers | 86 |
| Planning Horizon | 16 working days |
| Friday-Restricted Orders | 565 |
| Equity Measures | 5 |
| Equity Weights per Measure | 11 (0.0–1.0 in steps of 0.1) |
| Canonical Configurations | **55** |

Time, rather than vehicle carrying capacity, is the primary operational resource. Orders have service durations, admissible days and service windows; drivers have base locations and shift limits. Travel times are directed and may be asymmetric. The supplied spatial coordinates are for relative visualisation and **are not verified geographic latitude and longitude**.

## Explore the Results

| Dashboard | Explore | Run Locally |
|---|---|---|
| **Stage 2 — Assignment & Equity** | [Stage 2 dashboard](https://honours-vrp-equity-stage2-dashboard.streamlit.app/) | `streamlit run app.py` |
| **Stage 3 — Routing & Repair** | [Stage 3 dashboard](https://honours-vrp-equity-stage3-dashboard.streamlit.app/) | `streamlit run app_stage3.py` |
| **Stage 4 — ALNS Improvement** | [Stage 4 dashboard](https://honours-vrp-equity-stage4-dashboard.streamlit.app/) | `streamlit run app_stage4.py` |

The dashboards **visualise stored experiments**; selecting an equity measure or weight does not solve a new MILP.

## Research Approach

```text
Raw Orders, Drivers & Directed Travel Network
                     |
                     v
Stage 1   Data Validation, Preprocessing & Feasibility
                     |
                     v
Stage 2   MILP Order–Day–Driver Assignment
          5 Equity Measures × 11 Equity Weights
                     |
                     v
Stage 3   Driver-Day Route Construction, Validation & Repair
                     |
                     v
Stage 4   Feasibility-Aware ALNS Improvement
                     |
                     v
          Canonical Results & Decision-Support Dashboards
```

**Stage 1 — Data Foundation:** Validate the input datasets, clean orders, drivers and travel records, construct order-day feasibility, and prepare the inputs used to calculate driver–order assignment costs in Stage 2.

**Stage 2 — Assignment Optimisation:** Assign each order exactly once to an admissible day and driver, subject to estimated driver-day shift capacity. A normalised objective balances total estimated assignment time against one of five workload-equity measures, with equity emphasis controlled by $\lambda \in [0,1]$:

$$
\min_x\left[(1-\lambda)\frac{C(x)}{C_{\mathrm{ref}}}
+\lambda\frac{E(x)}{E_{\mathrm{ref}}}
+\varepsilon\frac{C(x)}{C_{\mathrm{ref}}}\right].
$$

The small efficiency tie-break term is included in the implementation. Stage 2 costs approximate travel using individual driver-base return journeys; they are **not** realised multi-stop route durations.

| Equity Measure | What it Emphasises | Code Identifier |
|---|---|---|
| Maximum Workload | The busiest driver's workload | `maximum` |
| Min-Max Deviation | Largest absolute deviation from mean workload | `minmax_deviation` |
| Mean Absolute Deviation | Average absolute deviation from mean workload | `mad_mean` |
| Workload Range | Difference between maximum and minimum workloads | `range` |
| Normalised Gini | Inequality across the driver workload distribution | `gini` |

**Stage 3 — Routing & Repair:** Convert each assignment solution into directed driver-day routes, respecting customer time windows, service times and driver shifts. Locally reassign orders where required to restore operational feasibility. Both initial and repaired route solutions are retained to support comparison.

**Stage 4 — ALNS:** Start from repaired routes and use adaptive order-removal and reinsertion operators to explore routing and assignment changes. Route-evaluation caching, feasibility-aware scoring and a simulated-annealing acceptance mechanism control the search. The canonical experiments use **100 iterations per configuration**; the runner derives configuration-specific random seeds from its starting seed.

The layered approach is computationally tractable for exploring alternative equity policies but does **not** certify global optimality of the integrated multi-period routing problem.

## Key Findings

The following results refer to the **canonical full-instance experiments** included in this repository.

| Finding | Result |
|---|---:|
| Stage 2 assignments passing independent validity checks | **55 / 55** |
| Stage 2 configurations meeting the configured solver stopping criterion | **49 / 55** |
| Stage 2 feasible incumbents without that certificate | **6 / 55** |
| Stage 3 configurations with at least one initially infeasible route | **42 / 55** |
| Recorded Stage 3 single-order repair moves | **83** |
| Repaired Stage 3 configurations with feasible routes | **55 / 55** |
| Stage 4 configurations with shorter routed duration | **39 / 55** |
| Mean Stage 4 routed-duration improvement | **0.90%** |
| Maximum Stage 4 routed-duration improvement | **3.80%** |
| Final Stage 4 configurations with feasible routes | **55 / 55** |

The efficiency-oriented baseline uses **76 of 86 drivers** and has an estimated assignment time of approximately **168,049 minutes**. Equity-aware assignments can substantially reduce workload concentration, although their efficiency cost depends on the equity measure and the selected weight. Explicit routing then reveals feasibility issues that cannot be established from assignment costs alone; Stage 3 repair restores feasibility, and Stage 4 finds further improvements for many configurations.

The five equity formulations do not produce interchangeable workload distributions. The appropriate measure depends on the operational meaning of fairness, including whether the decision maker prioritises the most heavily loaded driver, the overall workload range, or broader distributional inequality.

Solver stopping status and independently validated feasibility are separate concepts. A reported `Optimal` solver status under a relative-gap tolerance does not establish a zero-gap global optimum, and the six time-limited incumbents should not be described as certified optimal.

## Repository Structure

```text
honours-vrp-equity-public/
├── README.md
├── requirements.txt                    # Dashboard and common dependencies
├── requirements-optimisation.txt       # Additional solver packages
├── main.py                             # Stage 1 preprocessing and validation
├── run_full_equity_experiments.py      # Stage 2 equity-weight sweeps
├── run_build_equity_master_summary.py
├── run_stage_3_equity_routing.py
├── run_stage_3_equity_repair.py
├── run_stage_4_alns.py
├── run_build_stage_4_master_summary.py
├── compute_fleet_capacity_utilisation.py
├── run_generate_report_figures.py
├── app.py                              # Stage 2 dashboard
├── app_stage3.py                       # Stage 3 dashboard
├── app_stage4.py                       # Stage 4 dashboard
├── src/                                # Modelling, routing and data loaders
├── data/
│   ├── raw/                            # orders.csv, drivers.csv, routes.csv
│   ├── processed/                      # Full-instance inputs and cost matrix
│   └── reduced_candidate_graph/        # Supporting K=20 methodology artefacts
├── outputs/
│   ├── equity_experiments/full_instance/
│   ├── stage_3_routing/full_instance/
│   ├── stage_3_routing_repaired/full_instance/
│   ├── stage_4_alns/full_instance/
│   ├── analysis/                       # Fleet capacity comparison
│   ├── tables/                         # Stage 1 data validation summaries
│   └── figures/                        # Research figures (PNG and PDF)
└── .streamlit/config.toml
```

The complete final outputs span the five equity-measure directories and eleven values of `lambda_0_000` through `lambda_1_000`. Each Stage 2 configuration has assignment and workload files; each Stage 3 and Stage 4 configuration retains its detailed route files and solution summary. The supporting `data/reduced_candidate_graph/` files are **not** part of the canonical 55-configuration full-instance experiment.

### Canonical Results to Inspect

| Purpose | Location |
|---|---|
| Stage 2 Comparison Across All 55 Configurations | `outputs/equity_experiments/full_instance/equity_master_summary.csv` |
| Stage 2 Configuration-Specific Assignments | `outputs/equity_experiments/full_instance/<measure>/assignments/<lambda_tag>.csv` |
| Initial Stage 3 Routes | `outputs/stage_3_routing/full_instance/<measure>/<lambda_tag>/` |
| Repaired Stage 3 Routes | `outputs/stage_3_routing_repaired/full_instance/<measure>/<lambda_tag>/` |
| Stage 4 Routes, Iteration Traces & Operators | `outputs/stage_4_alns/full_instance/<measure>/<lambda_tag>/` |
| Stage 4 Comparison Across All 55 Configurations | `outputs/stage_4_alns/stage_4_master_summary.csv` |
| Fleet-Wide Capacity Utilisation Analysis | `outputs/analysis/fleet_capacity_utilisation_55.csv` |

For example, `outputs/stage_4_alns/full_instance/gini/lambda_0_100/` contains the final Gini ($\lambda=0.1$) solution, route summaries, route stops, assignments, driver workloads and ALNS diagnostics.

## Installation

The canonical development environment used **Python 3.10.10**. The included requirements specify the application dependencies and additional optimisation packages. Compatibility with every Python version or operating system has not been established.

```bash
# Clone the repository.
git clone https://github.com/benfrost331/honours-vrp-equity-public.git
cd honours-vrp-equity-public

# macOS / Linux: create and activate an environment.
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies for dashboards and stored-result analysis.
python -m pip install -r requirements.txt
```

On Windows PowerShell, create the environment with `py -3.10 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`.

For solver-based reproduction, additionally install:

```bash
python -m pip install -r requirements-optimisation.txt
python -c "import pulp; print(pulp.listSolvers(onlyAvailable=True))"
```

The code offers **CBC, HiGHS and Gurobi** solver choices. Canonical Stage 2 results were produced with Gurobi; reproducing those runs requires a working Gurobi installation and an appropriate licence. Alternative solvers may give different stopping behaviour and numerical outcomes.

## Running the Dashboards Locally

From the repository root, with the application requirements installed:

```bash
python -m streamlit run app.py         # Stage 2
python -m streamlit run app_stage3.py  # Stage 3
python -m streamlit run app_stage4.py  # Stage 4
```

Run **one application per terminal session**. All three read the committed canonical result files; no optimisation rerun is necessary to explore the reported results.

## Reproducing the Computational Workflow

**Important:** The repository already contains the canonical study outputs. The following scripts can overwrite existing files. Run the computational workflow in a **separate working copy** if you intend to reproduce or modify experiments. In particular, a subset Stage 4 run may replace its master summary with a partial summary even when individual configuration outputs are preserved.

The dependency order is:

```bash
# Stage 1: cleaning, validation and order-day feasibility
python main.py

# Stage 2: MILP sweeps over all five equity measures and 11 lambda values
# Per-solve target: 3% relative gap; limit: 7200 s; Gurobi with 4 threads
python run_full_equity_experiments.py \
  --solver gurobi --gap 0.03 --time-limit 7200 --threads 4
python run_build_equity_master_summary.py

# Stage 3: directed route construction and feasibility repair
python run_stage_3_equity_routing.py
python run_stage_3_equity_repair.py

# Stage 4: feasibility-aware ALNS improvement
python run_stage_4_alns.py --max-iterations 100 --random-seed 42
python run_build_stage_4_master_summary.py

# Post-processing and report figures
python compute_fleet_capacity_utilisation.py
python run_generate_report_figures.py
```

The Stage 2 runner can accept selected measures and weights, for example `--equity-measures maximum --lambda-values 0 0.1`, and supports `--no-resume`. Stage 3 and Stage 4 runners support `--equity-measures`, `--lambda-values` and `--no-overwrite`. These options do **not** remove the need to protect canonical outputs. Check each runner's `--help` before adapting the workflow.

Stage 2 builds `data/processed/assignment_cost_matrix.csv` if that file is absent; the committed processed matrix can be used to avoid regenerating it. Stage 3 and Stage 4 use the processed travel, feasibility and assignment-cost data. Computational reproducibility also depends on solver version, licence, hardware, time limits, warm starts and stochastic search seeds.

### Lightweight Verification of the Stored Results

The two master summaries each contain 55 configurations. With `requirements.txt` installed, this command checks the recorded configuration grid and summary feasibility flags without changing any files:

```bash
python - <<'PY'
import pandas as pd
s2 = pd.read_csv('outputs/equity_experiments/full_instance/equity_master_summary.csv')
s4 = pd.read_csv('outputs/stage_4_alns/stage_4_master_summary.csv')
keys = ['equity_measure_type', 'lambda_tag']
assert len(s2) == len(s4) == 55
assert not s2.duplicated(keys).any() and not s4.duplicated(keys).any()
assert set(map(tuple, s2[keys].to_numpy())) == set(map(tuple, s4[keys].to_numpy()))
assert s2['solution_valid'].astype(str).str.lower().eq('true').all()
assert s4['all_routes_feasible'].astype(str).str.lower().eq('true').all()
print('PASS: 55 matched configurations; stored validity and feasibility flags hold.')
PY
```

This checks reported result integrity; it does not replace the underlying route-feasibility validators or independently certify global optimality.

## Scope & Limitations

- **Approximate Assignment Costs:** Stage 2 uses independent return-trip proxies rather than actual multi-stop routed duration.
- **Layered Optimisation:** Stage 3 repair and Stage 4 ALNS do not establish global optimality of the integrated vehicle routing problem.
- **Finite Solver Limits:** Six Stage 2 configurations retain feasible incumbents without the requested solver stopping certificate.
- **Deterministic Operations:** Travel times and input demand are treated as fixed; dynamic orders, stochastic congestion and disruptions are outside scope.
- **Workload-Equity Definition:** Time-based workloads do not directly measure fatigue, employee preference, job difficulty or all dimensions of fairness.
- **Economic Scope:** The main efficiency metric is duration; a full monetary costing model is not included.
- **Generalisability:** Findings are specific to the provided routing instance and modelling assumptions.

## Authors, Citation & Reuse

**Ben Frost** and **Jadene Jaftha**, under the supervision of **Dr Georgina Rakotonirainy**, University of Cape Town (2026).

Suggested Academic Citation:

> Frost, B., & Jaftha, J. (2026). *Decision Support Models for Multi-Period Vehicle Routing with Equity Considerations*. BSc Honours Research Project, University of Cape Town.

