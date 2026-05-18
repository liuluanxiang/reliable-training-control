#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="sensitivity_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="results/${RUN_TAG}"
PROGRESS_LOG="results/progress_${RUN_TAG}.log"
EXPERIMENT_SUFFIX="__${RUN_TAG}"

for config in configs/sensitivity/*.json; do
  python run_experiment.py --config "$config" --results-dir "$RESULTS_DIR" --experiment-suffix "$EXPERIMENT_SUFFIX" --existing error --progress batch --progress-log "$PROGRESS_LOG" --progress-state results/progress_state.json --dashboard-path results/progress_dashboard.html
done

python generate_case_study_tables.py --results-dir "$RESULTS_DIR" --out-dir "tables/${RUN_TAG}"
python make_case_study_figures.py --results-dir "$RESULTS_DIR" --out-dir "figures/${RUN_TAG}"

if [ -f generate_conclusion_tables.py ]; then
  python generate_conclusion_tables.py --results-dir "$RESULTS_DIR" --out-dir "tables_conclusion/${RUN_TAG}"
fi

if [ -f make_conclusion_figures.py ]; then
  python make_conclusion_figures.py --results-dir "$RESULTS_DIR" --out-dir "figures_conclusion/${RUN_TAG}"
fi

if [ -f generate_statistical_tables.py ]; then
  python generate_statistical_tables.py --results-dir "$RESULTS_DIR" --out-dir "statistical_tables/${RUN_TAG}"
fi

if [ -f make_statistical_figures.py ]; then
  python make_statistical_figures.py --results-dir "$RESULTS_DIR" --out-dir "statistical_figures/${RUN_TAG}"
fi

python audit_experiment_results.py \
  --results-dir "$RESULTS_DIR" \
  --suite sensitivity \
  --check-statistical \
  --statistical-tables-dir "statistical_tables/${RUN_TAG}" \
  --statistical-figures-dir "statistical_figures/${RUN_TAG}"
