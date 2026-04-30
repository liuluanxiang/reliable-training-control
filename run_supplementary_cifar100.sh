#!/usr/bin/env bash
set -e

python run_experiment.py --config configs/cifar100_resnet50_supplement.json
python run_experiment.py --config configs/cifar100_vgg16_supplement.json

python generate_case_study_tables.py --results-dir results --out-dir tables
python make_case_study_figures.py --results-dir results --out-dir figures

if [ -f generate_conclusion_tables.py ]; then
  python generate_conclusion_tables.py --results-dir results --out-dir tables_conclusion
fi

if [ -f make_conclusion_figures.py ]; then
  python make_conclusion_figures.py --results-dir results --out-dir figures_conclusion
fi
