# reliable-training-control

Chapter 4 experiment code for reliability-aware training-time control in non-convex deep learning.

The project studies training-time decisions based on PAC-Bayes generalization signals and primal-dual optimality indicators. It includes experiment execution, reliability metrics, case-study tables, and Nature-style figures.

## Layout

```text
.
+-- configs/                 # Experiment configurations
+-- scripts/                 # Table and figure generation scripts
+-- src/                     # Data, models, metrics, controller, and signal code
+-- requirements.txt         # Python dependencies
+-- run_all_main.sh          # Recommended batch run
+-- run_supplementary_cifar100.sh  # CIFAR-100 supplementary architecture run
+-- run_experiment.py        # Main experiment entry point
+-- README.md
`-- LICENSE
```

## Setup

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

PyCharm should use this interpreter:

```text
D:\reliable-training-control\.venv\Scripts\python.exe
```

## Quick Test

```powershell
.\.venv\Scripts\python.exe run_experiment.py --config configs\quick_cifar10_resnet18.json
```

The experiment configs request the local GPU with:

```json
"device": "cuda"
```

If PyTorch cannot access CUDA, the run exits early instead of silently falling back to CPU.

After the quick run, generate tables and figures:

```powershell
.\.venv\Scripts\python.exe generate_case_study_tables.py --results-dir results --out-dir tables
.\.venv\Scripts\python.exe make_case_study_figures.py --results-dir results --out-dir figures
```

Conclusion-style figures and tables can also be regenerated with:

```powershell
.\.venv\Scripts\python.exe generate_conclusion_tables.py --results-dir results --out-dir tables_conclusion
.\.venv\Scripts\python.exe make_conclusion_figures.py --results-dir results --out-dir figures_conclusion
```

## Experiment Matrix

Main full case-study experiments:

- CIFAR-10 + ResNet-18
- CIFAR-100 + ResNet-18

Supplementary architecture-level validation:

- CIFAR-10 + ResNet-50
- CIFAR-100 + ResNet-50
- CIFAR-10 + VGG-16
- CIFAR-100 + VGG-16

ResNet-18 is used for the full trajectory analysis because it provides a tractable setting for detailed signal, control-event, calibration, and seed-stability studies. ResNet-50 and VGG-16 test whether the reliability-aware controller remains effective as model capacity increases and when the architecture changes.

## Recommended Main Run

On a shell with `bash` available:

```bash
bash run_all_main.sh
```

This runs:

- CIFAR-10 + ResNet-18
- CIFAR-100 + ResNet-18
- CIFAR-10 + ResNet-50
- CIFAR-100 + ResNet-50
- CIFAR-10 + VGG-16
- CIFAR-100 + VGG-16

To run only the new CIFAR-100 supplementary architecture experiments:

```bash
bash run_supplementary_cifar100.sh
```

## Outputs

Runtime artifacts are ignored by Git:

- `data/`
- `results/`
- `tables/`
- `figures/`

These outputs can be regenerated from the experiment configs and scripts.
