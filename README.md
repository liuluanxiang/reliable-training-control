# reliable-training-control

Chapter 4 experiment code for reliability-aware training-time control in non-convex deep learning.

The project studies training-time decisions based on PAC-Bayes generalization signals and primal-dual optimality indicators. It keeps the core experiment runner, batch runner, main case-study postprocessing, ablation and sensitivity suites, and paired statistical tests.

## Layout

```text
.
+-- configs/                 # Experiment configurations
+-- scripts/                 # Main case-study table and figure scripts
+-- src/                     # Data, models, metrics, controller, and signal code
+-- generate_statistical_tables.py
+-- make_statistical_figures.py
+-- audit_experiment_results.py
+-- requirements.txt         # Python dependencies
+-- run_batch.py             # Cross-platform batch runner
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

## Live Progress and Result Safety

`run_experiment.py` prints live progress by default. The output includes:

- current config, method, seed, and run index
- train/eval batch heartbeat
- epoch time and average epoch time
- estimated time remaining for the current run
- best validation accuracy so far
- existing-result audit before training starts

It also writes a live state file and dashboard:

```text
results/progress_state.json
results/progress_dashboard.html
```

To view the dashboard in a browser, serve the results directory:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 -d .\results
```

Then open:

```text
http://localhost:8000/progress_dashboard.html
```

Progress options:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --config configs\quick_cifar10_resnet18.json --progress batch --progress-log results\progress_quick.log
```

Use `--progress epoch` for less console output, or `--progress none` to silence progress messages.

Result collision protection is enabled by default:

```text
--existing error
```

For each planned `experiment/method/seed`, the runner checks the canonical output directory:

```text
results/<experiment_name>/<method>/seed_<seed>/
```

If any artifacts already exist, the run aborts before training so that no file is overwritten and no result is silently skipped. To inspect what would run before starting training:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --config configs\cifar10_resnet18_main.json --dry-run
```

If you intentionally need to rerun a completed or partial seed without losing the old output, use:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --config configs\cifar10_resnet18_main.json --existing archive
```

This moves the old seed directory under:

```text
results/_archive/<run_id>/
```

and then writes the fresh result to the normal path. `--existing overwrite` restores the old behavior and should only be used when replacing old artifacts is intentional.

The batch scripts create isolated output namespaces. A run tag such as:

```text
main_supplement_20260501_120000
```

is appended to experiment names and used in output directories:

```text
results/main_supplement_20260501_120000/
tables/main_supplement_20260501_120000/
figures/main_supplement_20260501_120000/
statistical_tables/main_supplement_20260501_120000/
statistical_figures/main_supplement_20260501_120000/
```

This keeps main, supplementary, ablation, sensitivity, table, figure, and CSV artifacts from overwriting or being skipped because of previous runs.

After the quick run, generate tables and figures:

```powershell
.\.venv\Scripts\python.exe scripts\generate_case_study_tables.py --results-dir results --out-dir tables
.\.venv\Scripts\python.exe scripts\make_case_study_figures.py --results-dir results --out-dir figures
```

By default, table and figure scripts exclude quick-test runs. Pass `--include-quick` if a diagnostic output should include them.

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

## Result Naming

Each config defines an `experiment_type`, and results are saved under:

```text
results/<experiment_name>/<method>/seed_<seed>/
```

The naming convention is:

```text
<experiment_type>_<dataset>_<model>
```

with compact prefixes for supplementary and sensitivity experiments:

- `quick_cifar10_resnet18`
- `main_cifar10_resnet18`
- `main_cifar100_resnet18`
- `supp_cifar10_resnet50`
- `supp_cifar100_resnet50`
- `supp_cifar10_vgg16`
- `supp_cifar100_vgg16`
- `ablation_cifar10_resnet18`
- `sens_cifar10_resnet18_alpha050_beta090`

`experiment_type` is saved into `history.csv`, `summary.json`, and `results/master_summary.csv`. Older result folders without this field are still supported; scripts infer the type from `experiment_name`.

## Recommended Main Run

Run the full main and supplementary batch:

```powershell
.\.venv\Scripts\python.exe run_batch.py main
```

This runs:

- CIFAR-10 + ResNet-18
- CIFAR-100 + ResNet-18
- CIFAR-10 + ResNet-50
- CIFAR-100 + ResNet-50
- CIFAR-10 + VGG-16
- CIFAR-100 + VGG-16

To run only the new CIFAR-100 supplementary architecture experiments:

```powershell
.\.venv\Scripts\python.exe run_batch.py supplementary-cifar100
```

## Additional Analyses

The pipeline also includes three optional analyses for Chapter 4.

1. Ablation study:
   Full vs PB-only vs PD-only vs no-smoothing vs validation-loss-only.

2. Hyperparameter sensitivity:
   alpha in {0.25, 0.50, 0.75}
   beta in {0.80, 0.90, 0.95}

3. Statistical tests:
   Ours vs best baseline using paired seeds. The statistical table reports bootstrap 95% CI, paired t-test, Wilcoxon test, Holm-corrected p-values, Cohen's d, pair counts, and significance markers.

Usage:

```powershell
.\.venv\Scripts\python.exe run_batch.py ablation
.\.venv\Scripts\python.exe run_batch.py sensitivity
.\.venv\Scripts\python.exe generate_statistical_tables.py --results-dir results --out-dir statistical_tables
.\.venv\Scripts\python.exe make_statistical_figures.py --results-dir results --out-dir statistical_figures
```

The statistical figure script writes:

- `fig_4_11_ablation_study.pdf/png`
- `fig_4_12_sensitivity_analysis.pdf/png`
- `fig_4_13_statistical_tests.pdf/png`

## Outputs

Runtime artifacts are ignored by Git:

- `data/`
- `results/`
- `tables/`
- `figures/`
- `statistical_tables/`
- `statistical_figures/`
- `outputs/`
- `*.rar`

These outputs can be regenerated from the experiment configs and scripts.

## Paper 1 Major Revision Workflow

The corrected revision protocol is isolated under `revision_experiments/`. Historical results use
the legacy augmented-validation protocol and are not reused for revision inference.
Formal protocol-v2 per-run artifacts are stored with the existing experiment archive under
`results_v2/<experiment>/<method>/seed_<n>/`. The `revision_experiments/results/` directory is
reserved for queue manifests, live state, logs, runtime estimates, smoke runs, and aggregated
revision analysis outputs.

```powershell
# Download/check Tiny ImageNet and the official DeiT-Tiny checkpoint.
.\revision_experiments\scripts\download_external_assets.ps1

# Regenerate the audit, versioned configs, and exact missing-run matrix.
.\.venv\Scripts\python.exe .\revision_experiments\scripts\audit_repository.py
.\.venv\Scripts\python.exe -m unittest tests.test_revision_protocol -v

# Benchmark all compute families and calculate the conservative completion estimate.
.\.venv\Scripts\python.exe .\revision_experiments\scripts\benchmark_runtime.py
.\.venv\Scripts\python.exe .\revision_experiments\scripts\estimate_runtime.py

# Inspect the 260-run physical queue without launching it.
.\.venv\Scripts\python.exe .\revision_experiments\scripts\run_revision.py

# Run the two five-method, one-seed smoke configurations before the full queue.
.\.venv\Scripts\python.exe .\run_experiment.py --config .\revision_experiments\configs\tiny_imagenet\smoke.json --existing rerun-partial
.\.venv\Scripts\python.exe .\run_experiment.py --config .\revision_experiments\configs\deit_cifar100\smoke.json --existing rerun-partial

# After smoke review, launch sequential corrected-protocol runs and generate analysis outputs.
.\.venv\Scripts\python.exe .\revision_experiments\scripts\run_revision.py --execute
.\.venv\Scripts\python.exe .\revision_experiments\scripts\analyze_revision.py
```

`revision_experiments/results/run_manifest.csv` is rewritten after every run and records completion/failure status. Completed
outputs are skipped only when their scientific configuration hash matches the requested run.
