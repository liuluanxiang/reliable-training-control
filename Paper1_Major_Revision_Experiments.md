# Paper 1 Major-Revision Experimental Plan for Codex

**Paper:** Reliability-Aware Training-Time Control for Non-Convex Deep Learning
**Purpose:** Execute the complete experimental revision requested by the supervisor before journal submission.
**Execution environment:** Existing PyCharm/PyTorch project; single-GPU execution; reuse the existing valid implementation and valid completed results whenever possible.
**Core rule:** **Do not optimize for prettier results. Optimize for a clean, fair, reproducible causal answer.**

---

# 0. What This Revision Must Establish

The revision must answer the following scientific question:

> **Does the proposed PB+PD multi-signal controller provide a reproducible and worthwhile reliability–efficiency benefit compared with four strong baselines, and is any benefit attributable to the multi-signal controller rather than merely to early stopping, arbitrary scaling, or unequal training budget?**

The revised paper will therefore use:

## Four baseline methods

1. **Step Decay**
2. **Cosine Annealing**
3. **ReduceLROnPlateau**
4. **Validation-Loss-Only Controller**

## Proposed method

5. **PB+PD Reliability-Aware Controller**

For **all primary and external-validation comparisons**, use **5 matched random seeds**.

---

# 1. Mandatory Scientific Principles

## 1.1 Do not redesign the existing method silently

Before any long experiment, Codex must audit the existing repository and determine the exact current implementation of:

- PB signal
- PD signal
- combined controller/control score
- online standardization
- exponential smoothing
- learning-rate reduction rule
- early-stopping rule
- checkpoint-selection rule
- optimizer
- initial learning rate
- weight decay
- batch size
- augmentations
- train/validation/test split
- maximum epochs by dataset/model
- random seeds
- `alpha`
- `beta`
- `gamma`
- `delta`
- `sigma`
- `lambda_g`
- `lambda_u`
- minimum learning rate
- LR patience
- stopping patience
- min-delta / tolerance
- generalization-gap implementation
- ECE implementation
- Brier implementation
- AURC implementation
- late-stage degradation implementation

Create:

```text
revision_experiments/audit/current_implementation_audit.md
revision_experiments/audit/current_controller_config.json
revision_experiments/audit/controller_hyperparameters.csv
```

Every parameter in `controller_hyperparameters.csv` must contain:

```text
parameter
value
scope: global / dataset-specific / model-specific
where_defined
how_selected
tuned: yes/no/unknown
notes
```

If a value cannot be found, write:

```text
UNRESOLVED
```

Do **not** invent it.

---

## 1.2 Existing results must be reused when valid

Before scheduling new runs, scan the project for:

- completed checkpoints;
- per-run JSON/CSV logs;
- seed-level metric files;
- previous ablation results;
- sensitivity results;
- timing logs.

A completed run can be reused only if its configuration can be verified.

For each reusable run create:

```text
dataset
model
method
seed
config_hash
split_hash
initialization_hash
checkpoint_path
metrics_path
reusable = true/false
reason
```

Save:

```text
revision_experiments/audit/reusable_runs.csv
```

Do not rerun valid experiments unnecessarily.

---

# 2. Data Leakage Prevention — Mandatory

Use:

```text
TRAIN
    parameter optimization only

VALIDATION
    PB/PD quantities requiring validation information
    LR control
    early stopping
    hyperparameter selection
    checkpoint selection

TEST
    final evaluation only
```

The test set must **never** be used for:

- LR scheduling;
- stopping;
- hyperparameter selection;
- choosing `alpha`, `beta`, `sigma`, `lambda_g`, or `lambda_u`;
- checkpoint selection;
- deciding which ablation variant to retain.

For each run save:

```text
test_evaluation_count
test_evaluation_timestamp
checkpoint_selected_before_test
test_used_during_training
```

A run is invalid if:

```text
test_used_during_training == true
```

Create:

```text
revision_experiments/audit/data_leakage_protocol.md
```

---

# 3. Reproducibility and Matched Comparison

Within any matched comparison group, keep identical:

```text
dataset
split
architecture
seed
initial parameters
data-order seed
augmentation seed
optimizer
batch size
base LR
weight decay
maximum training budget
precision
checkpoint-selection protocol
```

Only the intended method/ablation component may change.

For paired methods under the same seed, save:

```text
split_hash
initialization_hash
data_order_seed
```

Before statistical analysis, automatically verify that matched runs satisfy these equality constraints.

---

# 4. Main Experimental Matrix — Complete to 5 Seeds

## 4.1 Datasets

```text
CIFAR-10
CIFAR-100
```

## 4.2 Architectures

```text
ResNet-18
ResNet-50
VGG-16
```

## 4.3 Methods

```text
Step Decay
Cosine Annealing
ReduceLROnPlateau
Validation-Loss-Only Controller
Proposed PB+PD Controller
```

## 4.4 Seeds

```text
5 matched seeds per dataset × model × method
```

Final complete matrix:

```text
2 datasets × 3 architectures × 5 methods × 5 seeds
= 150 total main-matrix runs
```

Existing valid runs must be reused.

### Current completion expectation to verify

```text
ResNet-18:
    existing main experiments may already have 5 seeds

ResNet-50:
    existing main experiments may currently have only 3 seeds

VGG-16:
    existing main experiments may currently have only 3 seeds
```

Codex must determine the actual missing runs after the audit.

---

# 5. Required Metrics for Every Main Run

Save the following per run:

```text
dataset
model
method
seed

max_epochs
epochs_run
best_epoch
stopping_epoch
first_lr_reduction_epoch
num_lr_reductions
final_lr

test_accuracy
test_nll
test_ece
test_brier
test_aurc
generalization_gap
late_stage_degradation

wall_clock_seconds
avg_seconds_per_epoch
gpu_hours
peak_gpu_allocated_mb
peak_gpu_reserved_mb

controller_total_seconds
controller_avg_seconds_per_epoch
controller_overhead_percent

checkpoint_path
config_path
config_hash
split_hash
initialization_hash
git_commit
status
```

Output:

```text
revision_experiments/results/main_5seed/per_run_metrics.csv
revision_experiments/results/main_5seed/summary_by_setting.csv
```

---

# 6. Supervisor Concern: Why This PAC-Bayes-Inspired Form?

This is mainly a theoretical/manuscript issue, but the experiment package must preserve evidence needed to support it.

For every run save:

```text
parameter_displacement
PB_t
validation_NLL
empirical_generalization_gap
```

The revision report must be able to produce:

```text
rho(PB_t, validation_NLL)
rho(PB_t, generalization_gap)
PB trajectory
parameter-displacement trajectory
```

Do not call `PB_t` a formal PAC-Bayes certificate.

Preferred terminology in code/report:

```text
PAC-Bayes-inspired generalization signal
```

---

# 7. Experiment A — Sigma Sensitivity

## 7.1 Objective

Determine whether the controller conclusions depend excessively on the choice of `sigma`.

First identify:

```text
sigma_0 = exact current default sigma
```

Record:

```text
value
source file
how selected
fixed across models/datasets? yes/no
```

Create:

```text
revision_experiments/audit/sigma_audit.md
```

---

## 7.2 Sensitivity grid

Use:

```text
0.5 × sigma_0
1.0 × sigma_0
2.0 × sigma_0
4.0 × sigma_0
```

Representative setting:

```text
CIFAR-10
ResNet-18
Proposed controller
5 matched seeds
```

The default `1.0 × sigma_0` runs may be reused if valid.

Do not tune any other controller parameter separately for each sigma.

---

## 7.3 Save per epoch

```text
PB_t
PD_t
combined_control_signal
learning_rate
validation_NLL
generalization_gap
parameter_displacement
```

Save per run:

```text
first_lr_reduction_epoch
num_lr_reductions
stopping_epoch
epochs_run
accuracy
NLL
ECE
Brier
AURC
generalization_gap
wall_clock
```

Required outputs:

```text
revision_experiments/results/sigma_sensitivity/sigma_runs.csv
revision_experiments/figures/sigma_pb_trajectory.png
revision_experiments/figures/sigma_control_signal.png
revision_experiments/figures/sigma_endpoint_metrics.png
revision_experiments/figures/sigma_intervention_timing.png
```

---

# 8. Experiment B — PD Internal Component Ablation

## 8.1 Objective

The existing PD diagnostic is:

\[
PD_t = V_t + \lambda_g H_t + \lambda_u U_t
\]

where:

```text
V_t = positive train–validation risk mismatch
H_t = gradient norm
U_t = parameter-update norm
```

The revision must establish whether `H_t` and `U_t` contribute useful information beyond `V_t`.

---

## 8.2 First audit lambda values

Find and document the actual implementation values of:

```text
lambda_g
lambda_u
```

Do not assume values from notes if the code differs.

Create:

```text
revision_experiments/audit/pd_parameter_audit.md
```

---

## 8.3 Variants

Run:

```text
PD_FULL:
    PD_t = V_t + lambda_g * H_t + lambda_u * U_t

PD_V_ONLY:
    PD_t = V_t

PD_V_H:
    PD_t = V_t + lambda_g * H_t

PD_V_U:
    PD_t = V_t + lambda_u * U_t

PD_H_U:
    PD_t = lambda_g * H_t + lambda_u * U_t
```

All other controller components remain fixed.

Representative setting:

```text
CIFAR-10
ResNet-18
5 matched seeds
```

Reuse valid `PD_FULL` results when possible.

---

## 8.4 Required analysis

Endpoint:

```text
Accuracy
NLL
ECE
Brier
AURC
Generalization gap
Late-stage degradation
Epochs run
Wall-clock
GPU-hours
```

Trajectory correlations:

```text
rho(PD_t, V_t)
rho(PD_t, H_t)
rho(PD_t, U_t)
```

Decision differences:

```text
first LR-reduction epoch
number of LR reductions
stopping epoch
number of epochs with different LR state relative to PD_FULL
```

Create:

```text
revision_experiments/results/pd_component_ablation/pd_ablation_runs.csv
revision_experiments/results/pd_component_ablation/pd_ablation_summary.csv
revision_experiments/figures/pd_component_endpoint_metrics.png
revision_experiments/figures/pd_component_intervention_timing.png
revision_experiments/figures/pd_component_correlations.png
```

The final report must answer:

> **Do gradient norm and/or update norm materially change controller decisions, stability, reliability, or efficiency beyond using risk mismatch alone?**

Negative findings must be retained.

---

# 9. Experiment C — Fixed-Budget vs Adaptive-Budget

## 9.1 Objective

Separate these two questions:

```text
Is the signal/control policy better under equal training opportunity?

Does the natural stopping rule reduce training cost?
```

---

## 9.2 Methods

Use:

```text
Step Decay
Cosine Annealing
ReduceLROnPlateau
Validation-Loss-Only Controller
Proposed PB+PD Controller
```

Also include internal controller ablations where existing results make this practical:

```text
PB-only
PD-only
No-smoothing
```

For the manuscript's headline fairness comparison, the 5 primary methods are mandatory.

---

## 9.3 Fixed-budget mode

Representative setting first:

```text
CIFAR-10
ResNet-18
5 matched seeds
```

Use the exact existing maximum epoch budget.

Expected historical CIFAR-10 budget may be 90 epochs, but Codex must verify this.

Fixed-budget rules:

```text
disable training termination caused by early stopping
retain LR scheduling/control logic
train all methods to identical maximum epochs
retain identical checkpoint-selection rule
```

---

## 9.4 Adaptive-budget mode

Use each method's natural stopping behaviour.

For methods without early stopping, train according to their existing design.

Reuse valid existing runs.

---

## 9.5 Output

```text
revision_experiments/results/budget_comparison/budget_runs.csv
revision_experiments/results/budget_comparison/budget_summary.csv
```

Required comparison:

```text
Fixed-budget endpoint performance
Adaptive-budget endpoint performance
Epoch reduction
Wall-clock reduction
GPU-hour reduction
Endpoint performance change
```

---

# 10. Experiment D — Performance vs Computational Cost

This experiment uses timing data from all relevant runs.

## 10.1 Mandatory runtime instrumentation

Measure:

```text
training time per epoch
total wall-clock training time
GPU-hours
peak GPU allocated memory
peak GPU reserved memory
PB computation time
PD computation time
standardization/smoothing time
controller decision time
checkpoint overhead
total controller overhead
```

Use GPU synchronization around precise GPU timing:

```python
torch.cuda.synchronize()
```

Use warm-up before timing.

Do not compare a cold first run against warmed runs.

---

## 10.2 Controller overhead

Compute:

\[
\text{Controller Overhead (\%)} =
\frac{T_{\text{controller}}}{T_{\text{total training}}}\times 100
\]

Also report:

\[
\Delta T_{\text{epoch}} =
T_{\text{proposed per epoch}} - T_{\text{baseline per epoch}}
\]

---

## 10.3 Required figures

```text
accuracy_vs_epochs.png
nll_vs_epochs.png
brier_vs_epochs.png

accuracy_vs_wallclock.png
nll_vs_wallclock.png
brier_vs_wallclock.png

accuracy_vs_gpu_hours.png
nll_vs_gpu_hours.png
brier_vs_gpu_hours.png

pareto_accuracy_vs_gpu_hours.png
pareto_brier_vs_gpu_hours.png
```

For:

```text
Accuracy vs GPU-hours:
    upper-left is better

Brier vs GPU-hours:
    lower-left is better
```

Do not hide dominated or negative results.

---

# 11. Experiment E — External Dataset Validation: Tiny ImageNet

## 11.1 Role

Tiny ImageNet is **external validation**, not the primary development dataset.

Purpose:

> Test whether the controller remains useful beyond CIFAR under increased class and visual complexity.

---

## 11.2 Configuration

```text
Dataset:
    Tiny ImageNet

Architecture:
    ResNet-18

Methods:
    Step Decay
    Cosine Annealing
    ReduceLROnPlateau
    Validation-Loss-Only Controller
    Proposed PB+PD Controller

Seeds:
    5 matched seeds
```

Total:

```text
5 methods × 5 seeds = 25 runs
```

---

## 11.3 Do not invent the Tiny ImageNet recipe

Audit whether the existing thesis/project already defines:

```text
image size
normalization
augmentation
optimizer
base LR
weight decay
batch size
maximum epochs
validation construction
scheduler configuration
```

If existing configuration exists, reuse it.

If not, create:

```text
revision_experiments/audit/tiny_imagenet_config_unresolved.md
```

and mark unresolved fields.

Do not launch the full 25-run experiment until the configuration is explicit.

---

## 11.4 Smoke test

Before full execution:

```text
1 seed
5 epochs
all 5 methods if possible
```

Verify:

```text
dataset loads correctly
class count correct
no NaN
VRAM safe
metrics correct
checkpoint correct
controller signals have finite values
```

Then run 5 seeds.

---

## 11.5 Required metrics

Primary:

```text
Accuracy
NLL
Brier
Generalization gap
Cross-seed variability
GPU-hours
```

Supporting:

```text
ECE
AURC
Late-stage degradation
First LR reduction
Stopping epoch
Wall-clock
```

Output:

```text
revision_experiments/results/external_tiny_imagenet/
```

---

# 12. Experiment F — External Architecture Validation: Selected ViT

## 12.1 Role

Purpose:

> Test whether the controller transfers beyond CNN architectures.

---

## 12.2 Configuration

```text
Dataset:
    CIFAR-100

Architecture:
    selected ViT already defined in the project / thesis

Methods:
    Step Decay
    Cosine Annealing
    ReduceLROnPlateau
    Validation-Loss-Only Controller
    Proposed PB+PD Controller

Seeds:
    5 matched seeds
```

Total:

```text
5 methods × 5 seeds = 25 runs
```

---

## 12.3 ViT selection rule

Codex must first inspect the repository.

Use this priority:

```text
1. Reuse the ViT/DeiT implementation already present in the project.
2. Prefer the architecture already identified in the thesis/proposal.
3. Preserve its existing input size and training recipe.
4. If multiple ViTs exist, document them and use the currently designated selected ViT.
5. If no ViT exists, STOP before choosing one automatically.
```

If unresolved, create:

```text
revision_experiments/audit/vit_selection_unresolved.md
```

Do not silently download or substitute a different architecture.

---

## 12.4 Smoke test

Before full execution:

```text
1 seed
5 epochs
all 5 methods
```

Measure:

```text
seconds_per_epoch
peak_VRAM
estimated_full_run_hours
```

Then run all 5 seeds.

Output:

```text
revision_experiments/results/external_vit/
```

---

# 13. Experiment G — Full Baseline Transparency

For every primary setting and external-validation setting, export results for **all five methods**.

Do not only export:

```text
Proposed vs strongest baseline
```

Main reporting table:

```text
Dataset
Model
Method
Accuracy
NLL
ECE
Brier
AURC
Generalization gap
Late-stage degradation
Epochs
Wall-clock
GPU-hours
Seed SD
```

Create:

```text
revision_experiments/results/full_baseline_tables/all_methods_per_setting.csv
revision_experiments/results/full_baseline_tables/all_methods_summary.csv
```

The manuscript may still include a metric-specific strongest-baseline comparison, but the complete table must be available for transparency.

---

# 14. Reliability Operational Definition

The analysis scripts must not treat reliability as one scalar outcome.

Organize empirical evidence into:

## Predictive performance

```text
Accuracy
```

## Probability quality

```text
NLL
Brier score
```

## Calibration

```text
ECE
```

## Generalization

```text
Empirical generalization gap
```

## Stability / reproducibility

```text
Across-seed SD
Paired consistency
Late-stage degradation
```

## Selective prediction

```text
AURC
```

## Efficiency

```text
Epochs
Wall-clock
GPU-hours
Controller overhead
```

Preferred name for the PB+PD combined internal quantity:

```text
reliability-oriented control score
```

or:

```text
reliability-control signal
```

Do not state that it directly measures all empirical reliability dimensions.

---

# 15. Early-Stopping Rule — Exact Reproducibility Audit

Create:

```text
revision_experiments/audit/early_stopping_logic.md
```

Extract the exact implemented logic and write:

```text
score improvement definition
validation improvement definition
min_delta / epsilon
reliability stagnation counter
validation stagnation counter
LR reduction condition
stopping condition
AND/OR relationship
counter reset logic
checkpoint retention rule
```

Also write exact pseudocode.

Do not modify the logic unless an actual bug is discovered.

If a bug is discovered:

```text
document old behaviour
document corrected behaviour
identify which historical runs are invalidated
```

---

# 16. Generalization-Gap Definition Audit

Create:

```text
revision_experiments/audit/generalization_gap_definition.md
```

Document:

```text
exact formula
metric domain: loss / accuracy / other
train quantity
validation quantity
sign
absolute value: yes/no
epoch used
checkpoint used
```

The code and manuscript must use the same definition.

---

# 17. Alpha = 0.5 Analysis

Do not automatically rerun the full alpha/beta grid unless existing files are incomplete.

Use the existing sensitivity experiment and determine:

```text
default alpha/beta setting
best setting by accuracy
best setting by NLL
best setting by ECE
best setting by Brier
best setting by generalization gap
best setting by epochs
```

Create:

```text
revision_experiments/results/alpha_beta_existing_analysis.csv
```

Also audit the historical reason for selecting `alpha = 0.5`.

Allowed outcomes:

```text
documented a-priori choice
neutral/equal weighting chosen before final test evaluation
chosen by validation
heuristic
UNRESOLVED
```

Do not invent a justification.

---

# 18. Lambda_g and Lambda_u Reporting

Supervisor requires exact values and justification.

At minimum produce:

```text
lambda_g
lambda_u
actual values
where defined
whether tuned
whether fixed across datasets/models
why selected
```

If these values were not systematically studied and the PD component ablation shows sensitivity concerns, optionally perform a **local one-factor robustness check**:

```text
0.5 × default
1.0 × default
2.0 × default
```

on:

```text
CIFAR-10 + ResNet-18
3–5 matched seeds
```

This optional experiment is **not mandatory unless the audit indicates that parameter scaling is a major unresolved issue**.

Do not launch it automatically before reviewing PD ablation results.

---

# 19. Statistical Analysis — Mandatory Revision

## 19.1 Paired design

For every main comparison:

```text
pair by matched random seed
```

Report:

```text
n
mean
SD
paired mean difference
95% CI of paired difference
paired effect size
raw paired p-value
Holm-adjusted p-value
direction of effect
```

---

## 19.2 Multiple-comparison correction

Use Holm correction for each explicitly defined family of comparisons.

Save both:

```text
p_raw
p_holm
```

Never replace or discard raw values.

---

## 19.3 Effect sizes

Use an appropriate paired effect size.

The implementation must document the formula.

If using paired Cohen's `d_z`, save:

```text
mean_difference
sd_difference
d_z
```

---

## 19.4 Small-n results

The headline primary analyses must use 5 seeds.

Any exploratory 3-seed analysis must be labeled:

```text
supporting / exploratory
```

Do not use it for strong significance claims.

---

# 20. Statistical Output Files

Generate:

```text
revision_experiments/results/statistics/
    paired_results_raw.csv
    paired_results_holm.csv
    effect_sizes.csv
    confidence_intervals.csv
    consistency_summary.csv
```

Also generate a human-readable:

```text
revision_experiments/results/statistics/STATISTICAL_REPORT.md
```

---

# 21. Runtime Estimation — Codex Must Calculate Actual Time

Do not rely only on a fixed hand estimate.

Codex must calculate the real ETA from measured throughput.

---

## 21.1 Benchmark before long runs

For each major compute family, run a short benchmark:

```text
CIFAR-10 + ResNet-18
CIFAR-100 + ResNet-50
CIFAR-100 + VGG-16
Tiny ImageNet + ResNet-18
CIFAR-100 + selected ViT
```

Benchmark:

```text
1 seed
5 timed epochs after warm-up
```

Record:

```text
seconds_per_epoch
peak_VRAM
controller_seconds_per_epoch
```

---

## 21.2 Runtime formula

For a fixed-budget run:

\[
T_{\text{run}} =
T_{\text{epoch}} \times N_{\text{epochs}}
\]

For an adaptive run, use the historical mean stopping epoch for the same method/setting when available.

Otherwise use the maximum budget for conservative planning.

Calculate:

```text
estimated_hours_per_run
remaining_run_count
estimated_gpu_hours_by_experiment
estimated_total_gpu_hours
estimated_calendar_days_at_24h_per_day
estimated_calendar_days_at_20h_per_day
```

Create:

```text
revision_experiments/RUNTIME_ESTIMATE.md
revision_experiments/results/runtime_estimate.csv
```

---

## 21.3 Update ETA continuously

After every completed run:

```text
update measured mean seconds/epoch
update mean stopping epoch
update remaining GPU-hours
```

`REVISION_STATUS.md` must show:

```text
completed_runs
pending_runs
failed_runs
current_measured_gpu_hours
estimated_remaining_gpu_hours
estimated_completion_datetime
```

---

# 22. Expected Experiment Inventory

The final **complete design** is:

## Main matrix

```text
CIFAR-10 / CIFAR-100
× ResNet-18 / ResNet-50 / VGG-16
× 5 methods
× 5 seeds

= 150 total main-matrix runs
```

## External dataset

```text
Tiny ImageNet
× ResNet-18
× 5 methods
× 5 seeds

= 25 runs
```

## External architecture

```text
CIFAR-100
× selected ViT
× 5 methods
× 5 seeds

= 25 runs
```

## PD component ablation

```text
5 PD variants
× 5 seeds

= 25 runs
```

## Sigma sensitivity

```text
4 sigma levels
× 5 seeds

= 20 runs
```

## Budget experiment

Headline:

```text
5 primary methods
× 5 seeds
× fixed/adaptive modes
```

Reuse adaptive runs wherever valid.

Additional internal variants may be included for mechanism analysis.

---

# 23. Important: Total Runs vs New Runs

Do not confuse:

```text
final complete experiment inventory
```

with:

```text
new runs still required
```

After the audit, calculate:

```text
final_required_runs
valid_existing_runs
new_runs_required
```

The runtime estimate must use:

```text
new_runs_required
```

not the theoretical complete inventory.

---

# 24. Directory Structure

Use:

```text
revision_experiments/
├── audit/
│   ├── current_implementation_audit.md
│   ├── current_controller_config.json
│   ├── controller_hyperparameters.csv
│   ├── reusable_runs.csv
│   ├── data_leakage_protocol.md
│   ├── early_stopping_logic.md
│   ├── generalization_gap_definition.md
│   ├── sigma_audit.md
│   ├── pd_parameter_audit.md
│   ├── tiny_imagenet_config_unresolved.md
│   └── vit_selection_unresolved.md
│
├── configs/
│   ├── main_5seed/
│   ├── pd_ablation/
│   ├── budget_comparison/
│   ├── sigma_sensitivity/
│   ├── tiny_imagenet/
│   └── external_vit/
│
├── scripts/
│   ├── run_revision.py
│   ├── benchmark_runtime.py
│   ├── estimate_runtime.py
│   ├── aggregate_results.py
│   ├── statistical_analysis.py
│   ├── overhead_analysis.py
│   └── make_revision_figures.py
│
├── results/
│   ├── main_5seed/
│   ├── pd_component_ablation/
│   ├── budget_comparison/
│   ├── sigma_sensitivity/
│   ├── external_tiny_imagenet/
│   ├── external_vit/
│   ├── overhead/
│   ├── full_baseline_tables/
│   └── statistics/
│
├── figures/
├── logs/
├── REVISION_STATUS.md
├── RUNTIME_ESTIMATE.md
└── PAPER1_REVISION_EXPERIMENT_REPORT.md
```

Integrate with the existing trainer/model/dataset code.

Do not duplicate the entire training framework.

---

# 25. Resume and Failure Safety

Every run must have a unique ID:

```text
experiment_group
dataset
model
method
seed
budget_mode
variant
config_hash
```

Before launching:

```text
if valid completed result exists:
    skip

elif compatible checkpoint exists:
    resume

else:
    start new run
```

Never overwrite a completed run.

Failed runs must save:

```text
status = FAILED
exception
last_completed_epoch
checkpoint_path
timestamp
```

---

# 26. Automatic Run Validation

Flag a run if:

```text
NaN / Inf
wrong class count
missing checkpoint
metric file incomplete
test leakage
wrong seed
split hash mismatch
initialization hash mismatch in a paired experiment
unexpected budget mismatch
zero/impossible runtime
incompatible config hash
```

A flagged run must not enter final statistics until resolved.

---

# 27. Execution Order

## Stage 0 — Audit

```text
[ ] inspect repository
[ ] identify existing valid runs
[ ] identify exact current hyperparameters
[ ] document early-stopping logic
[ ] document generalization-gap definition
[ ] identify missing main seeds
[ ] identify Tiny ImageNet configuration
[ ] identify selected ViT
```

**Do not start long training before Stage 0 is complete.**

---

## Stage 1 — Instrumentation and smoke tests

Implement:

```text
[ ] unified metadata
[ ] timing
[ ] GPU memory logging
[ ] controller-overhead timing
[ ] config hashes
[ ] split hashes
[ ] initialization hashes
[ ] resume logic
```

Smoke test all new experiment paths.

---

## Stage 2 — Main 5-seed completion

Complete missing runs for:

```text
CIFAR-10 / CIFAR-100
ResNet-18 / ResNet-50 / VGG-16
5 methods
5 seeds
```

---

## Stage 3 — PD internal ablation

Run PD component ablation early.

Immediately generate an intermediate report.

This is scientifically critical because it determines whether the full PD construction is justified.

---

## Stage 4 — Fixed vs adaptive budget

Run fair matched-budget comparison.

Immediately generate performance–cost figures.

---

## Stage 5 — Sigma sensitivity

Run all 4 sigma levels × 5 seeds.

---

## Stage 6 — Tiny ImageNet

Run:

```text
5 methods × 5 seeds
```

after successful smoke test.

---

## Stage 7 — Selected ViT

Run:

```text
5 methods × 5 seeds
```

after successful smoke test.

---

## Stage 8 — Overhead and final timing verification

Complete controller overhead and GPU-memory analysis.

---

## Stage 9 — Statistics and figures

Generate:

```text
full baseline tables
paired comparisons
95% CI
effect sizes
Holm correction
seed stability summaries
performance-cost plots
external validation figures
```

---

## Stage 10 — Final revision report

Generate:

```text
PAPER1_REVISION_EXPERIMENT_REPORT.md
```

---

# 28. Final Report Structure

`PAPER1_REVISION_EXPERIMENT_REPORT.md` must contain:

```text
1. Revision objective
2. Repository audit
3. Data leakage protocol
4. Final experiment inventory
5. Reused vs newly executed runs
6. Main 5-seed benchmark results
7. Full five-method comparison
8. PD component ablation
9. Fixed-budget vs adaptive-budget
10. Performance–cost trade-off
11. Sigma sensitivity
12. Tiny ImageNet external validation
13. ViT external architecture validation
14. Computational overhead
15. Cross-seed reproducibility
16. Statistical analysis with Holm correction
17. Negative findings and limitations
18. Direct response to each supervisor experimental comment
19. Final conclusions supported by evidence
20. Exact list of manuscript tables/figures to update
```

---

# 29. Direct Mapping to Supervisor Comments

## PAC-Bayes proxy concern

Evidence:

```text
theoretical explanation in manuscript
sigma audit
sigma sensitivity
PB trajectory correlations
```

---

## Sigma concern

Evidence:

```text
exact sigma
selection history
4-level sensitivity × 5 seeds
```

---

## PD interpretability concern

Evidence:

```text
PD_FULL
V_ONLY
V+H
V+U
H+U
decision timing analysis
trajectory correlations
```

---

## Lambda_g / Lambda_u concern

Evidence:

```text
exact values
selection history
PD ablation
optional local sensitivity only if necessary
```

---

## Validation-loss-only challenge

Evidence:

```text
Validation-Loss-Only becomes a formal baseline
5 seeds
fixed-budget comparison
adaptive-budget comparison
performance–cost analysis
```

---

## Unequal training budget concern

Evidence:

```text
fixed-budget experiment
adaptive-budget experiment
same checkpoint-selection rule
```

---

## Seed concern

Evidence:

```text
all primary and external headline comparisons use 5 matched seeds
```

---

## Multiple comparisons

Evidence:

```text
Holm-adjusted p-values
effect sizes
95% CIs
consistency across settings
```

---

## Baseline-selection transparency

Evidence:

```text
complete five-method tables
metric-specific strongest-baseline summary remains secondary
```

---

## Reliability definition

Evidence:

```text
multi-dimensional operational definition
separate metric categories
AURC retained even when negative
```

---

## Reliability-score terminology

Use:

```text
reliability-oriented control score
```

or:

```text
reliability-control signal
```

unless the manuscript explicitly justifies another term.

---

## Early-stopping reproducibility

Evidence:

```text
exact Boolean rule
min-delta
counter reset logic
pseudocode
```

---

## Computational overhead

Evidence:

```text
seconds/epoch
wall-clock
GPU-hours
memory
controller overhead
```

---

## Experimental scope concern

Evidence:

```text
Tiny ImageNet + ResNet-18
5 methods × 5 seeds

CIFAR-100 + selected ViT
5 methods × 5 seeds
```

---

## Alpha concern

Evidence:

```text
existing alpha/beta sensitivity
historical selection rationale
no post-hoc retuning
```

---

## Generalization-gap definition

Evidence:

```text
explicit formula and orientation
code/manuscript consistency
```

---

# 30. Completion Criteria

The revision is complete only when:

```text
[ ] All original six dataset/model settings have five methods × 5 matched seeds.
[ ] Validation-Loss-Only is included as the fourth baseline.
[ ] Tiny ImageNet external validation has five methods × 5 matched seeds.
[ ] Selected ViT external validation has five methods × 5 matched seeds.
[ ] PD component ablation is complete.
[ ] Sigma sensitivity is complete.
[ ] Fixed-budget vs adaptive-budget comparison is complete.
[ ] Computational overhead is measured.
[ ] Complete five-method tables are exported.
[ ] All headline statistical tests use 5 seeds.
[ ] 95% CIs are generated.
[ ] Paired effect sizes are generated.
[ ] Holm correction is generated.
[ ] Early-stopping logic is unambiguous.
[ ] Generalization-gap definition is explicit.
[ ] sigma, lambda_g, and lambda_u are documented.
[ ] No test leakage is detected.
[ ] Negative results are retained.
[ ] Runtime report is generated from actual measured throughput.
[ ] Final revision report is generated.
```

---

# 31. Codex Must Calculate the Real GPU-Time Requirement

After Stage 0 and the smoke benchmarks, Codex must output:

```text
FINAL REQUIRED RUNS
VALID EXISTING RUNS
NEW RUNS REQUIRED

Measured seconds/epoch by model/dataset
Measured average stopping epoch by method
Estimated hours per run
Estimated GPU-hours by experiment group

TOTAL ESTIMATED NEW GPU-HOURS
TOTAL ESTIMATED CALENDAR DAYS @ 24 h/day
TOTAL ESTIMATED CALENDAR DAYS @ 20 h/day

Expected completion date/time
```

Do not use the previous hand estimate as the final runtime estimate.

The final estimate must come from the user's actual RTX 5070 Ti measurements.

---

# 32. Suggested Runtime Report Table

Codex should generate a table like:

| Experiment group | New runs | Mean h/run | Estimated GPU-h | Completed GPU-h | Remaining GPU-h |
|---|---:|---:|---:|---:|---:|
| Main matrix completion | ... | ... | ... | ... | ... |
| PD ablation | ... | ... | ... | ... | ... |
| Budget comparison | ... | ... | ... | ... | ... |
| Sigma sensitivity | ... | ... | ... | ... | ... |
| Tiny ImageNet | ... | ... | ... | ... | ... |
| Selected ViT | ... | ... | ... | ... | ... |
| Overhead verification | ... | ... | ... | ... | ... |
| **TOTAL** | ... | — | ... | ... | ... |

---

# 33. Manuscript-Level Outputs That Experiments Must Support

The final experimental package must allow the paper to make a careful conclusion such as:

> The proposed PB+PD controller should not be presented as universally superior on every endpoint metric. Its contribution must be evaluated as a multi-signal training-control mechanism that may provide improved reproducibility and a favourable reliability–computation trade-off under matched experimental conditions.

Do not write a stronger claim unless the new results support it.

---

# 34. Final Codex Instruction

**Work scientifically, not cosmetically.**

Codex must:

```text
reuse valid existing runs
run only genuinely missing experiments
preserve matched conditions
record failures
preserve negative results
avoid test leakage
avoid post-hoc parameter tuning
avoid silently changing the method
calculate runtime from measured throughput
generate all raw and aggregated evidence
```

The final revision must allow a reviewer to distinguish:

```text
1. whether PB adds useful generalization-oriented information;
2. whether H_t and U_t add value beyond V_t;
3. whether the controller improves decisions under equal budget;
4. whether earlier stopping provides meaningful computational savings;
5. whether the savings justify the controller overhead;
6. whether conclusions are stable across 5 seeds;
7. whether conclusions extend beyond CIFAR;
8. whether conclusions extend beyond CNN architectures.
```

---

# 35. Codex Kickoff Prompt

Paste this instruction to Codex after placing this Markdown file in the project root:

```text
Read this Markdown specification completely before modifying or running anything.

First perform Stage 0 only: audit the current repository, existing experiments, exact controller implementation, hyperparameters, dataset splits, seeds, completed results, Tiny ImageNet support, and selected ViT implementation.

Do not start long GPU experiments yet.

Create the requested audit files, determine which existing runs are reusable, calculate the exact missing-run matrix, implement short runtime benchmarks, and generate an initial REVISION_STATUS.md and RUNTIME_ESTIMATE.md.

Show me the audit results and unresolved issues before launching the full experiment queue.

Do not silently change any existing scientific setting.
Do not invent missing hyperparameter values.
Do not use test data for training decisions.
Do not rerun valid completed experiments unnecessarily.
```

---

# 36. Stage 0 Execution Record (2026-08-25)

Status: **Stage 0 complete; long-running revision queue not launched.**

## 36.1 Repository audit outcome

```text
Main matrix required slots: 150
Reusable historical main slots: 88
Missing main slots: 62

All currently specified revision groups: 295 slots
Reusable slots by analysis role: 114
New slots required: 181
Currently unblocked new slots: 131
Blocked external-validation slots: 50
```

The 62 missing main runs consist of:

```text
Validation-Loss-Only:
    6 dataset/model settings x 5 seeds = 30 runs

Missing seeds 4 and 5 for existing four methods:
    4 ResNet-50/VGG-16 dataset/model settings x 4 methods x 2 seeds = 32 runs
```

The generated unblocked run manifest contains:

```text
Main matrix completion: 62
PD component ablation: 22
Fixed/adaptive budget comparison: 30
Sigma sensitivity: 17
TOTAL CURRENTLY SCHEDULABLE: 131
```

Historical reuse limitations are recorded explicitly: old runs did not persist config, split, or initialization hashes or model checkpoints. Scientific config, split, and initialization hashes were reconstructed from the versioned code/config and recorded seeds for matched checks; historical model checkpoints remain unavailable.

## 36.2 Implementation findings

```text
sigma_0 = 0.05
lambda_g = 0.0001
lambda_u = 0.001
alpha = 0.5
beta = 0.9
```

The implemented reliability stagnation counter resets whenever it reaches `p_lr=8`; therefore it cannot reach `p_stop=18`. The current stopping rule is effectively driven by validation-NLL stagnation. This behaviour was documented and not silently corrected.

The CIFAR validation subset currently inherits the training dataset's random crop and horizontal-flip transform. This is not test leakage, but it makes validation/controller observations stochastic and requires an explicit comparability decision before the long queue.

Tiny ImageNet support and a selected ViT/DeiT architecture are absent. In accordance with the selection rules, no external recipe or substitute architecture was invented.

## 36.3 RTX 5070 Ti runtime benchmark

Each available family used one warm-up epoch followed by five timed full train+validation epochs:

| Family | Seconds/epoch | Controller seconds/epoch | Controller overhead | Peak reserved VRAM |
|---|---:|---:|---:|---:|
| CIFAR-10 + ResNet-18 | 72.92 | 9.17 | 12.57% | 2398 MB |
| CIFAR-100 + ResNet-50 | 90.09 | 26.32 | 29.21% | 6212 MB |
| CIFAR-100 + VGG-16 | 72.53 | 4.15 | 5.73% | 4404 MB |
| Tiny ImageNet + ResNet-18 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| CIFAR-100 + selected ViT | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |

The controller timing includes gradient-norm and update-norm monitoring during training, PB computation, online standardization/smoothing, and the controller decision. Excluding the per-batch gradient monitoring would materially understate the cost, especially for ResNet-50.

## 36.4 Updated completion estimate

Using current measured throughput and historical stopping epochs where applicable:

| Experiment group | New runs | Estimated remaining GPU-h |
|---|---:|---:|
| Main matrix completion | 62 | 161.7 |
| PD component ablation | 22 | 40.1 |
| Budget comparison | 30 | 69.9 |
| Sigma sensitivity | 17 | 31.0 |
| Tiny ImageNet | 25 | UNRESOLVED |
| Selected ViT | 25 | UNRESOLVED |
| **Known-family subtotal** | **131** | **302.6** |

```text
Known-family calendar time at 24 GPU-h/day: 12.61 days
Known-family calendar time at 20 GPU-h/day: 15.13 days
Earliest known-family completion at continuous execution: approximately 2026-09-07 10:12 +08:00
Known-family completion at 20 GPU-h/day: approximately 2026-09-09 22:44 +08:00
Completed Stage 0 benchmark consumption: approximately 0.80 GPU-h

Full completion datetime: UNRESOLVED
Reason: Tiny ImageNet and selected ViT recipes are absent and cannot be benchmarked.
```

Generated evidence and executable plans are under `revision_experiments/`. ETA values must be regenerated after the two external configurations are resolved and after each completed run.

---

# 37. Corrective Audit, External Assets, and Revised Runtime Record (2026-08-26)

Status: **corrective implementation and smoke verification complete; the long 260-run queue has not been launched.**

Section 36 is retained as the original Stage 0 snapshot, but its reuse counts, overhead measurements, and ETA are superseded by this section. The corrective audit found that the historical protocol used stochastic augmented validation observations and that the reliability-stagnation counter could not reach `p_stop=18` because it was reset at `p_lr=8`. These defects affect controller decisions and stopping trajectories, so none of the 150 historical main-matrix slots is reusable for protocol-v2 inference.

## 37.1 Corrected implementation and second logic audit

The following defects were fixed before any formal rerun:

1. CIFAR training and validation now use separate dataset objects over disjoint, seed-reproducible indices; validation has deterministic evaluation transforms.
2. Reliability LR patience and stopping patience use independent counters, making both LR reduction and reliability-driven early stopping reachable.
3. A full-model gradient norm is calculated once on the retained final minibatch instead of redundantly on every minibatch.
4. CUDA norm reductions remain on-device with explicit synchronization boundaries, so pending training work is not incorrectly attributed to controller overhead.
5. Completed-result reuse requires all nine output artifacts and an exact scientific-config hash; partial outputs or mismatches are archived and rerun.
6. Scheduled one-run configs retain a stable scientific experiment name, preserving method/seed grouping for paired statistics.
7. Validation-Loss-Only is included as the fourth baseline in all formal tables and paired tests.
8. Revision postprocessing implements five-seed paired bootstrap confidence intervals, paired effect size, raw tests, and Holm correction.
9. Queue status is persisted before and after every run, and failed runs remain visible unless explicitly handled.

The second code walk-through covered dataset class mapping and split isolation, controller state transitions, PB/PD norm computation and timing, checkpoint-before-test ordering, test evaluation count, result collision handling, physical-run sharing, and statistical pairing. No additional known core training/control logic gap remains. Scientific conclusions remain pending the formal runs.

Regression verification:

```text
python -m unittest tests.test_revision_protocol -v
Ran 12 tests: OK
python -m compileall: OK
git diff --check: OK (line-ending notices only)
```

## 37.2 Tiny ImageNet acquisition and protocol

```text
Archive: data/tiny-imagenet-200.zip
Size: 248,100,043 bytes
Verified MD5: 90528d7ca1a48142e341f4ef8d21d0de
Extracted classes: 200
Official train images: 100,000
Official labeled validation images: 10,000
Official unlabeled test images: 10,000 (unused)
```

Protocol v2 uses a seed-fixed 90,000/10,000 split of the official training images for optimization/internal validation. The 10,000 labeled official validation images are isolated for exactly one final test evaluation. The unlabeled official test set is not used. The fixed recipe is 64x64 CIFAR-style ResNet-18, ImageNet normalization, random crop with 8-pixel padding plus horizontal flip for training, deterministic validation/test transforms, SGD (`lr=0.1`, momentum `0.9`, weight decay `0.0005`), batch 128, and at most 120 epochs.

The required five-method, one-seed, five-epoch smoke test completed without NaN, loader/class errors, checkpoint errors, or incomplete artifacts. Every method produced all nine required result files.

| Tiny ImageNet smoke method | Epochs | Test accuracy |
|---|---:|---:|
| Step Decay | 5 | 0.3105 |
| Cosine Annealing | 5 | 0.3947 |
| ReduceLROnPlateau | 5 | 0.2712 |
| Validation-Loss-Only | 5 | 0.2734 |
| PB+PD Reliability | 5 | 0.2946 |

## 37.3 External architecture readiness

The previously requested external architecture is now fixed as DeiT-Tiny/16 with the official ImageNet-1k checkpoint. Only the 1,000-class head is discarded; the CIFAR-100 head is initialized per matched seed. The local checkpoint SHA-256 is `a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`. Its five-method, five-epoch smoke test also completed with all artifacts present; test accuracies were 0.7924, 0.8131, 0.7413, 0.7222, and 0.7311 for Step, Cosine, Plateau, Validation-Loss-Only, and PB+PD respectively.

## 37.4 Corrected physical-run inventory

```text
Final analysis inventory: 295 slots
Historical slots reusable under protocol v2: 0
Slots supplied to secondary analyses by corrected main runs: 35
Unique new physical runs: 260
Completed formal physical runs: 0
Remaining formal physical runs: 260
Failed formal runs: 0
Long queue launched: NO
```

The 35 shared slots are deliberate, documented reuse of corrected main runs: 5 PD-full slots, 25 adaptive-budget slots, and 5 default-sigma slots. They do not create duplicate GPU jobs.

## 37.5 RTX 5070 Ti measurements and revised completion time

Each family used one first-epoch warm-up plus five timed full training-and-validation epochs. Corrected controller timing is measured with explicit CUDA synchronization.

| Family | Timed seconds/epoch | Controller overhead | Peak reserved VRAM |
|---|---:|---:|---:|
| CIFAR-10 + ResNet-18 | 8.378 | 0.0975% | 2,398 MB |
| CIFAR-100 + ResNet-50 | 28.266 | 0.0668% | 6,320 MB |
| CIFAR-100 + VGG-16 | 4.960 | 0.1389% | 2,198 MB |
| Tiny ImageNet + ResNet-18 | 67.193 | 0.0115% | 4,164 MB |
| CIFAR-100 + DeiT-Tiny | 38.281 | 0.0468% | 4,196 MB |

| Experiment group | Physical runs | Conservative remaining GPU-h |
|---|---:|---:|
| Corrected main matrix | 150 | 71.7 |
| PD component ablation | 20 | 5.8 |
| Fixed-budget comparison | 25 | 7.3 |
| Sigma sensitivity | 15 | 4.4 |
| Tiny ImageNet + ResNet-18 | 25 | 56.4 |
| CIFAR-100 + DeiT-Tiny | 25 | 27.0 |
| **TOTAL** | **260** | **172.6** |

```text
Estimated remaining GPU time: 172.6 hours
Calendar time at 24 GPU-h/day: 7.19 days
Calendar time at 20 GPU-h/day: 8.63 days
Estimate generated: 2026-08-26 00:32:09 +08:00
Continuous-execution completion: 2026-09-02 05:08:15 +08:00
Completion at 20 GPU-h/day: 2026-09-03 15:39:28 +08:00
Completed benchmark consumption: approximately 1.21 GPU-h
```

This is a conservative maximum-epoch planning estimate: it includes measured first-epoch warm-up cost and assumes every run reaches its epoch ceiling. Corrected early-stopping trajectories do not yet exist, so no speculative stopping discount is applied. `revision_experiments/RUNTIME_ESTIMATE.md` must be regenerated after every completed formal run.

---

# 38. Formal Queue Launch and Live Monitoring Record (2026-08-26)

Status: **the corrected-protocol 260-run formal queue is active.** Section 37 remains the pre-launch audit and planning record; this section supersedes only its operational statement that the long queue had not yet started.

## 38.1 C5O2 resource handoff

Before starting this project, the C5O2 campaign was identified and stopped as a complete Windows process tree:

```text
Project: D:\chapter5-safe-sparsification
Campaign: 20260825T095903Z_resume
Campaign phase: confirmatory_sparse
Complete jobs preserved: 6
Interrupted job: cs9_cifar100_resnet18_cifar_safe_no_postft_check_s70_seed44
Interrupted phase: support_selection, attempt 2/7
Original campaign root PID: 17052
Pause method: terminate campaign process tree after recording command/state
```

C5O2 does not implement an application-level pause signal or epoch/attempt checkpoint for this job. Its completed jobs and disk artifacts remain intact, but the interrupted `s70_seed44` job must restart from the beginning when the campaign is resumed. The preserved resume command is:

```powershell
D:\chapter5-safe-sparsification\.venv\Scripts\python.exe scripts\run_experiment_campaign.py --campaign-id 20260825T095903Z_resume --skip-stage0 --skip-pilot --download --device cuda --batch-size 128 --num-workers 2 --wait-for-gpu-sec 7200
```

## 38.2 Paper 1 formal launch

```text
Launch time: 2026-08-26 00:49:04 +08:00
Queue command: revision_experiments/scripts/run_revision.py --execute --existing rerun-partial
Queue root PID at launch: 41900
Formal physical runs: 260
Execution order: main -> PD -> budget -> sigma -> Tiny ImageNet -> DeiT-Tiny
Progress resolution: every 20 batches or 5 seconds, plus every epoch/test transition
Queue stdout: revision_experiments/results/revision_queue.out.log
Queue stderr: revision_experiments/results/revision_queue.err.log
Current-run log: revision_experiments/results/current_run.log
Current-run JSON: revision_experiments/results/current_run_state.json
Manifest: revision_experiments/results/run_manifest.csv
```

The first formal run is `revision_main_cifar10_resnet18 / step / seed=1`. Live verification observed epoch 3/120, validation accuracy 0.5604, validation NLL 1.2345, GPU utilization approximately 92%, and no queue error.

## 38.3 Global visualization

```text
Dashboard URL: http://127.0.0.1:8000/
Refresh interval: 2 seconds
Server root PID at launch: 10628
Current server root PID after ETA-display restart: 5992
Listening interface: 127.0.0.1:8000 only
```

The visualization combines all live sources rather than displaying only the current single-run config. It exposes:

```text
260-run completed/running/pending/failed counts
main/PD/budget/sigma/Tiny ImageNet/DeiT group progress
current global run index, dataset, model, method, seed and output directory
epoch, batch, phase, elapsed time and current-run ETA
train loss/accuracy, validation loss/accuracy, best validation and LR
reliability signal when available
GPU utilization, memory, temperature and power
current accuracy/loss/reliability trajectories
filterable complete queue with per-run output paths
current-run log, queue stdout and queue stderr
remaining measured GPU-hours and continuously updated completion time
```

The system default browser was opened to the dashboard URL. The server and JSON API returned HTTP 200 before the GPU queue was launched, and batch-level values continued updating after launch.

The dashboard ETA subtracts the active run's elapsed GPU time between completed-run estimate regenerations, preventing the displayed completion timestamp from drifting forward during a long run. Restarting the dashboard to apply this display correction did not interrupt the training queue.

---

# 39. Formal Result Root Migration to `results_v2` (2026-08-26)

Status: **all formal protocol-v2 experiment results use `results_v2` as their authoritative root.**

The storage requirement was clarified after the formal queue started: new formal results must be placed together with the earlier experiment archive under the existing `results_v2` directory, rather than under `revision_experiments/results`.

## 39.1 Authoritative directory layout

```text
Formal per-run artifacts:
    results_v2/<experiment>/<method>/seed_<n>/

Formal experiment aggregates:
    results_v2/<experiment>/summary_by_seed.csv
    results_v2/master_summary.csv

Queue and visualization state:
    revision_experiments/results/run_manifest.csv
    revision_experiments/results/current_run_state.json
    revision_experiments/results/current_run.log
    revision_experiments/results/revision_queue.out.log
    revision_experiments/results/revision_queue.err.log
    revision_experiments/results/runtime_estimate.csv

Smoke and revision analysis outputs:
    revision_experiments/results/smoke/
    revision_experiments/results/statistics/
```

`revision_experiments/results` is therefore an orchestration/analysis root, not the authoritative formal per-run result root.

## 39.2 Live-queue transition

The first formal run had already started with the old output setting. It was allowed to complete, then its per-run directory and aggregate rows were verified and migrated into `results_v2`:

```text
Experiment: revision_main_cifar10_resnet18
Method: step
Seed: 1
Final directory: results_v2/revision_main_cifar10_resnet18/step/seed_1
Required artifacts: 9/9
Status: COMPLETED
Protocol: deterministic_validation_v2
Test accuracy: 0.9457
Scientific hash: b3e3abf6ceb86fb641864862dc5e0ba2fb618f942558096678c951a7a686df3d
Old per-run and aggregate copies: removed after verification
```

The second formal run and every subsequent run write directly to `results_v2`. This transition did not stop or restart the GPU queue.

## 39.3 Configuration and reader validation

```text
Formal config files checked: 15/15 use results_dir = results_v2
Planned physical runs checked: 260
Unique output paths: 260/260
Non-results_v2 formal output paths: 0
```

The config generator in `audit_repository.py` also emits `results_dir = results_v2`, so regenerating the audit/config matrix cannot silently revert the storage location. `analyze_revision.py` and `estimate_runtime.py` read protocol-v2 results from `results_v2` first and deduplicate by scientific configuration hash; the old revision result root remains a temporary fallback only for interruption-safe migration.

---

# 40. Experiment Source Documentation Record (2026-08-26)

Status: **the complete non-test Python experiment stack now has module-level and function/class-level documentation, with additional inline explanations around the scientific algorithm and model construction.**

This pass was intentionally documentation-only. It did not change optimizer parameters, model topology, dataset partitions, controller decisions, metric definitions, output paths, or the active formal queue.

## 40.1 Documentation coverage

The checked source scope comprises the primary runner, scientific modules, batch/orchestration tools, revision audit and analysis tools, runtime estimation, result migration, dashboard service, statistical plots/tables, and case-study plots/tables.

```text
Documented Python modules: 23/23
Documented functions and classes: 249/249
Missing module docstrings: 0
Missing function/class docstrings: 0
```

The main documented boundaries are:

- `run_experiment.py`: scientific hashes, existing-result policy, training/evaluation order, checkpoint selection, test isolation, progress reporting, and result aggregation;
- `src/signals.py`: initialization and epoch snapshots, online Welford standardization, PB/PD construction, component ablations, EMA smoothing, and CUDA timing boundaries;
- `src/controller.py`: separate learning-rate and stopping patience counters, improvement tolerance, minimum-learning-rate behavior, and fixed/adaptive budget semantics;
- `src/models.py`: CIFAR-style ResNet stems, VGG adaptive pooling/classifier construction, DeiT-Tiny creation, official checkpoint loading, classifier-head replacement, and compatibility checks;
- `src/data.py`: stochastic training augmentation, deterministic validation/test transforms, reproducible disjoint splits, Tiny ImageNet annotation parsing, and loader ordering;
- `src/metrics.py` and `src/stats_utils.py`: classification, calibration, selective prediction, confidence intervals, paired tests, multiplicity correction, and effect-size conventions.

## 40.2 Core signal-to-code correspondence

The comments in `src/signals.py` explicitly preserve the intended mathematical interpretation:

```text
Initialization complexity:
    C_t = ||theta_t - theta_0||^2 / (2 sigma^2)

PB signal:
    PB_t = validation_NLL_t
           + sqrt((C_t + log(2 sqrt(n) / delta)) / (2n))

Positive generalization-risk violation:
    V_t = max(0, validation_NLL_t - training_NLL_t)

Full PD signal:
    PD_t = V_t + lambda_g ||g_t||_2 + lambda_u ||theta_t - theta_epoch_start||_2

Standardized instantaneous reliability:
    r_t = alpha z(PB_t) + (1 - alpha) z(PD_t)

Smoothed reliability:
    rbar_t = beta rbar_(t-1) + (1 - beta) r_t
```

The source documentation also records that the gradient norm is sampled before the final mini-batch optimizer update, whereas the update norm is evaluated after the epoch against the epoch-start parameter snapshot. `theta_0` remains a separate, fixed initialization snapshot for the entire physical run.

## 40.3 Model and evaluation protocol comments

The model factory documentation explains why ResNet uses a 3x3 stride-1 stem without the ImageNet max-pool on small images, why VGG uses adaptive 1x1 pooling, and why the DeiT ImageNet-1K head is removed while all compatible backbone weights are loaded strictly from the locally hashed official checkpoint.

The main runner documentation marks the leakage-prevention boundary: each epoch performs training, validation, signal computation, scheduling/control, and validation-only checkpoint selection. The selected checkpoint is restored before the test loader is accessed, and the test set is evaluated exactly once per physical run.

## 40.4 Verification and live-run continuity

```text
Python compileall: PASS
Revision protocol tests: 12/12 PASS
git diff --check: PASS
Documentation coverage audit: PASS (0 missing docstrings)
Formal queue interruption caused by documentation work: NO
Formal result root after documentation work: results_v2
Dashboard health check: HTTP 200 at http://127.0.0.1:8000/
```

At the final documentation verification checkpoint, the 260-run queue had completed two physical runs and was training global run 3 (`revision_main_cifar10_resnet18 / step / seed=3`). The source edits were limited to comments and docstrings, so later processes loading the documented source execute the same scientific logic as the already-running process.
