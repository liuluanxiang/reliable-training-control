$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$RunTag = "main_supplement_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$ResultsDir = "results\$RunTag"
$ProgressLog = "results\progress_$RunTag.log"
$StatePath = "results\progress_state.json"
$DashboardPath = "results\progress_dashboard.html"
$TablesDir = "tables\$RunTag"
$FiguresDir = "figures\$RunTag"
$ConclusionTablesDir = "tables_conclusion\$RunTag"
$ConclusionFiguresDir = "figures_conclusion\$RunTag"
$ExperimentSuffix = "__$RunTag"

Write-Host "Run tag: $RunTag"
Write-Host "Results dir: $ResultsDir"
Write-Host "Tables dir: $TablesDir"
Write-Host "Figures dir: $FiguresDir"

& $Python run_experiment.py --config configs\cifar10_resnet18_main.json --results-dir $ResultsDir --experiment-suffix $ExperimentSuffix --existing error --progress batch --progress-log $ProgressLog --progress-state $StatePath --dashboard-path $DashboardPath
& $Python run_experiment.py --config configs\cifar100_resnet18_main.json --results-dir $ResultsDir --experiment-suffix $ExperimentSuffix --existing error --progress batch --progress-log $ProgressLog --progress-state $StatePath --dashboard-path $DashboardPath
& $Python run_experiment.py --config configs\cifar10_resnet50_supplement.json --results-dir $ResultsDir --experiment-suffix $ExperimentSuffix --existing error --progress batch --progress-log $ProgressLog --progress-state $StatePath --dashboard-path $DashboardPath
& $Python run_experiment.py --config configs\cifar100_resnet50_supplement.json --results-dir $ResultsDir --experiment-suffix $ExperimentSuffix --existing error --progress batch --progress-log $ProgressLog --progress-state $StatePath --dashboard-path $DashboardPath
& $Python run_experiment.py --config configs\cifar10_vgg16_supplement.json --results-dir $ResultsDir --experiment-suffix $ExperimentSuffix --existing error --progress batch --progress-log $ProgressLog --progress-state $StatePath --dashboard-path $DashboardPath
& $Python run_experiment.py --config configs\cifar100_vgg16_supplement.json --results-dir $ResultsDir --experiment-suffix $ExperimentSuffix --existing error --progress batch --progress-log $ProgressLog --progress-state $StatePath --dashboard-path $DashboardPath

& $Python generate_case_study_tables.py --results-dir $ResultsDir --out-dir $TablesDir
& $Python make_case_study_figures.py --results-dir $ResultsDir --out-dir $FiguresDir

if (Test-Path generate_conclusion_tables.py) {
    & $Python generate_conclusion_tables.py --results-dir $ResultsDir --out-dir $ConclusionTablesDir
}

if (Test-Path make_conclusion_figures.py) {
    & $Python make_conclusion_figures.py --results-dir $ResultsDir --out-dir $ConclusionFiguresDir
}
