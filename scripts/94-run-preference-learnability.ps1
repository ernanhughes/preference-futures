[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$OutputDirectory = "artifacts\postmortem\preference-learnability",
    [int]$Fold = 0,
    [string]$Budgets = "600,1200,2400,5000,10000",
    [string]$MemorizationSizes = "256,512",
    [int]$MemorizationSteps = 5000,
    [int]$TrainEvaluationSize = 2048,
    [string]$Device = "cuda",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.diagnostics.cli",
    "preference-learnability",
    "--training-dir",
    (Resolve-RepositoryPath -Path $TrainingDirectory),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory),
    "--fold",
    $Fold.ToString(),
    "--budgets",
    $Budgets,
    "--memorization-sizes",
    $MemorizationSizes,
    "--memorization-steps",
    $MemorizationSteps.ToString(),
    "--train-evaluation-size",
    $TrainEvaluationSize.ToString(),
    "--device",
    $Device
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
