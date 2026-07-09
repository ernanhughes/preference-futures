[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$Folds = "all",
    [string]$Regimes = "all",
    [string]$Device = "auto",
    [switch]$Force,
    [switch]$VerifyWhenComplete
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$contract = Resolve-RequiredFile `
    -Path (Join-Path $TrainingDirectory "contract.json") `
    -Label "Step 3 training contract"
$training = Split-Path -Parent $contract

$arguments = @(
    "-m",
    "preference_futures.training",
    "run",
    "--training-dir",
    $training,
    "--folds",
    $Folds,
    "--regimes",
    $Regimes,
    "--device",
    $Device
)
if ($Force) {
    $arguments += "--force"
}
Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments

if ($VerifyWhenComplete) {
    Invoke-CheckedCommand -FilePath $python -ArgumentList @(
        "-m",
        "preference_futures.training",
        "verify",
        "--training-dir",
        $training,
        "--folds",
        $Folds,
        "--regimes",
        $Regimes
    )
}

Write-Host "Step 3 fixed-budget training command complete." -ForegroundColor Green
Write-Host "  Runs: $(Join-Path $training 'runs')"
