[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$Device = "auto",
    [int]$SmokeSteps = 2,
    [switch]$Force
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
    "0",
    "--regimes",
    "all",
    "--device",
    $Device,
    "--smoke-steps",
    [string]$SmokeSteps
)
if ($Force) {
    $arguments += "--force"
}
Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.training",
    "verify",
    "--training-dir",
    $training,
    "--folds",
    "0",
    "--regimes",
    "all",
    "--smoke"
)

Write-Host "Step 3 six-regime smoke run passed." -ForegroundColor Green
Write-Host "  Verification: $(Join-Path $training 'training-verification-smoke.md')"
