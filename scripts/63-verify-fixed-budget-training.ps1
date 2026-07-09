[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$Folds = "all",
    [string]$Regimes = "all"
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$contract = Resolve-RequiredFile `
    -Path (Join-Path $TrainingDirectory "contract.json") `
    -Label "Step 3 training contract"
$training = Split-Path -Parent $contract

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

Write-Host "Step 3 confirmatory training verification passed." -ForegroundColor Green
Write-Host "  JSON:     $(Join-Path $training 'training-verification-confirmatory.json')"
Write-Host "  Markdown: $(Join-Path $training 'training-verification-confirmatory.md')"
