[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$OutputDirectory = "artifacts\transfer\encoder-selection"
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$training = Resolve-RequiredFile `
    -Path (Join-Path $TrainingDirectory "contract.json") `
    -Label "Step 3 training contract"
$trainingRoot = Split-Path -Parent $training
$output = Ensure-Directory -Path $OutputDirectory

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.selection",
    "--training-dir",
    $trainingRoot,
    "--output-dir",
    $output
)

Write-Host "Step 4 source-task diagnostics and encoder freeze passed." -ForegroundColor Green
Write-Host "  Summary:  $(Join-Path $output 'source-task-summary.json')"
Write-Host "  Manifest: $(Join-Path $output 'accepted-encoders.json')"
