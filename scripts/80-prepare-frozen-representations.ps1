[CmdletBinding()]
param(
    [string]$SelectionManifest = "artifacts\transfer\encoder-selection\accepted-encoders.json",
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$OutputDirectory = "artifacts\transfer\representations",
    [int]$BatchSize = 32,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$selection = Resolve-RequiredFile -Path $SelectionManifest -Label "Step 4 encoder manifest"
$trainingContract = Resolve-RequiredFile `
    -Path (Join-Path $TrainingDirectory "contract.json") `
    -Label "Step 3 training contract"
$trainingRoot = Split-Path -Parent $trainingContract
$output = Resolve-RepositoryPath -Path $OutputDirectory

$arguments = @(
    "-m",
    "preference_futures.representations",
    "prepare",
    "--selection-manifest",
    $selection,
    "--training-dir",
    $trainingRoot,
    "--output-dir",
    $output,
    "--batch-size",
    $BatchSize.ToString()
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
