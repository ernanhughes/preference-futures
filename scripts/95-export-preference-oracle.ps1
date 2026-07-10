[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$OutputDirectory = "artifacts\postmortem\preference-oracle",
    [int]$Fold = 0,
    [int]$SampleSize = 300,
    [int]$Seed = 17,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.diagnostics.cli",
    "oracle-export",
    "--training-dir",
    (Resolve-RepositoryPath -Path $TrainingDirectory),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory),
    "--fold",
    $Fold.ToString(),
    "--sample-size",
    $SampleSize.ToString(),
    "--seed",
    $Seed.ToString()
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
