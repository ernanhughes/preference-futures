[CmdletBinding()]
param(
    [string]$ExperimentDirectory = "artifacts\step8\editorial-mrq\future-transfer\xgboost-combined",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.xgboost_combined",
    "aggregate",
    "--experiment-dir",
    (Resolve-RepositoryPath -Path $ExperimentDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
