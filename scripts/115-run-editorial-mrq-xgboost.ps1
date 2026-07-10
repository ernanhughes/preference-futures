[CmdletBinding()]
param(
    [string]$ExperimentDirectory = "artifacts\step8\editorial-mrq\future-transfer\xgboost-combined",
    [string]$Folds = "all",
    [string]$Arms = "all",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.xgboost_combined",
    "run",
    "--experiment-dir",
    (Resolve-RepositoryPath -Path $ExperimentDirectory),
    "--folds",
    $Folds,
    "--arms",
    $Arms,
    "--device",
    $Device
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
