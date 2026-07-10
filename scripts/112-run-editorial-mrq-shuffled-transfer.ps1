[CmdletBinding()]
param(
    [string]$ControlDirectory = "artifacts\step8\editorial-mrq\future-transfer\shuffled-mrq-control",
    [string]$Replicas = "all",
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
    "preference_futures.editorial_mrq.shuffled_cli",
    "run-transfer",
    "--control-dir",
    (Resolve-RepositoryPath -Path $ControlDirectory),
    "--replicas",
    $Replicas,
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
