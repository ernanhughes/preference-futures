[CmdletBinding()]
param(
    [string]$ControlDirectory = "artifacts\step8\editorial-mrq\future-transfer\shuffled-mrq-control",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.shuffled_cli",
    "aggregate",
    "--control-dir",
    (Resolve-RepositoryPath -Path $ControlDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
