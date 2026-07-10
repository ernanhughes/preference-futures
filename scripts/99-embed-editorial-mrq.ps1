[CmdletBinding()]
param(
    [string]$EditorialDirectory = "artifacts\step8\editorial-mrq",
    [string]$Device = "cuda",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.cli",
    "embed",
    "--editorial-dir",
    (Resolve-RepositoryPath -Path $EditorialDirectory),
    "--device",
    $Device
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
