[CmdletBinding()]
param(
    [string]$EditorialDirectory = "artifacts\step8\editorial-mrq",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.aggregate",
    "--editorial-dir",
    (Resolve-RepositoryPath -Path $EditorialDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
