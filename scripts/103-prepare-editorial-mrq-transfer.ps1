[CmdletBinding()]
param(
    [string]$EditorialDirectory = "artifacts\step8\editorial-mrq",
    [string]$OutputDirectory = "artifacts\step8\editorial-mrq\future-transfer",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.transfer_cli",
    "prepare",
    "--editorial-dir",
    (Resolve-RepositoryPath -Path $EditorialDirectory),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
