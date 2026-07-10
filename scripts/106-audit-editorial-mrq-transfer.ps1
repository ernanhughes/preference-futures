[CmdletBinding()]
param(
    [string]$TransferDirectory = "artifacts\step8\editorial-mrq\future-transfer",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.specificity",
    "--transfer-dir",
    (Resolve-RepositoryPath -Path $TransferDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
