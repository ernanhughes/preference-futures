[CmdletBinding()]
param(
    [string]$TransferDirectory = "artifacts\step8\editorial-mrq\future-transfer",
    [string]$OutputDirectory = "artifacts\step8\editorial-mrq\future-transfer\matched-controls",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.matched_cli",
    "prepare",
    "--transfer-dir",
    (Resolve-RepositoryPath -Path $TransferDirectory),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
