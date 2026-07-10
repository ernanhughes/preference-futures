[CmdletBinding()]
param(
    [string]$TransferDirectory = "artifacts\step8\editorial-mrq\future-transfer",
    [string]$Folds = "all",
    [string]$Arms = "all",
    [string]$Device = "cuda",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.transfer_cli",
    "run",
    "--transfer-dir",
    (Resolve-RepositoryPath -Path $TransferDirectory),
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
