[CmdletBinding()]
param(
    [string]$MatchedDirectory = "artifacts\step8\editorial-mrq\future-transfer\matched-controls",
    [string]$Folds = "all",
    [string]$Arms = "all",
    [string]$Device = "cuda",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.matched_cli",
    "run",
    "--matched-dir",
    (Resolve-RepositoryPath -Path $MatchedDirectory),
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
