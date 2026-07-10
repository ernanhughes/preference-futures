[CmdletBinding()]
param(
    [string]$MatchedDirectory = "artifacts\step8\editorial-mrq\future-transfer\matched-controls",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.matched_cli",
    "aggregate",
    "--matched-dir",
    (Resolve-RepositoryPath -Path $MatchedDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
