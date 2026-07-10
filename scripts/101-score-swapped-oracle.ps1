[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalPredictions,
    [Parameter(Mandatory = $true)]
    [string]$SwappedPredictions,
    [Parameter(Mandatory = $true)]
    [string]$AnswerKey,
    [string]$OutputDirectory = "artifacts\step8\oracle-swap-score",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.cli",
    "oracle-score-swap",
    "--original-predictions",
    (Resolve-RequiredFile -Path $OriginalPredictions -Label "Original oracle predictions"),
    "--swapped-predictions",
    (Resolve-RequiredFile -Path $SwappedPredictions -Label "Swapped oracle predictions"),
    "--answer-key",
    (Resolve-RequiredFile -Path $AnswerKey -Label "Oracle answer key"),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
