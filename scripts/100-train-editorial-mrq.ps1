[CmdletBinding()]
param(
    [string]$EditorialDirectory = "artifacts\step8\editorial-mrq",
    [string]$Folds = "all",
    [string]$Rankers = "all",
    [string]$TeacherPredictions = "",
    [string]$Device = "cuda",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.cli",
    "train",
    "--editorial-dir",
    (Resolve-RepositoryPath -Path $EditorialDirectory),
    "--folds",
    $Folds,
    "--rankers",
    $Rankers,
    "--device",
    $Device
)
if ($TeacherPredictions) {
    $arguments += @(
        "--teacher-predictions",
        (Resolve-RequiredFile -Path $TeacherPredictions -Label "Teacher predictions")
    )
}
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
