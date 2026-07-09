[CmdletBinding()]
param(
    [string]$SplitsDirectory = "artifacts\transfer\splits",
    [string]$CorporaDirectory = "artifacts\transfer\corpora",
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$EncoderSelectionDirectory = "artifacts\transfer\encoder-selection",
    [string]$RepresentationDirectory = "artifacts\transfer\representations",
    [string]$ProbeDirectory = "artifacts\transfer\probes",
    [string]$ReleaseRoot = "artifacts\releases",
    [string]$RepositoryResultDirectory = "docs\results",
    [string]$MilestoneName = "preference-futures-v0.1-confirmatory-negative",
    [switch]$AllowDirty,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$repositoryRoot = Get-RepositoryRoot

$arguments = @(
    "-m",
    "preference_futures.milestone.cli",
    "--repository-root",
    $repositoryRoot,
    "--splits-dir",
    (Resolve-RepositoryPath -Path $SplitsDirectory),
    "--corpora-dir",
    (Resolve-RepositoryPath -Path $CorporaDirectory),
    "--training-dir",
    (Resolve-RepositoryPath -Path $TrainingDirectory),
    "--encoder-selection-dir",
    (Resolve-RepositoryPath -Path $EncoderSelectionDirectory),
    "--representation-dir",
    (Resolve-RepositoryPath -Path $RepresentationDirectory),
    "--probe-dir",
    (Resolve-RepositoryPath -Path $ProbeDirectory),
    "--release-root",
    (Resolve-RepositoryPath -Path $ReleaseRoot),
    "--repository-result-dir",
    (Resolve-RepositoryPath -Path $RepositoryResultDirectory),
    "--milestone-name",
    $MilestoneName
)
if ($AllowDirty) {
    $arguments += "--allow-dirty"
}
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
