[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [Parameter(Mandatory = $true)]
    [string]$SplitManifestPath,

    [string]$OutputDirectory = "artifacts\transfer\corpora",
    [int]$Seed = 17
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$episodes = Resolve-RequiredFile -Path $EpisodesPath -Label "Episodes JSONL"
$splitManifest = Resolve-RequiredFile -Path $SplitManifestPath -Label "Grouped split manifest"
$output = Ensure-Directory -Path $OutputDirectory

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.corpora",
    "--episodes",
    $episodes,
    "--split-manifest",
    $splitManifest,
    "--output-dir",
    $output,
    "--seed",
    [string]$Seed
)

Write-Host "Compute-matched training corpora written." -ForegroundColor Green
Write-Host "  Manifest: $(Join-Path $output 'corpus-manifest.json')"
Write-Host "  Summary:  $(Join-Path $output 'corpus-summary.md')"
