[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [string]$NumericFlagsPath = "",
    [string]$OutputDirectory = "artifacts\transfer\splits",
    [int]$Folds = 10,
    [int]$Seed = 17
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$episodes = Resolve-RequiredFile -Path $EpisodesPath -Label "Episodes JSONL"
$output = Ensure-Directory -Path $OutputDirectory

$arguments = @(
    "-m",
    "preference_futures.splits",
    "--episodes",
    $episodes,
    "--output-dir",
    $output,
    "--folds",
    [string]$Folds,
    "--seed",
    [string]$Seed
)

if (-not [string]::IsNullOrWhiteSpace($NumericFlagsPath)) {
    $numericFlags = Resolve-RequiredFile -Path $NumericFlagsPath -Label "Numeric flags JSONL"
    $arguments += @("--numeric-flags", $numericFlags)
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments

Write-Host "Grouped split artifacts written." -ForegroundColor Green
Write-Host "  Manifest: $(Join-Path $output 'manifest.json')"
Write-Host "  Summary:  $(Join-Path $output 'split-summary.md')"
