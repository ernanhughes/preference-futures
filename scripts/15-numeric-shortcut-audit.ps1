[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [string]$OutputDirectory = "artifacts\newsedits\numeric-shortcut"
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$episodes = Resolve-RequiredFile -Path $EpisodesPath -Label "Episodes JSONL"
$output = Ensure-Directory -Path $OutputDirectory
$jsonPath = Join-Path $output "numeric-shortcut.json"
$markdownPath = Join-Path $output "numeric-shortcut.md"
$flagsPath = Join-Path $output "numeric-flags.jsonl"

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.audit.numeric_cli",
    "--episodes",
    $episodes,
    "--json-out",
    $jsonPath,
    "--markdown-out",
    $markdownPath,
    "--flags-out",
    $flagsPath
)

Write-Host "Numeric shortcut artifacts written." -ForegroundColor Green
Write-Host "  JSON:     $jsonPath"
Write-Host "  Markdown: $markdownPath"
Write-Host "  Flags:    $flagsPath"
