[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [string]$OutputDirectory = "artifacts\newsedits\context-viability"
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$episodes = Resolve-RequiredFile -Path $EpisodesPath -Label "Episodes JSONL"
$output = Ensure-Directory -Path $OutputDirectory
$jsonPath = Join-Path $output "context-viability.json"
$markdownPath = Join-Path $output "context-viability.md"

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.audit",
    "--episodes",
    $episodes,
    "--json-out",
    $jsonPath,
    "--markdown-out",
    $markdownPath
)

Write-Host "Context viability artifacts written." -ForegroundColor Green
Write-Host "  JSON:     $jsonPath"
Write-Host "  Markdown: $markdownPath"
