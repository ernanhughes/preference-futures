[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [string]$OutputDirectory = ""
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$manifest = Resolve-RequiredFile -Path $ManifestPath -Label "Grouped split manifest"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $output = Split-Path -Parent $manifest
}
else {
    $output = Ensure-Directory -Path $OutputDirectory
}

$jsonPath = Join-Path $output "split-verification.json"
$markdownPath = Join-Path $output "split-verification.md"

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.splits.verify_cli",
    "--manifest",
    $manifest,
    "--json-out",
    $jsonPath,
    "--markdown-out",
    $markdownPath
)

Write-Host "Grouped split manifest independently verified." -ForegroundColor Green
Write-Host "  JSON:     $jsonPath"
Write-Host "  Markdown: $markdownPath"
