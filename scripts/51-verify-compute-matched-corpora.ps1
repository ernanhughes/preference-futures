[CmdletBinding()]
param(
    [string]$OutputDirectory = "artifacts\transfer\corpora"
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$output = Resolve-Path -LiteralPath $OutputDirectory -ErrorAction Stop

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.corpora.verify_cli",
    "--output-dir",
    $output.Path
)

Write-Host "Step 2 corpus artifacts independently verified." -ForegroundColor Green
Write-Host "  JSON:     $(Join-Path $output.Path 'corpus-verification.json')"
Write-Host "  Markdown: $(Join-Path $output.Path 'corpus-verification.md')"
