[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatabasePath,

    [string]$OutputDirectory = "artifacts\newsedits\smoke",
    [string]$Table = "",
    [string]$Sources = "",
    [int]$Seed = 17,
    [int]$MaxArticles = 100,
    [int]$MaxExamples = 1000,
    [int]$ContextBefore = 1,
    [int]$ContextAfter = 1,
    [int]$MinSentenceChars = 20,
    [int]$MaxSentenceChars = 500,
    [double]$MinEditSimilarity = 0.15,
    [double]$MaxEditSimilarity = 0.98
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$database = Resolve-RequiredFile -Path $DatabasePath -Label "NewsEdits database"
$output = Ensure-Directory -Path $OutputDirectory
$episodesPath = Join-Path $output "episodes.jsonl"
$auditPath = Join-Path $output "audit.json"

$arguments = @(
    "-m", "preference_futures.newsedits", "extract",
    "--db", $database,
    "--out", $episodesPath,
    "--audit-out", $auditPath,
    "--seed", [string]$Seed,
    "--max-articles", [string]$MaxArticles,
    "--max-examples", [string]$MaxExamples,
    "--context-before", [string]$ContextBefore,
    "--context-after", [string]$ContextAfter,
    "--min-sentence-chars", [string]$MinSentenceChars,
    "--max-sentence-chars", [string]$MaxSentenceChars,
    "--min-edit-similarity", [string]$MinEditSimilarity,
    "--max-edit-similarity", [string]$MaxEditSimilarity
)

if (-not [string]::IsNullOrWhiteSpace($Table)) {
    $arguments += @("--table", $Table)
}
if (-not [string]::IsNullOrWhiteSpace($Sources)) {
    $arguments += @("--sources", $Sources)
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
Invoke-CheckedScript `
    -ScriptPath "$PSScriptRoot\13-newsedits-verify.ps1" `
    -Parameters @{
        EpisodesPath = $episodesPath
        AuditPath = $auditPath
    }

Write-Host "Smoke extraction complete: $output" -ForegroundColor Green
