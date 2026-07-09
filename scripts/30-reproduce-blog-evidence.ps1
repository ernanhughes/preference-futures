[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatabasePath,

    [string]$OutputDirectory = "artifacts\newsedits\blog-evidence",
    [int]$Seed = 17,
    [int]$MaxArticles = 5000,
    [int]$MaxExamples = 50000,
    [switch]$SkipChecks
)

. "$PSScriptRoot\_common.ps1"

$output = Ensure-Directory -Path $OutputDirectory
$episodesPath = Join-Path $output "episodes.jsonl"
$auditPath = Join-Path $output "audit.json"

if (-not $SkipChecks) {
    Invoke-CheckedScript -ScriptPath "$PSScriptRoot\01-check.ps1"
}

Invoke-CheckedScript `
    -ScriptPath "$PSScriptRoot\12-newsedits-full.ps1" `
    -Parameters @{
        DatabasePath = $DatabasePath
        OutputDirectory = $output
        Seed = $Seed
        MaxArticles = $MaxArticles
        MaxExamples = $MaxExamples
    }

Invoke-CheckedScript `
    -ScriptPath "$PSScriptRoot\14-context-viability-audit.ps1" `
    -Parameters @{
        EpisodesPath = $episodesPath
        OutputDirectory = $output
    }

Invoke-CheckedScript `
    -ScriptPath "$PSScriptRoot\15-numeric-shortcut-audit.ps1" `
    -Parameters @{
        EpisodesPath = $episodesPath
        OutputDirectory = $output
    }

Write-Host "Blog evidence reproduction completed." -ForegroundColor Green
Write-Host "  Extraction audit:  $auditPath"
Write-Host "  Context audit:     $(Join-Path $output 'context-viability.json')"
Write-Host "  Numeric audit:     $(Join-Path $output 'numeric-shortcut.json')"
Write-Host "  Numeric flags:     $(Join-Path $output 'numeric-flags.jsonl')"
Write-Host "  Episode artifact:  $episodesPath"
