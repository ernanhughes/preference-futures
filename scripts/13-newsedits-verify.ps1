[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [Parameter(Mandatory = $true)]
    [string]$AuditPath
)

. "$PSScriptRoot\_common.ps1"

$episodesFile = Resolve-RequiredFile -Path $EpisodesPath -Label "Episodes JSONL"
$auditFile = Resolve-RequiredFile -Path $AuditPath -Label "Audit JSON"

$count = 0
$stable = 0
$revised = 0
$lineages = [System.Collections.Generic.HashSet[string]]::new()

foreach ($line in [System.IO.File]::ReadLines($episodesFile)) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }

    $record = $line | ConvertFrom-Json
    if ($record.schema_version -ne 1) {
        throw "Unsupported episode schema version: $($record.schema_version)"
    }
    if ($record.newsedits_schema_version -ne 1) {
        throw "Unsupported NewsEdits schema version: $($record.newsedits_schema_version)"
    }
    if ($record.selected_index -notin @(0, 1)) {
        throw "Invalid selected_index in episode $($record.episode_id): $($record.selected_index)"
    }
    if ([string]::IsNullOrWhiteSpace([string]$record.candidate_a) -or
        [string]::IsNullOrWhiteSpace([string]$record.candidate_b)) {
        throw "Blank candidate text in episode $($record.episode_id)"
    }

    $count += 1
    [void]$lineages.Add([string]$record.lineage_id)
    if ([bool]$record.future_revised) {
        $revised += 1
    }
    else {
        $stable += 1
    }
}

if ($count -eq 0) {
    throw "No episodes were found in $episodesFile"
}

$audit = Get-Content -LiteralPath $auditFile -Raw | ConvertFrom-Json
if ([int]$audit.accepted_examples -ne $count) {
    throw "Audit accepted_examples=$($audit.accepted_examples) but JSONL contains $count records."
}

Write-Host "NewsEdits artifacts verified." -ForegroundColor Green
Write-Host "  Episodes: $count"
Write-Host "  Lineages: $($lineages.Count)"
Write-Host "  Revised:  $revised"
Write-Host "  Stable:   $stable"
Write-Host "  Audit:    $auditFile"
