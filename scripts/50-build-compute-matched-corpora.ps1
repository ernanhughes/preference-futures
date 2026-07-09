[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatabasePath,

    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [Parameter(Mandatory = $true)]
    [string]$SplitsDirectory,

    [string]$OutputDirectory = "artifacts\transfer\corpora",
    [string]$SourceName = "nyt",
    [int]$Seed = 17,
    [int]$TemporalMaxArticles = 20000,
    [double]$TemporalPoolMultiplier = 2.0
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$database = Resolve-RequiredFile -Path $DatabasePath -Label "NewsEdits database"
$episodes = Resolve-RequiredFile -Path $EpisodesPath -Label "Episodes JSONL"
$splits = Resolve-RequiredFile `
    -Path (Join-Path $SplitsDirectory "manifest.json") `
    -Label "Grouped split manifest"
$resolvedSplitsDirectory = Split-Path -Parent $splits
$output = Ensure-Directory -Path $OutputDirectory
$multiplier = $TemporalPoolMultiplier.ToString(
    [Globalization.CultureInfo]::InvariantCulture
)

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.corpora",
    "--database",
    $database,
    "--episodes",
    $episodes,
    "--splits-dir",
    $resolvedSplitsDirectory,
    "--output-dir",
    $output,
    "--source-name",
    $SourceName,
    "--seed",
    [string]$Seed,
    "--temporal-max-articles",
    [string]$TemporalMaxArticles,
    "--temporal-pool-multiplier",
    $multiplier
)

Invoke-CheckedScript `
    -ScriptPath "$PSScriptRoot\51-verify-compute-matched-corpora.ps1" `
    -Parameters @{ OutputDirectory = $output }

Write-Host "Step 2 compute-matched corpora built and verified." -ForegroundColor Green
Write-Host "  Manifest:     $(Join-Path $output 'manifest.json')"
Write-Host "  Summary:      $(Join-Path $output 'corpus-summary.md')"
Write-Host "  Verification: $(Join-Path $output 'corpus-verification.md')"
