[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Database,
    [string]$OutputDirectory = "artifacts\postmortem\future-label-integrity",
    [string]$Table = "",
    [string]$SplitTable = "",
    [string]$SourceName = "",
    [int]$Seed = 0,
    [int]$MaxArticles = 0,
    [int]$MaxExamples = 0,
    [string]$Sources = "",
    [int]$SampleLimit = 50,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$databasePath = Resolve-RequiredFile -Path $Database -Label "NewsEdits database"
$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.diagnostics.cli",
    "future-label-audit",
    "--db",
    $databasePath,
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory),
    "--seed",
    $Seed.ToString(),
    "--max-articles",
    $MaxArticles.ToString(),
    "--max-examples",
    $MaxExamples.ToString(),
    "--sources",
    $Sources,
    "--sample-limit",
    $SampleLimit.ToString()
)
if ($Table) {
    $arguments += @("--table", $Table)
}
if ($SplitTable) {
    $arguments += @("--split-table", $SplitTable)
}
if ($SourceName) {
    $arguments += @("--source-name", $SourceName)
}
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
