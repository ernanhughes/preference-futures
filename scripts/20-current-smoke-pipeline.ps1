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
    [switch]$SkipChecks
)

if (-not $SkipChecks) {
    & "$PSScriptRoot\01-check.ps1"
}

$inspectArguments = @{
    DatabasePath = $DatabasePath
}
if (-not [string]::IsNullOrWhiteSpace($Table)) {
    $inspectArguments.Table = $Table
}
& "$PSScriptRoot\10-newsedits-inspect.ps1" @inspectArguments

$smokeArguments = @{
    DatabasePath = $DatabasePath
    OutputDirectory = $OutputDirectory
    Sources = $Sources
    Seed = $Seed
    MaxArticles = $MaxArticles
    MaxExamples = $MaxExamples
}
if (-not [string]::IsNullOrWhiteSpace($Table)) {
    $smokeArguments.Table = $Table
}
& "$PSScriptRoot\11-newsedits-smoke.ps1" @smokeArguments

Write-Host "Current smoke pipeline completed successfully." -ForegroundColor Green
