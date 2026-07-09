[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatabasePath,

    [string]$Table = "",
    [string]$SplitTable = ""
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$database = Resolve-RequiredFile -Path $DatabasePath -Label "NewsEdits database"
$arguments = @("-m", "preference_futures.newsedits", "inspect", "--db", $database)

if (-not [string]::IsNullOrWhiteSpace($Table)) {
    $arguments += @("--table", $Table)
}
if (-not [string]::IsNullOrWhiteSpace($SplitTable)) {
    $arguments += @("--split-table", $SplitTable)
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
