[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatabasePath,

    [string]$Table = ""
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$database = Resolve-RequiredFile -Path $DatabasePath -Label "NewsEdits database"
$arguments = @("-m", "preference_futures.newsedits", "inspect", "--db", $database)

if (-not [string]::IsNullOrWhiteSpace($Table)) {
    $arguments += @("--table", $Table)
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
