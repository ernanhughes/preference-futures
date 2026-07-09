[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$parseFailures = @()
$scriptFiles = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "*.ps1" -File | Sort-Object Name

foreach ($scriptFile in $scriptFiles) {
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $scriptFile.FullName,
        [ref]$tokens,
        [ref]$parseErrors
    )

    foreach ($parseError in $parseErrors) {
        $parseFailures += [PSCustomObject]@{
            Script = $scriptFile.Name
            Line = $parseError.Extent.StartLineNumber
            Column = $parseError.Extent.StartColumnNumber
            Message = $parseError.Message
        }
    }
}

if ($parseFailures.Count -gt 0) {
    $parseFailures | Format-Table -AutoSize | Out-String | Write-Host
    throw ("PowerShell parsing failed for {0} error(s)." -f $parseFailures.Count)
}

Write-Host ("PowerShell parsing passed for {0} script(s)." -f $scriptFiles.Count) -ForegroundColor Green
