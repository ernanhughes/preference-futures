[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipLint
)

. "$PSScriptRoot\_common.ps1"

$repositoryRoot = Get-RepositoryRoot
$python = Get-ProjectPython

Push-Location $repositoryRoot
try {
    if (-not $SkipTests) {
        Invoke-CheckedCommand -FilePath $python -ArgumentList @("-m", "pytest")
    }

    if (-not $SkipLint) {
        Invoke-CheckedCommand -FilePath $python -ArgumentList @("-m", "ruff", "check", ".")
    }

    Write-Host "Repository checks passed." -ForegroundColor Green
}
finally {
    Pop-Location
}
