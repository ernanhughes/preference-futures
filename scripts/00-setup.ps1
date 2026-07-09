[CmdletBinding()]
param(
    [switch]$Recreate
)

. "$PSScriptRoot\_common.ps1"

$repositoryRoot = Get-RepositoryRoot
$venvPath = Join-Path $repositoryRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

Push-Location $repositoryRoot
try {
    if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
        Write-Host "Removing existing virtual environment: $venvPath" -ForegroundColor Yellow
        Remove-Item -LiteralPath $venvPath -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        $bootstrapPython = Get-ProjectPython
        Invoke-CheckedCommand -FilePath $bootstrapPython -ArgumentList @("-m", "venv", $venvPath)
    }

    Invoke-CheckedCommand -FilePath $venvPython -ArgumentList @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-CheckedCommand -FilePath $venvPython -ArgumentList @("-m", "pip", "install", "-e", ".[dev]")

    Write-Host "Setup complete. Python: $venvPython" -ForegroundColor Green
}
finally {
    Pop-Location
}
