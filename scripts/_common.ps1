Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Get-RepositoryRoot {
    return $script:RepositoryRoot
}

function Get-ProjectPython {
    $venvPython = Join-Path $script:RepositoryRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "Python was not found. Run scripts\00-setup.ps1 after installing Python 3.11+."
    }
    return $pythonCommand.Source
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    Write-Host "> $FilePath $($ArgumentList -join ' ')" -ForegroundColor Cyan
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $exitCode, $FilePath)
    }
}

function Invoke-CheckedScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,

        [hashtable]$Parameters = @{}
    )

    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        throw ("Script does not exist: {0}" -f $ScriptPath)
    }

    Write-Host "> & $ScriptPath" -ForegroundColor Cyan
    try {
        & $ScriptPath @Parameters
        $succeeded = $?
    }
    catch {
        throw ("Script failed: {0}`n{1}" -f $ScriptPath, $_.Exception.Message)
    }

    if (-not $succeeded) {
        throw ("Script failed without a terminating error: {0}" -f $ScriptPath)
    }
}

function Resolve-RepositoryPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $Path))
}

function Resolve-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [string]$Label = "File"
    )

    $resolved = Resolve-RepositoryPath -Path $Path
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw ("{0} does not exist: {1}" -f $Label, $resolved)
    }
    return $resolved
}

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolved = Resolve-RepositoryPath -Path $Path
    New-Item -ItemType Directory -Path $resolved -Force | Out-Null
    return $resolved
}
