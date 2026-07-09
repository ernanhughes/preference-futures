[CmdletBinding()]
param(
    [string]$RepresentationDirectory = "artifacts\transfer\representations",
    [string]$OutputDirectory = "artifacts\transfer\probes",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$verification = Resolve-RequiredFile `
    -Path (Join-Path $RepresentationDirectory "representation-verification.json") `
    -Label "Step 5 representation verification"
$representationRoot = Split-Path -Parent $verification
$output = Resolve-RepositoryPath -Path $OutputDirectory

$arguments = @(
    "-m",
    "preference_futures.probes",
    "prepare",
    "--representation-dir",
    $representationRoot,
    "--output-dir",
    $output
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
