[CmdletBinding()]
param(
    [string]$RepresentationDirectory = "artifacts\transfer\representations",
    [string]$Folds = "all",
    [string[]]$Arms = @("all")
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$contract = Resolve-RequiredFile `
    -Path (Join-Path $RepresentationDirectory "contract.json") `
    -Label "Step 5 representation contract"
$representationRoot = Split-Path -Parent $contract
$armSelection = $Arms -join ","

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.representations",
    "verify",
    "--representation-dir",
    $representationRoot,
    "--folds",
    $Folds,
    "--arms",
    $armSelection
)
