[CmdletBinding()]
param(
    [string]$RepresentationDirectory = "artifacts\transfer\representations",
    [string]$Folds = "all",
    [string[]]$Arms = @("all"),
    [string]$Device = "auto",
    [switch]$Force,
    [switch]$VerifyWhenComplete
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$contract = Resolve-RequiredFile `
    -Path (Join-Path $RepresentationDirectory "contract.json") `
    -Label "Step 5 representation contract"
$representationRoot = Split-Path -Parent $contract
$armSelection = $Arms -join ","

$arguments = @(
    "-m",
    "preference_futures.representations",
    "run",
    "--representation-dir",
    $representationRoot,
    "--folds",
    $Folds,
    "--arms",
    $armSelection,
    "--device",
    $Device
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments

if ($VerifyWhenComplete) {
    Invoke-CheckedScript -ScriptPath "$PSScriptRoot\82-verify-frozen-representations.ps1" -Parameters @{
        RepresentationDirectory = $representationRoot
        Folds = $Folds
        Arms = $armSelection
    }
}
