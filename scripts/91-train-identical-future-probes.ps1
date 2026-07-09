[CmdletBinding()]
param(
    [string]$ProbeDirectory = "artifacts\transfer\probes",
    [string]$Folds = "all",
    [string[]]$Arms = @("all"),
    [string]$Device = "auto",
    [switch]$Force,
    [switch]$VerifyWhenComplete
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$contract = Resolve-RequiredFile `
    -Path (Join-Path $ProbeDirectory "contract.json") `
    -Label "Step 6 probe contract"
$probeRoot = Split-Path -Parent $contract
$armSelection = $Arms -join ","

$arguments = @(
    "-m",
    "preference_futures.probes",
    "run",
    "--probe-dir",
    $probeRoot,
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
    Invoke-CheckedScript -ScriptPath "$PSScriptRoot\92-verify-identical-future-probes.ps1" -Parameters @{
        ProbeDirectory = $probeRoot
        Folds = $Folds
        Arms = $armSelection
    }
}
