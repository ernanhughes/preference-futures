[CmdletBinding()]
param(
    [string]$ProbeDirectory = "artifacts\transfer\probes",
    [string]$Folds = "all",
    [string[]]$Arms = @("all")
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$contract = Resolve-RequiredFile `
    -Path (Join-Path $ProbeDirectory "contract.json") `
    -Label "Step 6 probe contract"
$probeRoot = Split-Path -Parent $contract
$armSelection = $Arms -join ","

Invoke-CheckedCommand -FilePath $python -ArgumentList @(
    "-m",
    "preference_futures.probes",
    "verify",
    "--probe-dir",
    $probeRoot,
    "--folds",
    $Folds,
    "--arms",
    $armSelection
)
