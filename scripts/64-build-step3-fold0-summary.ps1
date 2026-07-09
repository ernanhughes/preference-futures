[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$OutputZip = "artifacts\transfer\step3-fold0-summary.zip"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $training = [System.IO.Path]::GetFullPath(
        (Join-Path $repoRoot $TrainingDirectory)
    )

    $outputZipPath = [System.IO.Path]::GetFullPath(
        (Join-Path $repoRoot $OutputZip)
    )

    $bundle = Join-Path $training "step3-fold0-summary"
    $foldDirectory = Join-Path $training "runs\fold-00"

    $regimes = @(
        "language_adaptation",
        "pair_exposure",
        "temporal_direction",
        "random_label",
        "shuffled_preference",
        "authentic_preference"
    )

    $requiredTopLevelFiles = @(
        "training-plan.md",
        "model-source.json",
        "contract.json",
        "training-verification-smoke.json",
        "training-verification-smoke.md",
        "training-verification-confirmatory.json",
        "training-verification-confirmatory.md",
        "runs\last-run-summary.json"
    )

    Write-Host "Checking required Step 3 files..." -ForegroundColor Cyan

    foreach ($relativePath in $requiredTopLevelFiles) {
        $path = Join-Path $training $relativePath

        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required file is missing: $path"
        }
    }

    if (-not (Test-Path -LiteralPath $foldDirectory -PathType Container)) {
        throw "Confirmatory fold directory is missing: $foldDirectory"
    }

    foreach ($regime in $regimes) {
        $regimeDirectory = Join-Path $foldDirectory $regime
        $runReport = Join-Path $regimeDirectory "run.json"
        $metrics = Join-Path $regimeDirectory "metrics.jsonl"

        if (-not (Test-Path -LiteralPath $runReport -PathType Leaf)) {
            throw "Missing confirmatory run report: $runReport"
        }

        if (-not (Test-Path -LiteralPath $metrics -PathType Leaf)) {
            throw "Missing confirmatory metrics file: $metrics"
        }

        $run = Get-Content -LiteralPath $runReport -Raw |
            ConvertFrom-Json

        if ($run.non_confirmatory -ne $false) {
            throw "$regime is not a confirmatory run."
        }

        if ($run.status -ne "complete") {
            throw "$regime run status is not complete: $($run.status)"
        }

        if ($run.optimisation.optimizer_steps_completed -ne 600) {
            throw (
                "$regime completed " +
                "$($run.optimisation.optimizer_steps_completed) updates instead of 600."
            )
        }
    }

    $verificationPath = Join-Path `
        $training `
        "training-verification-confirmatory.json"

    $verification = Get-Content -LiteralPath $verificationPath -Raw |
        ConvertFrom-Json

    if ($verification.passed -ne $true) {
        throw "Confirmatory fold-0 verification did not pass."
    }

    if ($verification.mode -ne "confirmatory") {
        throw "Unexpected verification mode: $($verification.mode)"
    }

    if ($verification.observed.expected_jobs -ne 6 -or
        $verification.observed.observed_jobs -ne 6) {
        throw "Confirmatory verification does not contain all six fold-0 jobs."
    }

    Write-Host "Creating clean summary directory..." -ForegroundColor Cyan

    Remove-Item `
        -LiteralPath $bundle `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue

    New-Item `
        -ItemType Directory `
        -Path $bundle `
        -Force |
        Out-Null

    foreach ($relativePath in $requiredTopLevelFiles) {
        $source = Join-Path $training $relativePath
        $destinationName = Split-Path $relativePath -Leaf
        $destination = Join-Path $bundle $destinationName

        Copy-Item `
            -LiteralPath $source `
            -Destination $destination `
            -Force
    }

    foreach ($regime in $regimes) {
        $regimeDirectory = Join-Path $foldDirectory $regime

        Copy-Item `
            -LiteralPath (Join-Path $regimeDirectory "run.json") `
            -Destination (Join-Path $bundle "$regime-run.json") `
            -Force

        Copy-Item `
            -LiteralPath (Join-Path $regimeDirectory "metrics.jsonl") `
            -Destination (Join-Path $bundle "$regime-metrics.jsonl") `
            -Force
    }

    $outputParent = Split-Path -Parent $outputZipPath

    if (-not (Test-Path -LiteralPath $outputParent)) {
        New-Item `
            -ItemType Directory `
            -Path $outputParent `
            -Force |
            Out-Null
    }

    Remove-Item `
        -LiteralPath $outputZipPath `
        -Force `
        -ErrorAction SilentlyContinue

    Write-Host "Compressing confirmatory fold-0 reports..." -ForegroundColor Cyan

    Compress-Archive `
        -Path (Join-Path $bundle "*") `
        -DestinationPath $outputZipPath `
        -CompressionLevel Optimal `
        -Force

    $zip = Get-Item -LiteralPath $outputZipPath
    $hash = Get-FileHash -LiteralPath $outputZipPath -Algorithm SHA256

    Write-Host ""
    Write-Host "Step 3 fold-0 summary created successfully." -ForegroundColor Green
    Write-Host "  ZIP:    $($zip.FullName)"
    Write-Host "  Size:   $([math]::Round($zip.Length / 1MB, 3)) MB"
    Write-Host "  SHA256: $($hash.Hash.ToLowerInvariant())"
    Write-Host ""
    Write-Host "This archive contains reports and metadata only." -ForegroundColor Green
}
finally {
    Pop-Location
}