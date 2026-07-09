[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CorporaDirectory,

    [Parameter(Mandatory = $true)]
    [string]$EpisodesPath,

    [string]$OutputDirectory = "artifacts\transfer\training",
    [string]$ModelId = "distilbert/distilbert-base-uncased",
    [string]$ModelRevision = "main",
    [int]$Seed = 17,
    [int]$MaximumSequenceLength = 256,
    [int]$BatchSize = 16,
    [int]$UpdateSteps = 600,
    [double]$LearningRate = 0.00002,
    [double]$WeightDecay = 0.01,
    [int]$WarmupSteps = 60,
    [double]$GradientClipNorm = 1.0,
    [int]$LogEverySteps = 25,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$corpora = Resolve-RequiredFile `
    -Path (Join-Path $CorporaDirectory "manifest.json") `
    -Label "Step 2 corpus manifest"
$resolvedCorpora = Split-Path -Parent $corpora
$episodes = Resolve-RequiredFile -Path $EpisodesPath -Label "Preference episodes"
$output = Ensure-Directory -Path $OutputDirectory
$learningRate = $LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture)
$weightDecay = $WeightDecay.ToString([Globalization.CultureInfo]::InvariantCulture)
$clipNorm = $GradientClipNorm.ToString([Globalization.CultureInfo]::InvariantCulture)

$arguments = @(
    "-m",
    "preference_futures.training",
    "prepare",
    "--corpora-dir",
    $resolvedCorpora,
    "--episodes",
    $episodes,
    "--output-dir",
    $output,
    "--model-id",
    $ModelId,
    "--model-revision",
    $ModelRevision,
    "--seed",
    [string]$Seed,
    "--max-length",
    [string]$MaximumSequenceLength,
    "--batch-size",
    [string]$BatchSize,
    "--update-steps",
    [string]$UpdateSteps,
    "--learning-rate",
    $learningRate,
    "--weight-decay",
    $weightDecay,
    "--warmup-steps",
    [string]$WarmupSteps,
    "--gradient-clip-norm",
    $clipNorm,
    "--log-every-steps",
    [string]$LogEverySteps
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments

Write-Host "Step 3 fixed-budget contract prepared." -ForegroundColor Green
Write-Host "  Contract: $(Join-Path $output 'contract.json')"
Write-Host "  Plan:     $(Join-Path $output 'training-plan.md')"
Write-Host "  Snapshot: $(Join-Path $output 'base-snapshot')"
