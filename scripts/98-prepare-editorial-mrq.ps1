[CmdletBinding()]
param(
    [string]$TrainingDirectory = "artifacts\transfer\training",
    [string]$OutputDirectory = "artifacts\step8\editorial-mrq",
    [string]$ModelId = "sentence-transformers/all-mpnet-base-v2",
    [string]$ModelRevision = "main",
    [int]$Seed = 17,
    [int]$MaximumLength = 384,
    [int]$EmbeddingBatchSize = 24,
    [int]$RankerBatchSize = 256,
    [int]$MaximumEpochs = 100,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0001,
    [int]$Patience = 12,
    [int]$HiddenSize = 256,
    [int]$BottleneckSize = 64,
    [double]$Dropout = 0.1,
    [double]$TeacherWeight = 0.25,
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.cli",
    "prepare",
    "--training-dir",
    (Resolve-RepositoryPath -Path $TrainingDirectory),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory),
    "--model-id",
    $ModelId,
    "--model-revision",
    $ModelRevision,
    "--seed",
    $Seed.ToString(),
    "--max-length",
    $MaximumLength.ToString(),
    "--embedding-batch-size",
    $EmbeddingBatchSize.ToString(),
    "--ranker-batch-size",
    $RankerBatchSize.ToString(),
    "--maximum-epochs",
    $MaximumEpochs.ToString(),
    "--learning-rate",
    $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--weight-decay",
    $WeightDecay.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--patience",
    $Patience.ToString(),
    "--hidden-size",
    $HiddenSize.ToString(),
    "--bottleneck-size",
    $BottleneckSize.ToString(),
    "--dropout",
    $Dropout.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--teacher-weight",
    $TeacherWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
