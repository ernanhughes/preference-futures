[CmdletBinding()]
param(
    [string]$Prompts = "artifacts\postmortem\preference-oracle\oracle-prompts.jsonl",
    [string]$OutputDirectory = "artifacts\step8\oracle-swap",
    [switch]$Force
)

. "$PSScriptRoot\_common.ps1"

$python = Get-ProjectPython
$arguments = @(
    "-m",
    "preference_futures.editorial_mrq.cli",
    "oracle-swap",
    "--prompts",
    (Resolve-RequiredFile -Path $Prompts -Label "Oracle prompts"),
    "--output-dir",
    (Resolve-RepositoryPath -Path $OutputDirectory)
)
if ($Force) {
    $arguments += "--force"
}

Invoke-CheckedCommand -FilePath $python -ArgumentList $arguments
