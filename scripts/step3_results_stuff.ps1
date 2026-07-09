$training = "artifacts\transfer\training"
$bundle = "artifacts\transfer\step3-smoke-summary"

Remove-Item $bundle -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $bundle | Out-Null

Copy-Item "$training\training-plan.md" $bundle
Copy-Item "$training\training-verification-smoke.md" $bundle
Copy-Item "$training\training-verification-smoke.json" $bundle
Copy-Item "$training\model-source.json" $bundle
Copy-Item "$training\contract.json" $bundle

Get-ChildItem "$training\smoke-runs\fold-00" -Directory | ForEach-Object {
    $regime = $_.Name

    Copy-Item `
        "$($_.FullName)\run.json" `
        "$bundle\$regime-run.json"

    Copy-Item `
        "$($_.FullName)\metrics.jsonl" `
        "$bundle\$regime-metrics.jsonl"
}

Compress-Archive `
    -Path "$bundle\*" `
    -DestinationPath "artifacts\transfer\step3-smoke-summary.zip" `
    -Force