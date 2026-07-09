cd C:\Projects\preference-futures

$completed = Get-ChildItem `
    artifacts\transfer\training\runs `
    -Recurse `
    -Filter run.json |
    ForEach-Object {
        Get-Content $_.FullName -Raw | ConvertFrom-Json
    } |
    Where-Object {
        $_.status -eq "complete" -and $_.non_confirmatory -eq $false
    }

"Completed confirmatory jobs: $($completed.Count) / 60"
"Remaining jobs: $(60 - $completed.Count)"