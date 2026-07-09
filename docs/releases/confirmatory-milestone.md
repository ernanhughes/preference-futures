# Freeze the Steps 1–6 Confirmatory Milestone

The completed experiment should be marked before any exploratory postmortem changes the analysis surface.

The milestone is named:

```text
preference-futures-v0.1-confirmatory-negative
```

This is a scientific boundary, not a claim that the project is finished forever. It means:

```text
Steps 1–6 are complete and verified
primary confirmatory hypothesis was not supported
all later analyses are exploratory unless replicated on fresh data
```

## What the workflow validates

The milestone command refuses to run unless it can find and validate:

- the Step 1 grouped-split manifest and passing verification;
- the Step 2 corpus manifest and passing verification;
- the Step 3 confirmatory contract and 60-job passing verification;
- the Step 4 frozen manifest with 70 eligible encoder entries;
- the Step 5 contract and 70-job passing representation verification;
- the Step 6 contract, 70-job passing probe verification, and complete summary;
- a clean tracked Git working tree, unless `-AllowDirty` is explicitly supplied.

## Run the freeze

From the repository root:

```powershell
.\scripts\93-freeze-confirmatory-milestone.ps1
```

The command writes a tracked result record:

```text
docs/results/step-06-confirmatory-result.json
docs/results/step-06-confirmatory-result.md
```

It also creates an untracked compact archive:

```text
artifacts/releases/preference-futures-v0.1-confirmatory-negative/
artifacts/releases/preference-futures-v0.1-confirmatory-negative.zip
artifacts/releases/preference-futures-v0.1-confirmatory-negative.zip.sha256
artifacts/releases/preference-futures-v0.1-confirmatory-negative.summary.json
```

## Compact evidence included

The ZIP includes:

- contracts, manifests, summaries, and verification reports for Steps 1–6;
- Step 3 run reports and metric trajectories;
- Step 5 run reports and label-free row identities;
- Step 6 probe weights, run reports, validation predictions, and test predictions;
- a file-by-file SHA-256 manifest;
- the generated confirmatory result record.

It deliberately excludes:

- original NewsEdits sentence text and source-task corpora;
- all trained Transformer checkpoints;
- the roughly 2.5–3 GB Step 5 representation matrices.

The excluded local artifacts remain linked through persisted paths and hashes.

## Review before committing

```powershell
Get-Content docs\results\step-06-confirmatory-result.md

git status --short
```

The expected tracked changes are only:

```text
docs/results/step-06-confirmatory-result.json
docs/results/step-06-confirmatory-result.md
```

The release archive remains under `artifacts/` and should stay out of Git.

## Commit the result record

```powershell
git add `
  docs\results\step-06-confirmatory-result.json `
  docs\results\step-06-confirmatory-result.md

git commit -m "Record Steps 1-6 confirmatory negative result"
git push
```

## Create the permanent Git mark

Create the tag only after the result record is committed on the branch intended to represent the milestone, normally `main`:

```powershell
git tag -a v0.1-confirmatory-negative `
  -m "Steps 1-6 confirmatory experiment complete: authentic preference did not improve future prediction"

git push origin v0.1-confirmatory-negative
```

The tag should point to the commit containing the generated Step 6 result record.

## Optional GitHub release

A GitHub release can later be created from `v0.1-confirmatory-negative` with these two attachments:

```text
preference-futures-v0.1-confirmatory-negative.zip
preference-futures-v0.1-confirmatory-negative.zip.sha256
```

Publishing the compact archive is useful but optional. The Git tag and tracked result record are the essential marks.

## Re-running

Do not use `-Force` casually. The archive is intended to be immutable.

Use `-Force` only when correcting a packaging error without changing any scientific result:

```powershell
.\scripts\93-freeze-confirmatory-milestone.ps1 -Force
```

Use `-AllowDirty` only when the dirty tracked files have been inspected and are unrelated to the evidence chain. A clean tree is strongly preferred.
