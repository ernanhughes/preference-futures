# Step 7 — Diagnostic Gates Before New Architecture

Steps 1–6 are frozen at `v0.1-confirmatory-negative`. Step 7 is explicitly exploratory.
It does not replace or reinterpret the confirmatory result. It localizes the failure before
any hard-relationship or multi-head experiment is designed.

## Gate A — Can authentic preference be learned?

The confirmatory authentic-preference source task achieved 50.257% validation accuracy and
was null-like in all ten folds. The next experiment distinguishes three possibilities:

1. the training implementation cannot fit the task;
2. 600 updates were insufficient;
3. the local pair-and-context target can be memorized but does not generalize.

### Design

One fixed exploratory fold is used. Every condition restarts from the immutable Step 3 base
snapshot.

Tiny-set memorization:

```text
256 records × 5,000 updates
512 records × 5,000 updates
```

Extended-budget learning curve:

```text
600
1,200
2,400
5,000
10,000 updates
```

The optimizer, learning rate, batch size, maximum sequence length, weight decay and gradient
clipping are inherited from the frozen Step 3 contract. No future label is available through
the source-store allow list.

The audit also reports four fixed validation shortcuts:

- longer candidate;
- shorter candidate;
- greater token overlap with the supplied context;
- more numeric tokens.

These are descriptive baselines, not trained models.

### Run

```powershell
.\scripts\94-run-preference-learnability.ps1
```

Outputs:

```text
artifacts/postmortem/preference-learnability/
├── runs/
│   ├── memorize-256.json
│   ├── memorize-512.json
│   ├── budget-600.json
│   ├── budget-1200.json
│   ├── budget-2400.json
│   ├── budget-5000.json
│   └── budget-10000.json
├── preference-learnability-summary.json
└── preference-learnability-summary.md
```

### Interpretation

```text
tiny-set memorization fails
→ implementation or optimization failure remains possible

memorization succeeds, validation remains null-like
→ the target can be fitted but does not generalize from this input view

validation becomes learned-above-prior at a larger budget
→ the original 600-update budget was insufficient on this fold
```

A one-fold exploratory success is not a new confirmatory result. It determines whether a
full preregistered replication is worth designing.

## Blinded oracle packet

Export a balanced validation sample with prompts and answers in separate files:

```powershell
.\scripts\95-export-preference-oracle.ps1
```

Outputs:

```text
artifacts/postmortem/preference-oracle/
├── oracle-prompts.jsonl
├── oracle-answer-key.jsonl
└── oracle-manifest.json
```

The prompt file contains no preference or future labels. Candidate order is the same
deterministically randomized order used by the original experiment. Keep the answer key
separate while a human or stronger model makes predictions.

A prompted model is a diagnostic rather than a formal oracle. A serious evaluation should
use multiple prompt wordings, audit positional consistency and report results by edit type.

## Gate B — Is the future label consistent with alignment?

The NewsEdits aligner compares V1 and V2 after whitespace, case and typographic-quote
normalization. The persisted `future_revised` property compares whitespace-collapsed strings
without the case/quote normalization.

Gate B re-extracts the canonical examples from the original SQLite database and counts:

```text
normalise_sentence(V1) == normalise_sentence(V2)
AND
current future_revised == true
```

### Run

For the source-specific database used in the experiment:

```powershell
.\scripts\96-audit-future-label-integrity.ps1 `
  -Database E:\data\newsedits\nyt-matched-sentences.db
```

Use the exact database and extraction options that produced the original episode set.
Optional parameters include `-Table`, `-SplitTable`, `-SourceName`, `-Sources`,
`-MaxArticles` and `-MaxExamples`.

Outputs:

```text
artifacts/postmortem/future-label-integrity/
├── future-label-integrity.json
└── future-label-integrity.md
```

The report separates:

- case-only differences;
- typographic-quote-only differences;
- combined case and quote differences;
- any other equality already implied by the existing aligner normalization.

This audit does not change the frozen result. A material disagreement means that any fresh
replication should rebuild the future target with one canonical equality rule.

## Stop/go rule

Do not implement the multi-head transfer experiment yet.

Proceed to a hard-negative relationship benchmark only after reading both Step 7 reports.
Proceed to a joint preference-plus-relationship model only when:

```text
authentic preference is demonstrably learnable
AND
hard relationship learning beats explicit shortcut baselines
```
