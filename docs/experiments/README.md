# Representation-Transfer Experiment Steps

This directory extends the blog into an executable sequence. Each step contains:

- the question being tested;
- the code and command;
- frozen pass/fail criteria;
- generated artifacts;
- the limited claim supported by a passing result;
- a results section completed only after a local run.

## Sequence

| Step | Document | Status |
|---:|---|---|
| 1 | [Freeze article-grouped split manifests](01-grouped-split-manifests.md) | Implemented; awaiting real-data run |
| 2 | Build compute-matched training corpora | Planned |
| 3 | Train authentic and control representations | Planned |
| 4 | Verify source-task learning and freeze encoders | Planned |
| 5 | Extract frozen representations | Planned |
| 6 | Train identical future probes | Planned |
| 7 | Run metadata and numeric-only baselines | Planned |
| 8 | Run numerical, clean-prose and reversal ablations | Planned |
| 9 | Run future-label sample-efficiency curves | Planned |
| 10 | Produce the confirmatory transfer decision report | Planned |

## Rule

A later step must consume the committed artifacts from the earlier step. It must not silently regenerate splits, labels or shortcut flags with a new seed or changed definition.

The publication-facing prose fragments live under:

```text
docs/blog/blocks/
```

After each local run, the generated results are copied into both the detailed step document and its matching blog block.
