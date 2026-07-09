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
| 1 | [Freeze article-grouped split manifests](01-grouped-split-manifests.md) | **Verified on 12,056 episodes / 3,386 lineages** |
| 2 | [Build compute-matched source corpora](02-compute-matched-corpora.md) | **Verified: 120 files / 651,024 persisted records** |
| 3 | [Train six representations under one fixed budget](03-fixed-budget-representation-training.md) | **Verified: 60/60 confirmatory jobs passed** |
| 4 | [Diagnose source tasks and freeze encoders](04-source-task-diagnostics-and-encoder-freeze.md) | **Implemented; local artifact generation next** |
| 5 | Extract frozen representations | Planned |
| 6 | Train identical future probes | Planned |
| 7 | Run metadata and numeric-only baselines | Planned |
| 8 | Run numerical, clean-prose and reversal ablations | Planned |
| 9 | Run future-label sample-efficiency curves | Planned |
| 10 | Produce the confirmatory transfer decision report | Planned |

## Frozen Step 1 identity

```text
seed: 17
outer folds: 10
episodes SHA-256: df4e40330ad6d3f6d4977e1630e2e54e3cfc06b01277d1aa98b7994e8c63e5ab
numeric flags SHA-256: abf517a03760da77bf60029d3385887ec6d3b73bd7db7e3d74f238ead07d75c1
```

The compact result record is [`docs/results/step-01-grouped-splits.json`](../results/step-01-grouped-splits.json).

## Frozen Step 2 identity

```text
preference episodes: 12,056
evaluation lineages: 3,386
independent temporal pairs: 24,112
independent temporal lineages: 5,135
evaluation-lineage overlap: 0
expected corpus files: 120
observed corpus files: 120
persisted source-task records verified: 651,024
```

All builder and persisted-verification gates passed.

The compact result record is [`docs/results/step-02-compute-matched-corpora.json`](../results/step-02-compute-matched-corpora.json).

## Step 2 comparison arms

Six encoders receive additional source-task training:

```text
language_adaptation
pair_exposure
temporal_direction
random_label
shuffled_preference
authentic_preference
```

The untouched pretrained encoder remains a seventh arm.

The exact-pair authentic and temporal targets are identical on V0→V1 revision pairs. Step 2 therefore builds the temporal-direction corpus from separate NewsEdits article lineages that are disjoint from every future-evaluation lineage.

The temporal arm is approximately 7% longer under the whitespace-token audit.

## Verified Step 3 optimisation boundary

The Step 3 confirmatory contract was executed without modification:

```text
base model: distilbert/distilbert-base-uncased
precision: FP32
maximum length: 256
padding: max_length
batch size: 16
updates per job: 600
trained jobs: 60
padded token positions per job: 2,457,600
padded token positions total: 147,456,000
checkpoint: final update 600
source-task early stopping: forbidden
```

The all-fold verifier observed 60/60 complete jobs, one CUDA device type, one runtime environment, matching source and artifact hashes, finite validation metrics and no errors.

Pair exposure learned strongly. Temporal direction and authentic preference remained null-like alongside random-label and shuffled-preference controls. This is a source-task result, not a future-transfer result.

Step 3 matches encoder update opportunity rather than claiming exact total FLOPs. The masked-language-model head is larger than the five binary heads, and that limitation remains explicit.

## Step 4 freeze policy

Step 4 separates:

```text
artifact validity
source-task learning
future-probe eligibility
```

Mechanically valid preregistered arms remain eligible even when their source classifier is null-like. This prevents post-result control removal and preserves the central transfer question for Steps 5 and 6.

The future-bearing episode artifact remains unavailable during Steps 3 and 4.

## Rule

A later step must consume the committed artifacts from the earlier step. It must not silently regenerate splits, labels, shortcut flags, temporal pools, corpus assignments, model snapshots, training contracts or encoder-selection manifests after observing downstream outcomes.

The publication-facing prose fragments live under:

```text
docs/blog/blocks/
```

After each local run, the generated results are copied into both the detailed step document and its matching blog block.
