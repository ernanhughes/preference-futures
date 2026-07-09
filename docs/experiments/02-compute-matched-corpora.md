# Step 2 — Build the Compute-Matched Source Corpora

## Question

Can we construct the authentic preference source task and its strongest alternatives before training, using the frozen Step 1 lineages, without leaking the future target and without giving one trained regime more source examples than another?

Step 2 is a data-contract step. It does not train an encoder and it does not test future transfer.

It freezes exactly what each source-task encoder will be trained to do.

## The correction discovered while designing the temporal control

The first design proposed training temporal direction on the same V0→V1 pairs used for authentic preference prediction.

That is not an independent control.

For these revision-derived episodes:

```text
V0 = earlier rejected sentence
V1 = later retained sentence
```

After candidate-order randomisation, both source tasks have the same target:

```text
which candidate did the editor retain?
=
which candidate is newer?
```

An exact-pair temporal-direction model would therefore receive the same inputs and labels as the authentic preference model. Renaming the target would not create a different experiment.

Step 2 records this identification limit explicitly rather than hiding it.

The independent temporal-direction corpus is instead extracted from other NewsEdits article lineages that never appear in the preference-future evaluation set. It uses the same publication domain and revision mechanism, but it cannot leak any evaluation article trajectory.

This does not fully identify a uniquely “preference-specific” mechanism. A positive final result must still be described as transfer from authentic revision-choice supervision unless it also beats this independent temporal representation and later survives broader datasets in which preference is not synonymous with chronological replacement.

## Frozen inputs

Step 2 consumes:

```text
artifacts/newsedits/viability-5000/episodes.jsonl
artifacts/transfer/splits/manifest.json
artifacts/transfer/splits/fold-00.json
...
artifacts/transfer/splits/fold-09.json
E:/data/newsedits/nyt-matched-sentences.db
```

The episode and split hashes from Step 1 remain authoritative. Step 2 does not create a new row-level or lineage-level split.

## The seven comparison arms

Six encoders receive additional source-task training. The untouched generic encoder is retained as a seventh comparison arm.

### 1. Generic encoder

No additional training.

This arm has no Step 2 corpus.

### 2. Compute-matched language adaptation

The encoder receives the same canonical NewsEdits preference episodes, but the objective is deterministic masked-word reconstruction rather than preference prediction.

Each record specifies:

```text
source episode ID
serialised pair-and-context view
whitespace-token count
deterministic mask positions
```

This controls for additional NewsEdits language and domain adaptation.

### 3. Pair-exposure representation

The encoder predicts whether the two candidates originate from the same revision episode.

Half the records retain the true candidate pair. Half replace candidate B with candidate B from a deterministic different-lineage donor episode. The donor mapping is a permutation, so the corpus still exposes the complete candidate inventory once per partition while removing preference supervision.

This controls for learning from revision-pair structure.

### 4. Independent temporal-direction representation

The encoder predicts which candidate is newer using one-to-one sentence replacements extracted from NewsEdits articles outside all 3,386 evaluation lineages.

The temporal pool is:

- source-matched;
- article-lineage disjoint from future evaluation;
- grouped into deterministic temporal outer buckets;
- record-count matched to each authentic fold partition;
- approximately matched to the authentic input-length distribution.

This is the strongest available direct control for generic “newness detection” without duplicating the authentic labels on the exact same pairs.

### 5. Random-label representation

The encoder sees the authentic pair-and-context inputs but receives deterministic balanced labels unrelated to the editor’s choice.

This controls for additional optimisation through the same binary-classification path without meaningful supervision.

### 6. Shuffled-preference representation

The encoder sees each authentic pair-and-context input, but its target is donated by a different episode from a different article lineage.

The authentic label multiset is preserved exactly within every train and validation partition, while the link between the observed choice and its article state is broken.

This controls for preference-shaped class balance and training dynamics without authentic decision alignment.

### 7. Authentic preference representation

The encoder predicts which candidate the editor retained.

It sees no V2 sentence, future-revision label or future outcome field.

## Reproduction command

Run Step 2 only after Step 1 is verified:

```powershell
.\scripts\50-build-compute-matched-corpora.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db" `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -SplitsDirectory artifacts\transfer\splits `
  -OutputDirectory artifacts\transfer\corpora `
  -Seed 17 `
  -TemporalMaxArticles 20000 `
  -TemporalPoolMultiplier 2.0
```

The script:

```text
validates the frozen Step 1 assignments
→ extracts an external temporal-direction pool
→ builds six train corpora for every outer fold
→ builds six source-validation corpora for every outer fold
→ writes a corpus manifest and summary
→ reopens every artifact and verifies it independently
```

If the external temporal pool is too small, increase `-TemporalMaxArticles`. Do not reduce the temporal-pair target merely to force the step to pass.

## Output artifacts

```text
artifacts/transfer/corpora/
├── manifest.json
├── corpus-summary.md
├── corpus-verification.json
├── corpus-verification.md
├── temporal-pairs.jsonl
├── temporal-pairs-audit.json
├── fold-00/
│   ├── authentic_preference/
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   ├── language_adaptation/
│   ├── pair_exposure/
│   ├── temporal_direction/
│   ├── random_label/
│   └── shuffled_preference/
├── ...
└── fold-09/
```

The fold files are compact training instructions. They reference frozen source IDs instead of duplicating the article text.

## Forbidden leakage

No Step 2 source-task JSONL record may contain:

```text
future_revised
future_stable
future_label
future_outcome
v2_sentence
v2_version_id
```

The future target remains unavailable until the frozen source encoders are evaluated in the later probe stage.

## Step 2 pass criteria

The build passes only when all generated gates are true:

```text
all six trained corpora have equal record counts within every fold partition
no preference-derived source record uses the fold’s test lineages
no source-task record contains future or V2 fields
random labels are balanced
pair-exposure labels are balanced
negative pair donors come from different lineages
shuffled preference preserves authentic label counts
shuffled-label donors come from different lineages
temporal pairs are external to all evaluation lineages
temporal-direction candidate orientation is balanced
```

The persisted verifier separately checks:

```text
all 120 expected corpus files exist
persisted line counts agree with the manifest
record corpus, fold and partition fields agree with their paths
source IDs are unique inside each corpus file
source hashes still match
future fields remain absent after writing
external temporal pool and audit artifacts exist
```

For ten folds, two partitions and six trained regimes:

```text
10 × 2 × 6 = 120 corpus JSONL files
```

## What “compute matched” means here

Step 2 matches the number of source-task records and records the approximate text exposure of every regime.

That is necessary but not sufficient for equal compute.

Step 3 must enforce the real compute contract:

```text
same starting encoder checkpoint
same tokenizer
same frozen maximum sequence length
same fixed padding policy
same batch size
same optimiser
same learning-rate schedule
same number of update steps
same precision
same gradient-accumulation rule
same fixed checkpoint schedule
no task-specific early stopping
```

Different objectives produce different losses, but they must not receive different opportunities to update the encoder.

## What a passing Step 2 proves

A passing result supports this limited claim:

> The authentic preference source task and five trained alternatives have been materialised under the frozen article-grouped evaluation boundary with equal source-record budgets, no direct future-label leakage and an independent NewsEdits temporal-control pool.

It does not prove that the regimes receive equal compute until Step 3 executes the frozen trainer contract.

It does not prove that any representation predicts the future better.

## What would fail or revise this step

Step 2 must be revised before training if:

- any source-task file contains a future or V2 field;
- any fold’s test lineage enters preference-derived source training;
- any trained regime receives a different record count;
- temporal-control articles overlap the evaluation lineages;
- shuffled or negative-pair donors remain in the same lineage;
- the independent temporal pool cannot supply the required train and validation budgets;
- persisted verification disagrees with the in-memory builder gates.

## Results

### Result status

```text
PENDING LOCAL RUN
```

### Values to record

Copy the generated values from:

```text
artifacts/transfer/corpora/corpus-summary.md
artifacts/transfer/corpora/corpus-verification.md
artifacts/transfer/corpora/temporal-pairs-audit.json
```

Record:

```text
external temporal pairs extracted
external temporal article lineages
corpus files written
records per train corpus by fold
records per validation corpus by fold
minimum and maximum exposure-token estimates
all builder gates
all persisted-verification gates
source artifact hashes
```

## Next step

Step 3 implements one trainer that consumes these manifests and trains all six additional encoders under the same fixed optimisation budget.

The trainer must not inspect future labels, future-probe scores or test-fold outcomes while selecting source-task checkpoints.
