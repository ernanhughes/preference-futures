## Step Two: Freeze What Every Encoder Is Allowed to Learn From

A fair representation experiment cannot begin by training the preferred model and inventing its controls afterward.

Before touching an encoder, we materialised six additional-training regimes under the article-grouped boundary frozen in Step One:

```text
language adaptation
pair exposure
temporal direction
random labels
shuffled preference
authentic preference
```

The untouched pretrained encoder remains a seventh comparison arm.

Each trained regime receives exactly the same number of source-task records inside each fold’s train and validation partitions. None receives the future-revision label. None receives V2. Every preference-derived record remains inside the lineages assigned to that fold’s train or validation partition.

Run:

```powershell
.\scripts\50-build-compute-matched-corpora.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db" `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -SplitsDirectory artifacts\transfer\splits `
  -OutputDirectory artifacts\transfer\corpora `
  -Seed 17
```

The command writes the corpus manifest, all ten folds, an independent temporal-pair pool and a persisted verification report.

### The temporal control forced a correction

At first, the obvious temporal control seemed to be:

```text
show the same V0 and V1 pair
predict which sentence is newer
```

But on these episodes that is not a different target.

```text
V0 = earlier rejected sentence
V1 = later retained sentence
```

After candidate randomisation:

```text
retained candidate index
=
newer candidate index
```

Training temporal direction on the exact authentic pairs would therefore reproduce the authentic labels and then pretend they represented a competing explanation.

The repository refuses that shortcut.

Instead, the temporal-direction encoder is trained on one-to-one replacements from other NewsEdits article lineages that never appear in the preference-future evaluation set. It learns generic revision newness in the same publication domain without seeing an evaluation article trajectory.

That correction narrows what the eventual result can claim. NewsEdits alone cannot fully separate preference learning from learning the semantics of an accepted chronological revision. A positive result must beat the independent temporal representation and should still be described carefully until it generalises to datasets where preference is not synonymous with chronological replacement.

### The six trained regimes

**Language adaptation** reconstructs deterministic masked words from the same NewsEdits pair-and-context inputs. It controls for domain exposure.

**Pair exposure** predicts whether two candidates came from the same revision episode. Every candidate remains exposed, but authentic preference labels are absent.

**Temporal direction** predicts the newer candidate using external, evaluation-disjoint NewsEdits lineages.

**Random labels** use the authentic pair inputs with deterministic balanced targets unrelated to the decision.

**Shuffled preference** preserves the authentic label count but takes each target from a different article lineage.

**Authentic preference** predicts which candidate the editor retained.

The source-task artifacts were accepted only if:

```text
all six corpora had identical record counts
no test lineage entered preference-derived source training
future and V2 fields were absent
random and pair-exposure labels were balanced
negative and shuffled donors crossed article lineages
temporal articles were disjoint from evaluation articles
all 120 expected corpus files survived persisted verification
```

### The real-data result

The seed-17 run used the same 12,056 preference episodes and 3,386 evaluation lineages frozen in Step One.

For the independent temporal control, it extracted:

```text
24,112 temporal-direction pairs
5,135 external article lineages
0 evaluation-lineage overlap
```

The extractor selected 20,000 external articles and reached the target after reading 6,849 of them. It examined 44,840 one-to-one replacement candidates before accepting the required pool.

All ten folds were then materialised.

```text
train records per corpus:      9,643–9,646
validation records per corpus: 1,204–1,207
expected corpus files:         120
observed corpus files:         120
persisted records verified:    651,024
builder gates passed:          10 of 10
verification checks passed:    10 of 10
verification errors:           0
```

The verifier reopened every corpus file. Every line count matched the manifest. Every record identified the correct fold, partition and corpus. Every source ID was unique within its file. All frozen source hashes still matched. No future label or V2 field appeared anywhere.

That closes the data-contract question.

### Equal records are not yet equal compute

The authentic, language-adaptation, random-label and shuffled-preference regimes use the same underlying episode text, so their exposure totals are identical. Pair exposure differs negligibly because its negative examples substitute candidate B from another lineage.

The external temporal corpus is longer. Across the ten training folds, its whitespace-token exposure was between 6.51% and 7.47% higher than authentic preference, with a mean difference of 7.05%. Validation showed a similar mean difference of 7.09%, although individual folds ranged from 3.70% to 10.76%.

That does not invalidate Step Two. It tells Step Three exactly what it must control.

Every regime must use:

```text
one starting checkpoint
one tokenizer
one frozen maximum sequence length
one padding policy
one batch size
one optimiser and schedule
one update count
one fixed checkpoint rule
```

Otherwise the temporal arm could receive more effective compute merely because its source sentences are longer.

### What Step Two established

> The authentic revision-choice objective and five trained alternatives were frozen before model training with equal source-record budgets, no direct future-label leakage and an independent temporal-control pool.

That is now verified rather than planned.

It does not prove equal optimisation compute. The next step must enforce it.

It does not prove transfer.

It makes the transfer comparison hard enough to be worth believing.
