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

Training “temporal direction” on the exact authentic pairs would therefore reproduce the authentic labels and then pretend they represented a competing explanation.

The repository refuses that shortcut.

Instead, the temporal-direction encoder is trained on one-to-one replacements from other NewsEdits article lineages that never appear in the preference-future evaluation set. It learns generic revision newness in the same publication domain without seeing any evaluation article trajectory.

That correction narrows what the eventual result can claim. NewsEdits alone cannot fully separate “preference learning” from “learning the semantics of an accepted chronological revision,” because the observed choice and the later sentence are the same event. A positive result must first beat the independent temporal representation and should still be described carefully until it generalises to datasets where preference is not synonymous with chronological replacement.

### The six trained regimes

**Language adaptation** reconstructs deterministic masked words from the same NewsEdits pair-and-context inputs. It controls for domain exposure.

**Pair exposure** predicts whether two candidates came from the same revision episode. Every candidate remains exposed, but authentic preference labels are absent.

**Temporal direction** predicts the newer candidate using external, evaluation-disjoint NewsEdits lineages.

**Random labels** use the authentic pair inputs with deterministic balanced targets unrelated to the decision.

**Shuffled preference** preserves the authentic label count but takes each target from a different article lineage.

**Authentic preference** predicts which candidate the editor retained.

The source-task artifacts are accepted only when:

```text
all six corpora have identical record counts
no test lineage enters preference-derived source training
future and V2 fields are absent
random and pair-exposure labels are balanced
negative and shuffled donors cross article lineages
temporal articles are disjoint from evaluation articles
all 120 expected corpus files survive persisted verification
```

### Result

```text
Status:                             PENDING LOCAL RUN
Independent temporal pairs:         PENDING
Independent temporal lineages:      PENDING
Corpus JSONL files:                 120 expected
Records per train corpus by fold:   PENDING
Records per validation corpus:      PENDING
Builder gates:                      PENDING
Persisted verification gates:       PENDING
```

A passing Step Two result proves only this:

> The authentic revision-choice objective and five trained alternatives were frozen before model training with equal source-record budgets, no direct future-label leakage and an independent temporal-control pool.

It does not yet prove equal compute. The next step must train every regime from the same checkpoint with the same tokenizer, fixed sequence length, padding policy, optimiser, batch size, learning-rate schedule and number of updates.

It also does not prove transfer.

It makes the transfer comparison hard enough to be worth believing.
