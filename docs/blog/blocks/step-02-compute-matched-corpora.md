## Step Two: Build the Compute-Matched Training Corpora

A preference-trained representation beating an untouched encoder would not be enough.

It could win simply because it saw more NewsEdits text. Or because it saw revision pairs. Or because it received extra optimisation. Or because it learned which sentence looked newer.

So the next repository step freezes the actual source-task corpora before any model training begins.

Run:

```powershell
.\scripts\50-build-training-corpora.ps1 `
  -EpisodesPath artifacts\newsedits\viability-5000\episodes.jsonl `
  -SplitManifestPath artifacts\transfer\splits\manifest.json `
  -OutputDirectory artifacts\transfer\corpora `
  -Seed 17
```

The builder refuses to run if the episode JSONL does not match the SHA-256 recorded in the Step 1 split manifest.

It emits six corpora:

| Corpus | What it receives | What it predicts |
|---|---|---|
| `authentic_preference` | Same pair/context text | Which candidate the editor retained |
| `language_modeling_control` | Same pair/context text | No pair label |
| `pair_exposure_control` | Same pair/context text | No selection label |
| `temporal_direction_control` | Same pair/context text | Which candidate is newer |
| `random_label_control` | Same pair/context text | A deterministic random label |
| `shuffled_preference_control` | Same pair/context text | A partition-shuffled preference label |

Every corpus uses the same serialized input shape:

```text
CONTEXT_BEFORE
CANDIDATE_A
CANDIDATE_B
CONTEXT_AFTER
```

This means the controls are matched on:

- article-lineage split;
- row population;
- fold and partition membership;
- serialized input text;
- whitespace-token input budget.

They differ only in the source-task supervision.

The shuffled-preference corpus preserves the selected-label prevalence within each fold partition while breaking the episode-specific link between text and authentic editorial choice. The random-label corpus preserves the optimization pipeline shape without meaningful supervision. The temporal-direction corpus is the direct control for the claim that the model merely learns which sentence looks newer.

Future labels are not written into corpus JSONL records. They remain reserved for the later frozen-representation future probe.

### Result

```text
Status:                         PENDING LOCAL RUN
Episodes:                       PENDING
Article lineages:               PENDING
Outer folds:                    10
Corpora:                        6
Record-count matching:          PENDING
Input-token matching:           PENDING
Future-label redaction:         PENDING
```

A passing result proves only this:

> The authentic preference source task and its controls consume the same examples, same article-grouped partitions and same input text, so later representation differences cannot be attributed to unequal corpus construction.

It does not prove that preference learning transfers to future prediction.

It makes the eventual comparison fair enough to run.
