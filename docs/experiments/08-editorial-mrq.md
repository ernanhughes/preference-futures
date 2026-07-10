# Step 8 — Editorial MR.Q

Step 8 is an exploratory salvage experiment. It does not alter the frozen Steps 1–7
artifacts or the confirmatory negative result.

## Motivation

Step 7A established that DistilBERT could memorize authentic preference labels but could not
generalize them across held-out article lineages. The blinded strong-model oracle subsequently
recovered the selected candidate above chance. The remaining engineering question is whether
that textually recoverable component can be compressed into a small, stable value model.

Step 8 therefore changes two things:

1. the language representation is frozen before preference training;
2. candidate preference is represented as an order-symmetric value difference.

The language encoder receives no preference label and no future outcome.

## Step 8.0 — Candidate-swap oracle audit

Export a swapped copy of the blinded prompt packet:

```powershell
.\scripts\97-export-swapped-oracle.ps1
```

The command exchanges Candidate A and Candidate B while preserving item and episode IDs. It
does not open an answer key.

Run the swapped packet through the same oracle process used for the original packet. Save the
new file with records of the form:

```json
{"item_id": 0, "prediction": "A"}
```

Then score the two orientations:

```powershell
python -m preference_futures.editorial_mrq.cli oracle-score-swap `
  --original-predictions artifacts\postmortem\preference-oracle\oracle-predictions.jsonl `
  --swapped-predictions artifacts\step8\oracle-swap\oracle-predictions-swapped.jsonl `
  --answer-key artifacts\postmortem\preference-oracle\oracle-answer-key.jsonl `
  --output-dir artifacts\step8\oracle-swap-score
```

The primary oracle result is accuracy on items whose translated swapped answer agrees with the
original answer.

## Step 8.1–8.2 — Frozen geometry and symmetric rankers

### Prepare

```powershell
.\scripts\98-prepare-editorial-mrq.ps1
```

The default embedder is:

```text
sentence-transformers/all-mpnet-base-v2
```

The model is snapshotted at an immutable resolved revision. The default representation is
attention-masked mean pooling followed by L2 normalization.

Three views are embedded independently:

```text
context
context + candidate A
context + candidate B
```

No preference or future field is included in the text passed to the embedder.

### Extract embeddings

```powershell
.\scripts\99-embed-editorial-mrq.ps1
```

The embeddings are computed once and then reused by every fold and ranker. The embedder is
never updated by the preference labels.

### Train source-task rankers

Start with one fold:

```powershell
.\scripts\100-train-editorial-mrq.ps1 `
  -Folds "0" `
  -Rankers "all"
```

Then, only after the fold-0 reports are mechanically sound, run all ten folds:

```powershell
.\scripts\100-train-editorial-mrq.ps1
```

## Ranker A — antisymmetric linear baseline

The linear model receives only features that negate when A and B are exchanged:

```text
A - B
(A - context)^2 - (B - context)^2
A * context - B * context
```

It has no intercept. Therefore:

```text
logit(A, B) = -logit(B, A)
```

L2 regularization is selected on the lineage-held-out validation partition.

## Ranker B — tiny MR.Q value network

One shared value network scores each candidate in relation to the other candidate and the
context:

```text
q_A = Q(A, B, context)
q_B = Q(B, A, context)
preference logit = q_A - q_B
```

Because the same Q network is reused, reversing candidate order reverses the preference logit
by construction. The embedder remains frozen; only the small value network is trained.

The default network is:

```text
6 × embedding dimension
→ 256 GELU
→ dropout
→ 64 GELU
→ scalar Q
```

Validation log loss selects the checkpoint. Early stopping operates only on the validation
partition; the test partition is evaluated after selection.

## Optional teacher regularization

The training command accepts:

```powershell
-TeacherPredictions path\to\teacher-probabilities.jsonl
```

Each record must contain:

```json
{"episode_id": "...", "probability_a": 0.67}
```

Teacher probabilities are a regularizer, not ground truth. Authentic editor selections remain
the primary target. The first Step 8 run should omit teacher predictions so that frozen
geometry and pairwise architecture are tested independently.

## Source-task gate

A ranker passes only when its held-out test result satisfies all of the following:

```text
Wilson 95% lower accuracy bound > 0.50
test log loss < log(2)
maximum observed swap-logit error <= 1e-5
```

Future-transfer testing is blocked until at least one ranker passes this gate. This prevents a
repeat of the original experiment, where the downstream transfer question was asked of a
source representation that had not demonstrably learned authentic preference.

## Interpretation

### Frozen linear model passes

The original failure was primarily destructive fine-tuning or poor pooling. Editorial
preference is close to linearly recoverable from stronger semantic geometry.

### Tiny MR.Q passes but linear model fails

Preference is nonlinear but low-dimensional over stable semantic features. This is the main
Editorial MR.Q success case.

### Only teacher-regularized MR.Q passes

The authentic labels contain substantial hidden-context noise. Strong-model reasoning can
isolate and distil the textually identifiable component.

### No compact ranker passes

The oracle signal depends on reasoning that is not compressible into this frozen embedding and
small-head design. The practical evaluator remains a larger language model.

## Step 8.4 — Future transfer

Step 8.4 is intentionally not run automatically. Once a source ranker passes, its frozen
decision representation can be compared against the generic frozen embedding using the same
future-probe discipline as Step 6. Until then, no new future-transfer claim is permitted.
