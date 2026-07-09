+++
date = '2026-07-09T12:30:00+01:00'
draft = true
title = 'What Does a Preference Know About the Future?'
+++

## An Executable Test of Whether Preference Learning Captures Future-Relevant State

Most preference-learning systems use a choice to change the future.

A model produces two responses. A human selects one. The chosen response becomes positive evidence, the rejected response becomes negative evidence, and training makes outputs resembling the chosen response more likely.

The preference acts as an instruction:

> Produce more things like this.

That is an extraordinarily useful way to learn from human judgement.

It may not be the only one.

A human choice can also be treated as an observation.

When an editor replaces one sentence with another, the decision may reflect more than an immediate judgement of quality. The old sentence may contain an unresolved fact. The replacement may satisfy an editorial constraint that remains active throughout later revisions. The choice may reveal that the underlying state is stable—or that it is still moving.

When a programmer rejects one patch and accepts another, the decision may reflect expectations about tests, maintenance and future regressions.

When a support agent selects one response over another, the decision may contain an implicit forecast about whether the case will close, reopen or escalate.

The evaluator normally does not state that forecast.

The preference record preserves only its compressed result:

```text
A > B
```

Modern preference learning reads that record primarily as an optimisation signal:

> Make A more likely than B.

This post asks whether the same record can be read in the opposite direction:

> What does the selection of A over B reveal about the future that follows?

That is not another way of asking which candidate the human prefers.

It is a different prediction problem.

Conventional preference learning estimates:

$$
P(Y \mid H,C)
$$

where:

- $H$ is the current history or state;
- $C$ is the available candidate set;
- $Y$ is the observed or predicted preference.

Preference-conditioned forecasting estimates:

$$
P(F \mid H,C,Y)
$$

where $F$ is a specified later outcome.

The preference stops being the target.

It becomes evidence.

This post does not merely propose that idea. It turns the first stage of the argument into an executable repository:

> [github.com/ernanhughes/preference-futures](https://github.com/ernanhughes/preference-futures)

Every empirical statement below is tied to a script, an output artifact and a condition under which the statement would have to be weakened or abandoned.

---

## What the First Experiment Already Established

My previous post, *The Preference Was Only the Beginning*, tested the narrowest useful form of this idea.

It used sentence-revision histories from NewsEdits. Each clean episode followed three successive states from the same article lineage:

```text
V0: earlier sentence
        ↓ replaced by
V1: retained replacement
        ↓ survives or changes
V2: next observed state
```

The question was not whether the editor preferred V1. That preference had already been revealed by the revision.

The question was:

> After accounting for the retained sentence and its context, does the rejected sentence improve prediction of whether the retained sentence changes again?

The baseline estimated:

$$
P(F \mid H,S_{\text{retained}})
$$

The preference-informed model estimated:

$$
P(F \mid H,S_{\text{retained}},S_{\text{rejected}},R)
$$

where $R$ represented the measured relationship between the retained and rejected sentences.

Across 133,872 revision episodes from 35,816 article lineages, the authentic rejected sentence improved prediction of whether the retained sentence would change again.

The effect was modest, but consistent:

- positive under log loss;
- positive under Brier score;
- positive in all ten article-grouped train/test splits;
- stronger than a matched permutation control;
- carried primarily by the language that had actually been removed.

The rejected sentence alone accounted for approximately 82% of the full forecasting gain. Rejected text combined with its lexical relationship to the retained sentence recovered approximately 99% of the gain. Edit magnitude alone did not reliably explain it.

That experiment established a narrow empirical fact:

> Authentic preference evidence can contain incremental information about a decision-linked future.

It did not establish why.

It also did not show that a model trained to predict human preference learns anything about future consequences.

The future predictor was trained directly on future labels. It learned:

```text
current text
+
rejected text
+
later outcome labels
→
future forecast
```

That is a valid forecasting result.

But it leaves the deeper question unanswered.

---

## Did the Preference Objective Learn the Signal?

Suppose we train a model only to predict which alternative an editor retained.

The model sees:

```text
article state
candidate A
candidate B
```

and learns:

```text
which candidate replaced the other?
```

It is never shown whether the selected sentence survives the next revision.

After training, we freeze the encoder. We then train a small future-prediction head on top of its internal representation:

```text
frozen preference representation
+
small future head
→
will the selected sentence change again?
```

Let:

$$
\phi_{\text{generic}}(H,C)
$$

be the representation produced by the original encoder, and:

$$
\phi_{\text{preference}}(H,C)
$$

be the representation produced after training the same encoder to predict the authentic editorial choice.

We train identical future heads $g$ and compare:

$$
g\!\left(\phi_{\text{generic}}(H,C)\right)
$$

against:

$$
g\!\left(\phi_{\text{preference}}(H,C)\right)
$$

on the held-out future target $F$.

The hypothesis is:

$$
\mathcal{L}\!\left(F,g[\phi_{\text{preference}}(H,C)]\right)
<
\mathcal{L}\!\left(F,g[\phi_{\text{generic}}(H,C)]\right)
$$

where $\mathcal{L}$ is a probabilistic forecast loss.

A positive result would mean that preference-specific training produced a representation that was more useful for consequence prediction than ordinary language representation.

A negative result would mean something equally important:

> Preference evidence can contain future information without a preference-prediction objective learning how to preserve it.

That is the central experiment.

But before running it, the dataset itself had to survive a hostile audit.

---

## The Repository Turns the Question Into a Test

The repository is deliberately organised as a sequence of falsifiable stages rather than one large training script.

The canonical episode is:

```text
V0: rejected sentence
V1: selected replacement
V2: later observed state of the selected branch
```

Candidate order is deterministically randomised, but the selected identity and future outcome remain attached to V1.

The official NewsEdits databases do not contain one convenient article-version table. The source-specific files contain a `split_sentences` table with approximately this shape:

```text
entry_id
version
sent_idx
sentence
```

The adapter reconstructs ordered article versions from those rows, extracts conservative one-to-one V0→V1 replacements and then resolves the observed fate of V1 in V2.

Run the schema inspection:

```powershell
.\scripts\10-newsedits-inspect.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

The detected schema should identify:

```text
split_sentences.entry_id
split_sentences.version
split_sentences.sent_idx
split_sentences.sentence
```

That proves only that the database can be read.

The next stage asks whether enough scientifically useful episodes survive.

---

## Claim One: The Dataset Is Large Enough to Test the Hypothesis

The deterministic viability run sampled 5,000 New York Times article lineages with seed 17.

Run the complete evidence pipeline:

```powershell
.\scripts\30-reproduce-blog-evidence.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

The extraction stage writes:

```text
artifacts/newsedits/blog-evidence/episodes.jsonl
artifacts/newsedits/blog-evidence/audit.json
```

The current verified run produced:

| Measure | Result |
|---|---:|
| Articles sampled | 5,000 |
| Articles yielding at least one episode | 3,386 |
| Accepted episodes | 12,056 |
| Replacement opcodes considered | 40,343 |
| Accepted replacement rate | 29.88% |
| Stable selected futures | 8,952 |
| Revised selected futures | 3,104 |
| Future-revision rate | 25.75% |

The 95% Wilson interval for the future-revision rate is approximately 24.97% to 26.53%.

This clears the first viability gate:

- more than 10,000 episodes;
- more than 3,000 independent article lineages;
- a future target that is imbalanced but not degenerate;
- enough positive outcomes for grouped evaluation.

This does **not** prove preference transfer.

It proves that the proposed test can be run on real decision-linked trajectories at useful scale.

---

## Claim Two: The Target and Presentation Order Are Usable

The context audit is run automatically by the full reproduction script, or separately with:

```powershell
.\scripts\14-context-viability-audit.ps1 `
  -EpisodesPath artifacts\newsedits\blog-evidence\episodes.jsonl `
  -OutputDirectory artifacts\newsedits\blog-evidence
```

It writes:

```text
context-viability.json
context-viability.md
```

The audit measures:

- future-label balance;
- candidate-order balance;
- episodes per article lineage;
- context availability;
- residual sentence-boundary artifacts;
- exact candidate-pair reuse and reversals;
- edit-similarity bands;
- sentence-position bands;
- version-position bands.

The extracted candidate order is approximately balanced: selected V1 appears as candidate A and candidate B at close to equal rates.

About 90% of episodes have local context on both sides of the selected sentence.

Residual source-boundary artifacts remain, but after preserving the official NewsEdits sentence rows they fall below the frozen 3% gate.

The audit also exposes a fact that is easy to hide with row-level random splits: exact candidate pairs can reverse direction at later points in the same article.

A sentence can be selected over another sentence and later lose to the same alternative.

That means preference is not an immutable ranking between strings.

It is conditional on article state.

All train, validation, test and bootstrap operations must therefore group by `lineage_id`.

---

## The Reviewer Objection That Changed the Experiment

A critic raised a simple example:

```text
50 people died in New York
→
40 people died in New York
→
30 people died in New York
```

Each transition is technically an edit. But perhaps the model does not learn anything about editorial preference or latent constraints.

Perhaps it learns only this:

> Provisional numerical claims tend to be revised again.

That criticism is exactly the kind of thing the repository must be able to test rather than discuss rhetorically.

Run:

```powershell
.\scripts\15-numeric-shortcut-audit.ps1 `
  -EpisodesPath artifacts\newsedits\blog-evidence\episodes.jsonl `
  -OutputDirectory artifacts\newsedits\blog-evidence
```

The audit writes:

```text
numeric-shortcut.json
numeric-shortcut.md
numeric-flags.jsonl
```

Each episode receives explicit flags including:

```text
contains_number
number_changed
number_only_edit
number_dominant_edit
date_or_update_edit
money_or_percentage_edit
sports_numeric_edit
casualty_count_update
repeated_numeric_trajectory
repeated_casualty_trajectory
```

The first audit found a much more interesting answer than either simple acceptance or rejection of the criticism.

### The narrow casualty criticism is real but small

Repeated casualty and injury updates exist. We found evolving lineages in which death or injury counts moved repeatedly as the event developed.

But casualty-count edits account for less than 1% of the extracted dataset. Repeated casualty trajectories are smaller again.

Removing every casualty-count update barely changes the overall future-revision rate.

So this claim is not supported:

> The entire result is just death tolls being updated.

The data disproves that explanation as a dataset-wide account.

### The broader numerical-volatility criticism is real

Numerical edits are much broader than casualty counts. They include:

- dates and update headers;
- market prices and percentages;
- election totals;
- sports scores and rankings;
- case, arrest and flight counts;
- weather measurements;
- casualty and injury totals.

In the current 5,000-article artifact, approximately 10% of episodes change one or more numerical values. Those episodes have a future-revision rate of roughly 52%, compared with approximately 26% for the dataset as a whole.

Number-dominant edits—where the text is almost identical after replacing values with `<NUM>`—are revised again at an even higher rate.

Excluding all numerical-change episodes lowers the dataset-wide future-revision rate by roughly three percentage points.

That is too large to dismiss as noise.

The stronger conclusion is:

> Repeated casualty updates do not dominate the dataset, but volatile numerical claims form a genuine shortcut class.

The critic was wrong about the scale of the narrow example and right about the underlying mechanism.

---

## The First New AI Contribution: Preference Data Is Also Volatility Data

A preference event can carry more than one kind of information.

Consider:

```text
“At least 35 people were killed”
→
“At least 40 people were killed”
```

The revision is simultaneously:

- a selected textual replacement;
- a factual correction;
- evidence that the underlying event remains unresolved;
- evidence that another correction is more likely.

The same supervision event contains at least two latent signals:

```text
Which text was selected?
What kind of state produced this selection?
```

Preference datasets are normally treated as value data:

```text
chosen
versus
rejected
```

Revision-derived preference data can also be state data:

```text
stable claim
versus
provisional claim
```

That distinction matters well beyond news editing.

The same confound can appear whenever preference or revision records contain inherently volatile fields:

- prices;
- dates;
- measurements;
- counts;
- rankings;
- test results;
- performance metrics;
- changing external facts.

A model can appear to learn the future consequences of a choice while actually learning which fields are likely to move again.

The methodological contribution is therefore not merely “mask numbers.”

It is a broader audit principle:

> When preference data is paired with later outcomes, separate preference learning from latent state-volatility learning.

That requires explicit controls, not a limitation paragraph added after training.

---

## The Numerical Controls Are Now Part of the Main Experiment

The transfer experiment must be reported under at least four numerical conditions.

### All episodes

The primary dataset, preserving the natural distribution of editorial decisions.

### Casualty-count updates excluded

This directly answers the original “50 → 40 → 30 dead” criticism.

### Number-dominant edits excluded

This removes episodes whose apparent semantic revision is largely explained by changed values.

### Numbers masked

Every numerical expression is replaced with the same token:

```text
“At least 40 people were killed”
→
“At least <NUM> people were killed”
```

The model can still see that a quantity is present, but not its exact value.

The model must also beat a numeric-features-only baseline receiving:

- whether either candidate contains a number;
- numerical-expression counts;
- whether the numbers differ;
- date, money, percentage, sports and casualty flags;
- article version position;
- sentence position.

If apparent preference transfer disappears under these controls, that is not a failed experiment.

It is a result:

> The preference objective learned latent numerical volatility rather than a broader consequence representation.

If transfer weakens but survives, the representation contains a mixture of shallow volatility cues and deeper structure.

If transfer survives unchanged, numerical volatility cannot explain the effect.

All three outcomes teach us something.

---

## Preference Prediction Is Not Future Prediction

The intuition behind the experiment is easy to overstate.

If a model can predict what a human chooses, perhaps it has learned why the human chose it. If the choice reflects anticipated consequences, perhaps the same representation can help predict those consequences.

But none of that follows automatically.

From:

$$
P(Y \mid H,C)
$$

we cannot mathematically recover:

$$
P(F \mid H,C,Y)
$$

The two tasks may share information, but they are not inverses.

A useful hypothesis introduces a latent condition $Z$:

$$
Z \rightarrow Y
$$

$$
Z \rightarrow F
$$

The hidden condition might include:

- an unresolved factual problem;
- an editorial constraint;
- an expected reader reaction;
- an institutional rule;
- an anticipated downstream cost;
- a structural incompatibility;
- an implicit objective;
- the volatility of the underlying state.

The observed preference provides evidence about that hidden condition:

$$
P(Z \mid H,C,Y)
$$

The future also depends on it:

$$
P(F \mid H,C,Z)
$$

Therefore preference may improve the forecast:

$$
P(F \mid H,C,Y)
=
\int P(F \mid H,C,Z)P(Z \mid H,C,Y)\,dZ
$$

That is a hypothesis, not an identification result.

The preference does not cause the future. Both may be influenced by an unobserved editorial state.

The experiment is predictive:

> Did the preference objective preserve information useful for forecasting the selected branch?

It is not causal:

> What would have happened if the rejected sentence had been selected instead?

NewsEdits observes only the future of the branch that was actually retained.

---

## What Authentic Preference Learning Must Beat

A preference-trained representation beating an untouched encoder would not be enough.

It would be confounded by extra training, NewsEdits domain exposure and revision-pair exposure.

The experiment requires compute- and data-matched alternatives.

### Generic language representation

The original pretrained encoder.

### Compute-matched language adaptation

The same encoder receives additional language-model training on the same NewsEdits text for approximately the same optimisation budget.

This controls for domain adaptation.

### Pair-exposure representation

The encoder sees the same earlier and later sentence pairs but is not trained to identify the selected sentence.

This controls for merely seeing revision pairs.

### Temporal-direction representation

The encoder predicts which sentence is newer.

This is the strongest direct test of the “newness detector” explanation.

### Random-label representation

The encoder follows the same preference-training pipeline with random labels.

This controls for additional optimisation without meaningful supervision.

### Matched shuffled-preference representation

The encoder receives preference-shaped supervision whose link to the authentic decision lineage has been broken.

### Authentic preference representation

The encoder predicts which sentence the editor actually retained.

The decisive comparison is not:

```text
preference representation
>
generic representation
```

It is:

```text
authentic preference representation
>
best compute-matched non-preference representation
```

under article-lineage grouped evaluation and the numerical shortcut controls.

That is a considerably harder test.

It is also the test that matters.

---

## What Is Proved, Refined and Still Pending

The repository contains a living [claim ledger](../CLAIMS.md).

The current state is:

| Claim | Status |
|---|---|
| Official NewsEdits databases can be reconstructed into ordered article versions | Verified |
| A 5,000-article sample yields enough V0→V1→V2 episodes for grouped experiments | Verified |
| The future target is usable rather than degenerate | Verified |
| Candidate A/B order is approximately balanced | Verified |
| Casualty-count update sequences exist | Verified |
| Casualty updates dominate the dataset | Rejected and refined |
| Numerical revisions form a predictive shortcut class | Verified |
| Authentic preference training improves future prediction | Pending |
| Any improvement survives temporal, numerical and compute-matched controls | Pending |
| Preference training improves future-label sample efficiency | Pending |

This separation matters.

The data pipeline and shortcut discovery are already empirical contributions.

The representation-transfer claim remains a falsifiable hypothesis.

The blog should not pretend otherwise.

---

## Reproduce the Evidence

Clone the repository and install the development environment:

```powershell
git clone https://github.com/ernanhughes/preference-futures.git
cd preference-futures
.\scripts\00-setup.ps1
```

Run the complete current evidence chain:

```powershell
.\scripts\30-reproduce-blog-evidence.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

The command performs:

```text
repository checks
→
NewsEdits extraction
→
artifact verification
→
context viability audit
→
numeric shortcut audit
```

It writes:

```text
artifacts/newsedits/blog-evidence/episodes.jsonl
artifacts/newsedits/blog-evidence/audit.json
artifacts/newsedits/blog-evidence/context-viability.json
artifacts/newsedits/blog-evidence/context-viability.md
artifacts/newsedits/blog-evidence/numeric-shortcut.json
artifacts/newsedits/blog-evidence/numeric-shortcut.md
artifacts/newsedits/blog-evidence/numeric-flags.jsonl
```

Each reported claim maps to one of those artifacts.

Readers do not have to trust a table copied into a blog post. They can inspect the extraction funnel, read individual episodes, change the seed, tighten the filters or challenge the category definitions.

That is the standard this kind of AI claim should meet.

---

## How This Thesis Could Be Wrong

The repository is designed to make the thesis easier to reject, not harder.

The deeper preference-transfer hypothesis is wrong or materially weakened if:

- authentic preference training does not beat the generic encoder;
- compute-matched language adaptation explains the gain;
- temporal-direction training matches or beats authentic preference;
- random or shuffled labels produce the same transfer;
- a metadata-only future predictor explains the result;
- the advantage disappears after masking numbers;
- the advantage disappears after excluding number-dominant edits;
- the advantage exists only in exact candidate-pair reversals;
- row-level gains disappear under article-grouped splits;
- the result requires so much future-labelled data that preference pretraining provides no sample-efficiency benefit.

Any one of those outcomes would narrow the claim.

Several would reduce it to a shortcut result.

That would still be useful.

A model can become better at predicting human choices without becoming better at representing consequences. If that is what the experiment finds, it is a warning for preference optimisation far beyond writing systems.

---

## From Preference as Instruction to Preference as State Update

Preference learning currently treats choice as a force applied to a model.

The human selects A over B, and the model changes so that A-like outputs become more probable.

This research treats the same choice as an update to our understanding of the situation.

Before the decision, several hidden explanations may be possible.

After the decision, some become more likely:

```text
the explicit wording was unstable
the factual claim was unresolved
the structure rejected explanation
the institution preferred a safer formulation
the selected branch preserved more future options
the underlying numerical state was still volatile
```

The preference does not reveal the complete hidden state.

It changes its probability distribution.

That is the deeper transformation:

```text
preference as target
→
preference as evidence
→
preference as latent-state update
```

The first experiment established that authentic rejected text can improve a direct future forecast.

The repository now establishes that the second experiment is viable—and that it contains a real shortcut class that must be controlled before any positive representation result can be trusted.

The question is no longer only:

> What did the human prefer?

It is:

> What did learning that preference teach the model about what happens next—and how much of that lesson was merely learning which parts of the world were still moving?
