# Preference Futures Claim Ledger

This document separates verified repository facts from active hypotheses. Every empirical claim must identify the command that produces its evidence, the artifact to inspect, and the result that would weaken or falsify it.

## Status vocabulary

- **Verified:** reproduced from a committed script and a real NewsEdits artifact.
- **Refined:** the original criticism was partly correct, but measurement changed the claim.
- **Pending:** the repository contains the design but not yet the completed experiment.
- **Rejected:** the evidence contradicted the proposed claim.

## Claims

| ID | Claim | Status | Reproduction | Evidence artifact | Falsification or revision condition |
|---|---|---|---|---|---|
| C1 | The official source-specific NewsEdits databases can be reconstructed into ordered article versions without requiring a full article-text table. | Verified | `scripts/10-newsedits-inspect.ps1` and `scripts/12-newsedits-full.ps1` | `audit.json` | Schema discovery fails, sentence ordering is inconsistent, or fixture and real-data reconstruction disagree. |
| C2 | A deterministic 5,000-article NYT sample yields enough clean V0→V1→V2 episodes for grouped representation experiments. | Verified | `scripts/30-reproduce-blog-evidence.ps1` | `audit.json`, `context-viability.json` | Fewer than 10,000 episodes, fewer than 3,000 lineages, or strong concentration in a small number of lineages. |
| C3 | The selected-branch future target is usable rather than degenerate. | Verified | `scripts/14-context-viability-audit.ps1` | `context-viability.json` | Future-revision prevalence falls outside the frozen 15%–40% viability band. |
| C4 | Candidate A/B presentation is not carrying the selected label. | Verified | `scripts/14-context-viability-audit.ps1` | `context-viability.json` | Selected-B rate falls outside 45%–55% or becomes predictive after grouped evaluation. |
| C5 | Repeated casualty-count updates exist in NewsEdits. | Verified | `scripts/15-numeric-shortcut-audit.ps1` | `numeric-shortcut.json`, `numeric-flags.jsonl` | No casualty-count changes or repeated casualty lineages are detected under the committed rules. |
| C6 | Repeated casualty-count updates are too rare to dominate the extracted dataset. | Refined | `scripts/15-numeric-shortcut-audit.ps1` | `numeric-shortcut.json` | Casualty-count updates exceed 2% of episodes or their exclusion materially removes the dataset-level target signal. |
| C7 | Numerical revisions more broadly form a shortcut class because changed numerical claims are unusually likely to change again. | Verified | `scripts/15-numeric-shortcut-audit.ps1` | `numeric-shortcut.json` | `number_changed` has no meaningful risk increase over its complement, or number masking/exclusion leaves no measurable difference. |
| C8 | Preference-derived datasets encode latent state volatility as well as immediate choice. | Supported interpretation | `scripts/15-numeric-shortcut-audit.ps1` plus grouped metadata baselines | Numeric audit and future baseline reports | The numeric and temporal metadata baselines fail to predict future revision above the constant prior. |
| H1 | Authentic preference training creates a frozen representation that predicts later selected-branch outcomes better than the same generic encoder. | Pending | Future representation-transfer script | Future transfer report | Authentic preference representation does not beat the generic encoder on grouped held-out lineages. |
| H2 | Any transfer advantage is specific to authentic preference rather than extra training, domain adaptation, pair exposure, or temporal discrimination. | Pending | Future compute-matched control suite | Future control comparison report | MLM, pair exposure, temporal direction, random labels, or shuffled preference match or beat authentic preference. |
| H3 | Preference transfer survives numerical masking, number-dominant exclusion, clean-prose filtering, and exact-pair-reversal exclusion. | Pending | Future ablation suite using `numeric-flags.jsonl` and context flags | Future ablation report | The advantage disappears under one or more shortcut controls. |
| H4 | Preference training improves future-label sample efficiency. | Pending | Future learning-curve script | Future sample-efficiency report | The generic representation reaches the same loss using no more future-labelled lineages. |

## Current verified checkpoint

The deterministic NYT viability run with seed `17`, `5,000` sampled articles, and a `50,000` episode cap produced:

```text
12,056 accepted episodes
3,386 article lineages
3,104 revised futures
8,952 stable futures
25.75% future-revision rate
29.88% replacement-opcode acceptance rate
```

These values establish dataset viability. They do **not** establish preference-representation transfer.

## One-command reproduction

```powershell
.\scripts\30-reproduce-blog-evidence.ps1 `
  -DatabasePath "E:\data\newsedits\nyt-matched-sentences.db"
```

The command writes:

```text
artifacts/newsedits/blog-evidence/episodes.jsonl
artifacts/newsedits/blog-evidence/audit.json
artifacts/newsedits/blog-evidence/context-viability.json
artifacts/newsedits/blog-evidence/context-viability.md
artifacts/newsedits/blog-evidence/numeric-shortcut.json
artifacts/newsedits/blog-evidence/numeric-shortcut.md
artifacts/newsedits/blog-evidence/numeric-flags.jsonl
```

## Publication rule

A blog or paper sentence that begins with “we found” must map to a **Verified**, **Refined**, or explicitly qualified **Supported interpretation** row above. Pending hypotheses must remain written as questions, experimental predictions, or falsifiable conditions.
