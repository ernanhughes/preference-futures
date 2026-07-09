# Executable Blog Blocks

These sections extend the main draft as each representation experiment becomes executable and then verified.

## Order

| Step | Block | Status |
|---:|---|---|
| 1 | [`step-01-grouped-splits.md`](step-01-grouped-splits.md) | Verified |
| 2 | [`step-02-compute-matched-corpora.md`](step-02-compute-matched-corpora.md) | Implemented; awaiting real-data run |

## Publication rule

A block moves from procedure to finding only after its committed command has been run on the frozen artifacts and the generated result has been recorded in both:

```text
docs/experiments/
docs/results/
```

Until then, result placeholders remain explicit and the prose describes what the step tests rather than claiming what it found.

## Current insertion point

The Step 2 block replaces the weaker control outline under:

```text
## What Authentic Preference Learning Must Beat
```

in the main draft. In particular, it supersedes the idea that temporal direction can be trained as an independent objective on the exact V0→V1 preference pairs. On those pairs, retained identity and chronological newness are the same label by construction.
