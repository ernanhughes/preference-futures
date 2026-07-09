# Step 6 Confirmatory Result

**Status:** VERIFIED — NOT SUPPORTED

## Primary comparison

```text
generic pooled out-of-fold log loss
-
authentic-preference pooled out-of-fold log loss
```

**Observed improvement:** -0.000719

**Paired lineage-bootstrap 95% CI:** [-0.003425, +0.002020]

**Episodes:** 12,056

**Lineages:** 3,386

**Probe jobs:** 70/70 verified

## Conclusion

Under the frozen DistilBERT, source-training, pair-and-context input, final-layer first-token representation, and identical linear-probe contract, authentic-preference training did not improve out-of-fold prediction of future revision relative to the untouched generic encoder.

## Claim limits

- The result tests linear decodability from the frozen Step 5 representation.
- It does not prove that preference supervision can never transfer in another model or task.
- Analyses designed after observing this result are exploratory unless replicated on fresh data.

## Seven-arm results

| Arm | Log loss | Improvement vs generic | Brier | ROC AUC |
|---|---:|---:|---:|---:|
| authentic_preference | 0.540832 | -0.000719 | 0.177918 | 0.628365 |
| generic | 0.540113 | +0.000000 | 0.177659 | 0.630119 |
| language_adaptation | 0.538201 | +0.001913 | 0.176783 | 0.630541 |
| pair_exposure | 0.536768 | +0.003345 | 0.176264 | 0.633184 |
| random_label | 0.546001 | -0.005887 | 0.179883 | 0.622488 |
| shuffled_preference | 0.546044 | -0.005931 | 0.180006 | 0.623609 |
| temporal_direction | 0.548960 | -0.008847 | 0.181052 | 0.613058 |

This file marks the completed confirmatory experiment. Subsequent analyses are
exploratory unless repeated on fresh held-out data.
