# Step 4 Source-Task Diagnostics Result

**Status:** VERIFIED

## Frozen encoder inventory

| Measure | Value |
|---|---:|
| Folds | 10 |
| Arms per fold | 7 |
| Manifest entries | 70 |
| Eligible entries | 70 |
| Trained encoders | 60 |
| Unique trained hashes | 60 |
| Trained/base hash collisions | 0 |

## Aggregate diagnostics

| Regime | Accuracy | Mean loss | Status |
|---|---:|---:|---|
| language_adaptation | 44.9265% | 3.710544 | learned |
| pair_exposure | 99.3033% | 0.033325 | learned above prior |
| temporal_direction | 49.4194% | 0.693319 | null-like |
| random_label | 49.7180% | 0.693344 | null-like |
| shuffled_preference | 50.3318% | 0.693599 | null-like aggregate |
| authentic_preference | 50.2571% | 0.693361 | null-like |

Nine shuffled-preference folds were null-like. Fold 5 was below the class prior and remains eligible as a mechanically valid preregistered control.

## Selection rule

Downstream eligibility is determined by mechanical artifact validity, not by source-task success. Source-task results remain diagnostic labels. No checkpoint was changed and no regime was retrained after observing its result.

## Interpretation

Pair exposure and language adaptation produced clear source-task learning. Temporal direction and authentic preference did not. This weakens the direct source-learning account, but it does not test whether any trained encoder improves prediction of the later selected-branch outcome.

The central transfer hypothesis therefore remains pending Steps 5 and 6.
