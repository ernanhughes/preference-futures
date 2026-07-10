# Step 8 Editorial MR.Q result

## Status

Exploratory positive result under the frozen Step 8 and Step 8.4 decision rules.

## Source task

Across 12,056 pooled out-of-fold episodes:

| Ranker | Accuracy | Log loss | 95% accuracy interval | Gate |
|---|---:|---:|---|---|
| Linear frozen embeddings | 0.524635 | 0.728408 | [0.515714, 0.533540] | failed |
| Editorial MR.Q | 0.523308 | 0.691868 | [0.514386, 0.532215] | passed |

MR.Q improved pooled source-task log loss by 0.036540 relative to the antisymmetric linear embedding control while preserving exact candidate-order symmetry.

## Future transfer

Across the same 12,056 pooled out-of-fold episodes and 3,386 article lineages:

| Arm | Accuracy | Log loss | Brier score | ROC AUC |
|---|---:|---:|---:|---:|
| generic_unoriented | 0.748507 | 0.563218 | 0.186420 | 0.605822 |
| generic_choice_aware | 0.750000 | 0.565477 | 0.186327 | 0.603013 |
| mrq_blind | 0.747760 | 0.555810 | 0.185051 | 0.603322 |
| mrq_choice_aware | 0.746184 | 0.556076 | 0.185135 | 0.602955 |

Primary comparison:

- MR.Q choice-aware minus generic choice-aware log loss: -0.009401
- paired lineage-bootstrap 95% interval: [-0.015635, -0.003066]
- treatment-minus-control accuracy: -0.003816

Secondary comparison:

- MR.Q blind minus generic unoriented log loss: -0.007408
- paired lineage-bootstrap 95% interval: [-0.013178, -0.001707]
- treatment-minus-control accuracy: -0.000747

The frozen Step 8.4 decision rule therefore passed.

## Supported claim

A compact representation produced by authentic editorial-preference training improves held-out future-revision log loss relative to the corresponding raw generic semantic geometry under the frozen identical-probe protocol.

## Important qualification

The improvement is probabilistic rather than classificatory. MR.Q improves log loss and Brier score, but not threshold accuracy or ROC AUC. The MR.Q states are also much lower-dimensional than the generic controls. Consequently, the current result does not yet isolate preference-specific information from compression, regularisation, or calibration effects.

The next required experiment is a dimension- and regularisation-matched specificity audit, followed by an identically shaped MR.Q null trained on shuffled preference labels if the compression controls do not explain the gain.
