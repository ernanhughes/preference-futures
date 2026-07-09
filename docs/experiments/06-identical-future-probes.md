# Step 6 — Train Identical Future Probes

## Question

Does an authentic-preference-trained encoder produce a frozen representation from which the selected branch's next revision is more linearly predictable than from the untouched generic encoder?

This is the central representation-transfer test.

## Inputs

Step 6 consumes only the fully verified Step 5 archive:

```text
artifacts/transfer/representations/contract.json
artifacts/transfer/representations/representation-verification.json
artifacts/transfer/representations/runs/fold-00/...
...
artifacts/transfer/representations/runs/fold-09/...
```

The Step 5 verifier must already report:

```text
70/70 jobs
210/210 partition matrices
status PASS
```

Step 6 does not load an encoder and cannot change a representation.

## Frozen target

The target is joined by `episode_id` from the original frozen episode artifact:

```text
future_revised = 1  selected V1 changes in the next observed state
future_revised = 0  selected V1 remains stable in the next observed state
```

The label was unavailable during source training and representation extraction.

## Frozen preprocessing

Every fold and arm is treated identically:

```text
fit feature mean on train only
fit feature population standard deviation on train only
replace zero or near-zero scale with 1
apply the train statistics to train, validation and test
```

No global, validation or test statistics enter preprocessing.

## Frozen probe

| Setting | Value |
|---|---|
| Architecture | one linear logit with bias |
| Link | sigmoid |
| Class weighting | none |
| Initialisation | all zeros |
| Training data | train partition only |
| Batching | full batch |
| Loss | mean binary cross entropy plus L2 on weights |
| Optimizer | PyTorch LBFGS |
| Maximum iterations | 100 per candidate |
| Line search | strong Wolfe |
| Precision | float32 |
| Retrain after selection | no |
| Post-hoc calibration | none |

The probe is deliberately linear. A positive result means the future-relevant state is linearly decodable from the frozen representation. A negative result does not prove that no nonlinear model could use it.

Class weighting is forbidden because the primary endpoint is probabilistic forecast loss, not balanced classification accuracy.

## Frozen regularization grid

Every fold and arm trains exactly five candidates:

```text
0.00001
0.0001
0.001
0.01
0.1
```

The candidate with the lowest validation log loss is selected. Numerical ties within `1e-12` are resolved in favor of stronger regularization.

The selected candidate is not retrained on train plus validation. This preserves validation as a pure selection partition and prevents the test representation from entering any fitting decision.

## Metrics

Primary:

```text
test log loss
```

Secondary:

```text
test Brier score
test ROC AUC
```

Descriptive:

```text
accuracy at 0.5
mean predicted probability
future-revision prevalence
```

Each fold also includes a constant forecast equal to the training-partition prevalence.

## Confirmatory estimand

The primary quantity is:

```text
generic pooled out-of-fold log loss
-
authentic-preference pooled out-of-fold log loss
```

A positive value means authentic-preference training improved future prediction.

Uncertainty is estimated with a paired article-lineage bootstrap:

```text
seed: 17
replicates: 10,000
interval: two-sided 95% percentile
```

The primary comparison is authentic preference versus generic.

Specificity checks are:

```text
authentic preference versus random label
authentic preference versus shuffled preference
```

Language adaptation, pair exposure and temporal direction are additional descriptive comparisons.

## Prepare the contract

Do this before training any future probe:

```powershell
.\scripts\90-prepare-identical-future-probes.ps1 `
  -RepresentationDirectory artifacts\transfer\representations `
  -OutputDirectory artifacts\transfer\probes
```

Expected output:

```text
Jobs:       70
L2 grid:    [1e-05, 0.0001, 0.001, 0.01, 0.1]
Primary:    test_log_loss
```

Do not use `-Force` after any test predictions have been generated.

## No real-data pilot

Step 5 could use a pilot because representation extraction did not inspect the future target.

Step 6 must not use a fold-0 pilot as an informal decision point. A pilot would expose real test outcomes before the full comparison. The implementation is tested with synthetic fixtures; the real contract should run all 70 jobs unchanged.

## Train or resume all probes

```powershell
.\scripts\91-train-identical-future-probes.ps1 `
  -ProbeDirectory artifacts\transfer\probes `
  -Folds all `
  -Arms all `
  -Device cuda
```

Completed jobs are skipped only when their contract, representation source and output hashes still match.

## Per-job artifacts

```text
artifacts/transfer/probes/runs/fold-00/authentic_preference/
├── probe.safetensors
├── validation.predictions.jsonl
├── test.predictions.jsonl
└── run.json
```

`probe.safetensors` contains:

```text
weight
bias
feature_mean
feature_scale
```

The run report contains all five candidate validation results, the selected L2 value, train-prior baseline and final validation/test metrics.

## Verify and aggregate

```powershell
.\scripts\92-verify-identical-future-probes.ps1 `
  -ProbeDirectory artifacts\transfer\probes `
  -Folds all `
  -Arms all
```

The verifier independently:

- reloads every frozen Step 5 validation and test matrix;
- reloads every saved probe and standardization vector;
- reconstructs logits and probabilities;
- verifies every prediction row and future label;
- recomputes all validation and test metrics;
- reproduces the validation-only L2 selection;
- requires one device type and one runtime environment;
- requires exactly one out-of-fold test prediction per episode and arm;
- computes pooled metrics for all seven arms;
- performs the paired lineage bootstrap.

It writes:

```text
probe-verification.json
probe-verification.md
probe-summary.json
probe-summary.md
```

## Interpretation

Possible outcomes include:

1. **Authentic beats generic and controls.** This supports preference-specific representation transfer.
2. **Authentic beats generic but not shuffled/random controls.** Training updates may help, but the effect is not preference-specific.
3. **Language or pair exposure beats generic while authentic does not.** Domain or pair structure transfers, but authentic preference does not.
4. **No trained arm beats generic.** Additional source training did not improve linear future decodability.
5. **Authentic is worse than generic.** Under this contract, preference training degraded the future-relevant representation.

The result must be reported regardless of direction.

## Supported claim after a passing verification

> Seventy identical linear future probes were trained using train-only standardization and validation-only regularization selection, then evaluated once on frozen article-grouped test partitions. Their complete out-of-fold forecasts and paired lineage-bootstrap comparisons were independently reconstructed from persisted matrices and probe weights.

## Results

```text
IMPLEMENTED — AWAITING LOCAL CONTRACT, FULL TRAINING AND VERIFICATION
```
