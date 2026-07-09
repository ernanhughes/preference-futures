# Step 4 publication block — source-task diagnostics

The fixed-budget run produced 60 mechanically valid trained encoders. Step 4 now separates two questions that are easy to collapse into one:

```text
Did the source-task head learn its target?
Did the resulting encoder help predict a later outcome?
```

The first question is diagnostic. The second is the experiment's central transfer test.

Pair exposure learned strongly, demonstrating that the training and validation path can learn a clear source signal. Temporal direction and authentic preference remained close to their class-prior baselines, alongside the random-label and shuffled-preference controls. That weakens the immediate claim that the selected candidate was directly learnable under this contract.

It does not yet establish that the authentic encoder is useless for future prediction. Step 4 therefore retains every mechanically valid preregistered arm, labels its source-task behaviour honestly and freezes the exact encoder paths and hashes for the next stage.

No checkpoint is changed. No weak arm is retrained. No disappointing control is removed.

The resulting manifest contains the untouched generic encoder plus all six trained regimes for each of the ten grouped folds. Future representation extraction must consume that manifest rather than rediscover or select encoders after seeing transfer results.
