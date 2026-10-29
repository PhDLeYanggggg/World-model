# Inner-Fit Census: Pooling Worsens Fitting and Does Not Repair Transfer

Fresh_run: all72 fixed inner-fit views completed after the same-code resume,
using cached_verified108 final checkpoints. Zero optimizer steps and no new held
inference occurred. The held comparison uses hashed aggregates from the completed
readout. The successful resumed process took77.52s; that is not the total wall
time including the first SSH255 interruption. Both training and held readout
had already completed before this diagnostic began.

Each arm includes897,060 repeated inner-fit row occurrences, including18,978
unknown outcomes. Every row remains in inference. Combined with the419,766 held
occurrences, the split accounts for1,316,826 original TRAIN occurrences per arm.
These are repeated views, not independent examples or datasets.

| Arm | Fit known easy violations /72 | Held known easy violations /72 | Defined fit pass, held fail /72 | Fit mean signed MSE | Held mean signed MSE | Fit / held completion support |
|---|---:|---:|---:|---:|---:|---:|
| Local |1|46|43|0.559739|4.013844|5 / 0|
| Shared global |10|46|37|1.968616|11.859315|1 / 0|
| Shared source |9|40|33|1.965634|7.452018|7 / 1|

Known easy ratios are undefined on7/6/7 fit views and7/8/12 held views, in the
same arm order. Undefined is not a pass. All72 paired MSE ratios are defined.
Their medians are6.698272 local,2.267628 shared-global and2.089436 shared-source.
The smaller shared held/fit ratio is not improved predictive performance: its
TRAIN denominator is much worse and absolute held error is also worse. The
registered shared-minus-local held MSE intervals are both above zero.

## Supported Interpretation

1. Local fitting and recording transfer are different problems.43/72 local views
   pass the known fit calculation but fail on held recordings. Passing known
   TRAIN risk is not a safety or generalization guarantee.
2. Pooling adds a fitting problem. Both shared models have about1.97 mean TRAIN
   signed MSE compared with0.56 for local heads, along with more known fit-risk
   violations. It is not justified to attribute every shared failure only to
   out-of-recording shift or inadequate training sample count.
3. Separate source constraints modestly change violation counts but do not pass
   the utility, prediction or full-support requirements. Unknown outcomes and
   defined known-label violations are separate limitations.
4. This comparison does not uniquely identify architecture capacity, normalized
   source aliasing, decoder priors or risk-objective conflict. The pooled head
   has one32-wide parameter set rather than four local sets; local and shared
   priors/RMS also differ. Those factors require controlled isolation.

## Next Repair Boundary

Retain all failures and the fixed recording partitions. Before another outer
development readout, test whether a shared encoder with source-specific decoding
preserves the local conditional cost maps, against shared and local controls at
a declared parameter and sampling budget. First test matching and causal input
invariance; do not merely increase the penalty or search held thresholds.

The alternative explanation is that risk weighting sacrifices cost accuracy.
A matched no-risk cost objective can isolate that factor inside the same TRAIN
boundary, without changing the2% evaluation requirement. Which repair to advance
must depend on prespecified TRAIN-internal contrasts, not a favorable source or
seed. Neither proposed repair has been trained in this census.

No independent role was opened. The source corpus remains previously exposed,
so the inner partition does not become a new independent benchmark. Not new
world dynamics, not metric/seconds, not true3D or foundation. Stage5C and SMC
remain disabled; no model is promoted.

```bash
.venv-pytorch/bin/python -m scripts.diagnose_m3w_shared_train_fit check
.venv-pytorch/bin/python -m scripts.report_m3w_shared_train_risk --check
```

[Machine-readable census](summary.json), [held evidence](../readout/conclusions.md).
