# Seven-Arm Readout Frozen; Real Comparison Not Run

## Material Passport

Mode: experiment implementation and engineering verification.
Result: fresh code/tests/admission checks, not trained model performance.
Registration commit: `4b842ae2`.
[Protocol](protocol.md), [machine-readable checks](engineering_verification.json).

## What Is Ready

The reader compares original forest, additive quality, Poisson harm, squared-cost
harm, and the three new neural cost heads (none, rowmean, temporal). It verifies
all 216 final checkpoints before accessing development predictions. It retains
the original predictor/producer chain, recording splits, fixed 2% risk rule and
all four strong controls. Original controls must reproduce their old hashes and
metrics. Missing controls cannot be silently omitted.

Actions never depend on outcomes or future availability. Readout includes all
known-label and original-selected signed-score MSE; full and same-query-count
utility; intervention/unknown counts; known and completion-upper positive harm.
In addition to the existing difference-of-lower-bounds proxy, paired completion
tracks the same unknown outcome on both policies: shared selections cancel,
while exchanged selections retain their disagreement-envelope uncertainty.

The bootstrap retains 12 locality blocks, not 72 independent heads or hundreds
of thousands of independent windows. Undefined support in even one required
view invalidates that contrast; it is not dropped to manufacture a finite CI.
Three cost-head seeds are not three new trajectory-forecaster trainings.

## What Was Actually Run

- 28 new readout tests passed in 4.71 seconds.
- 56 combined trainer/temporal-target/readout tests passed in 4.91 seconds.
- Synthetic Torch optimization, checkpoint loading and seven-arm readout were
  exercised together. This is not a real-data validation experiment.
- Scalar implementations independently checked decisions, query tie-breaking,
  weights, moment ratios, paired unknown completion and 3,000-draw locality CI.
- The full historical test suite was not run; these are scoped engineering tests.
- All 149 registered source bindings and 72 source/head identities were resolved.

The real readout admission check reports `readout_allowed=false`: no completed
training freeze exists, no real neural checkpoints exist, and no new validation
predictions were loaded or scored. There are no new model-performance numbers.
Zero receipts are not zero error or a null-effect estimate; evaluation is
`not_run`.

## Current Resource Blocker

At 12:07 UTC on October 5, local free space was 8,961,761,280 bytes, versus the
registered 11,007,950,848-byte reserve plus checkpoint allowance. The shortfall
was 2,046,189,568 bytes (about 1.91 GiB). Available space has changed since the
earlier 1.67 GiB observation. Retain the original 10 GiB reserve and recheck
immediately before fitting; roughly 3 GiB additional free space would give some
headroom over this latest shortfall. No unrelated files were removed.

The authorized read-only CREATE retry at 12:05 UTC timed out after 20.05 seconds.
The simulation chat has no newer completion message. Current remote ownership,
quota, environment and job state are unknown. No job absence/failure is inferred,
and no submission, cancellation or other project modification occurred.

## Next Scientific Action

Restore checkpoint-safe storage or stable owned CREATE access. Then run the real
three-arm pilot with exact resume replay, use measured compute cost to place all
216 fits, commit their final freeze, run this readout and its exact replay. Keep
negative comparisons visible; lower auxiliary loss alone cannot support a new
policy. No additional threshold search or independent-source readout is allowed.

This advances experiment readiness, not model quality. There is no deployment or
submission promotion. Image-local detector-silver obs8/pred12 rawstride12 only;
no metric, seconds, human-gold, physical-safety, true-3D or foundation claims.
Stage5C and SMC remain off.
