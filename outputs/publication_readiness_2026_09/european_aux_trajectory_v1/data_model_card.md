# Training-Trajectory Data and Model Card

## Intended Use

Diagnose the training-time behavior of the already exposed European source
cost-head experiment. This is not a new trajectory forecaster, deployable
selector, independent replication, or an independent test of generalization.
The final states must match the original strong-cap heads exactly.

## Inputs and Labels

The original 383-dimensional fitting feature bank, localities, row identifiers,
normalization and cost scales are hash-checked. Nested row-locality-excluded
four-component cost labels and cap-event labels are supervision/diagnostic
targets only. Unknown targets remain masked; zero-envelope known costs remain
in the original sampler. The true/shuffled auxiliary comparisons keep exactly
the same fitting features, initial weights and sampled row streams.

Only exposed source fitting labels enter updates and diagnostics. Shared
source loaders contain multiple source roles, so this is not a claim of
filesystem blinding. Fitting IDs are applied before target indexing. No new
outer outcome readout is run. Independent model selection, reserved risk
calibration and confirmation are not opened. No future endpoint, central
official velocity, test-endpoint goals or test normalization is added to inputs.

## Estimator

The unchanged GELU64 cost head has the original bounded nonnegative four-cost
outputs, event-logit auxiliary head, AdamW optimizer and gradient clipping.
Arms are cost-only, true cap-event auxiliary and locality-shuffled auxiliary.
This study does not introduce projection, a new loss weight, checkpoint
selection, threshold fitting, residual generation or policy intervention.

All 144 fitting views, three arms, seeds 17/29/43, full/motion inputs and six
producer/controller assignments are retained. Measurements occur only at
updates 200/600/1000/1400/2000. The original first pilot's extra step-100 trace
entry is retained as a logging exception; prescribed numerical entries and
final model/AdamW/sampler/Torch-RNG states must still match exactly.

## Statistical Interpretation

Severity thresholds use positive fitting easy-harm targets and the original
equal-locality weights. Zero-harm rows and every severity bin are separately
reported. Empty strata are unsupported. Additive SSE accounting measures where
the observed fitting-error difference accumulates, not why it is caused.

Three seeds and three fitting contexts are averaged within each locality.
The four source localities are resampled 3,000 times for descriptive intervals.
Contexts, models and assignments overlap. Neither windows, diagnostic batches,
seeds nor snapshots are independent scene replications. Inspecting five fixed
times cannot identify exact causal onset. No model is selected from these data.
Intervals are unadjusted across times, strata and assignments and are not
formal efficacy tests. Apparent curve differences require a separately
registered intervention before a mechanism or repair claim is justified.

## Runtime and Claim Boundaries

Native arm64 PyTorch, CPU4/interop1/workers0, resumable atomic checkpoints,
per-snapshot hashes and heartbeat. Raw data and model states remain private.
The current protocol is obs8/pred12 native annotation steps in detector pixels.
No seconds, metric calibration, physical-safety guarantee, human-gold label,
true3D or foundation-model claim. Stage5C and SMC stay disabled.
