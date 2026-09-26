# Auxiliary Gradient Diagnostic: No Supported Projection Repair

I tested the gradient-interference explanation before investing in another
full training sweep. The registered repair screen failed in all four parts.
I am not promoting gradient projection, changing deployment, or opening an
independent evaluation role on the basis of these results.

## What Was Actually Run

Fresh computation: 17,280 isolated AdamW updates from 432 frozen step-2000
models, across 144 views and three seeds. There were 3,456 final-state batch
diagnostics and 144 initial checks. Each comparison started from the same
model and optimizer moments. Frozen fitting features, nested targets and
checkpoints were hash-verified. This is not 432 newly trained models.

Results were frozen in 9e676e03 before aggregate readout. A display-ID fix and
a NumPy serialization fix are explicitly recorded, with the original
registration retained. Neither changed the numerical experiment or decision
rule. No failed run or adverse comparison was dropped.

## What the Results Say

For full-input, previously cap-supervised models, the main four-cost gradient
conflicts with the true auxiliary shared gradient in 120/576 sampled batches
(20.83%). Easy-harm gradient conflicts occur in 63/576 (10.94%). These are
dependent batch counts, not independent observations. Median shared cosine
is positive: 0.0880 for cost4 and 0.1723 for positive-envelope easy harm.

Among the 120 cost-conflict batches, projection lowers actual probe cost in
64 and raises it in 56. For positive-envelope easy harm, it lowers error in
54 and raises it in 66. Correcting a local raw-gradient conflict therefore
does not consistently improve actual finite-step AdamW probe behavior.
This does not isolate AdamW, clipping, curvature or probe variation as the cause.

| Full-input cap-auxiliary state comparison | Metric | Positive / negative / overlapping locality intervals |
|---|---|---|
| Projected true vs true auxiliary | Four-cost MSE | 0 / 0 / 6 |
| Projected true vs true auxiliary | Positive-envelope easy-harm MSE | 1 / 1 / 4 |
| Projected true vs projected shuffled | Four-cost MSE | 0 / 1 / 5 |
| Projected true vs projected shuffled | Positive-envelope easy-harm MSE | 1 / 2 / 3 |

Projected-versus-unprojected easy-harm point changes range from -0.00024568%
to +0.00007107% in these one-step probes. These tiny fitting-loss changes
are not trajectory gains and must not be extrapolated to a full run.
All motion-only results and the other frozen arms remain in the aggregates.

## What This Does Not Establish

The diagnosis uses already exposed fitting populations. Disjoint update/probe
row IDs do not make windows or probes independent. Four-locality bootstrap
intervals after seed/context averaging describe fitting variation only.
No held-scene generalization, calibrated risk control or independent test
was measured. Final-state gradients cannot establish early-training behavior.
The 144 initial shared cosines are not estimable because output weights start
at zero, not because task agreement or representation collapse was proved.

The earlier auxiliary-cost failure remains. Gradient conflict exists locally,
but the current evidence does not support it as a sufficient explanation or
support the tested projection as a repair. See failure_analysis.md and
project_gap.md for the next discriminating experiments.

Obs8/pred12 native annotation steps, detector pixels. No metric/seconds,
physical-safety, human-gold, true3D or foundation claims. No Stage5C or SMC.
M3W is not yet a defensible submission-ready world-model result.
