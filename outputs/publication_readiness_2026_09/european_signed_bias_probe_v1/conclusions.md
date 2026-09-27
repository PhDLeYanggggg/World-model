# Exact Loss-Weighted Bias Exists, but Is Not a Deployment Result

## Evidence

The analytic fit used only each group's two fitting sources and the original
source/query/subset-balanced objective. All other model parameters and every
held decision stayed frozen. Results are fresh analytic fitting, not a new
neural-network training run.

For aggregate-trained heads,91/108 all-risk offsets and106/108 easy-risk offsets
are positive. Their mean offsets are0.007033 and0.003367 in fitting-cost-scale
units. Median fitting objective reduction is0.106876%. The pointwise controls
show the same pattern:92/108 and106/108 positive offsets; median reduction
0.103412%.

Thus the positive row residual cannot be explained solely by reporting rows
while training with query/source weights. A small constant adjustment still
reduces the *exact weighted objective*. This is evidence of a remaining
fitting-score bias, not proof that this bias causes all held failures. The
correction magnitude and loss reduction are small, and global centering may
not address conditional errors or source shift.

The reduction is mathematically guaranteed by the constrained quadratic fit.
It is not model lift, independent calibration, a risk certificate or a novel
learning algorithm. No held labels selected these offsets. No held policy
evaluation was run for this probe.

## Next Controlled Experiment

Use these already-fitted nonnegative offsets as the sole changed factor in a
separately registered policy experiment. Retain the old forecaster, floor,
utility, feature schema,2%risk definition and role split. Compare:

1. The frozen original joint policy.
2. Joint policy using the additive fitting-only signed offsets.
3. Original-score joint selection with the same per-query intervention count
   as the centered policy, so fewer interventions cannot masquerade as better
   allocation.

Freeze every action before readout. Keep both parent head families; do not
select a winner from these existing diagnostic results. Preserve abstention,
unknown labels, all twelve development localities and undefined conditional
risk. Report benefit loss and harm reduction separately. The old structurally
incomplete primary is not repaired by this new experiment or by a new denominator.

This experiment has not been run yet. Deployment remains unchanged. Independent
selection/calibration/confirmation stay closed. No Stage5C/SMC, metric, seconds,
human-gold, true3D, foundation or physical-safety claim. The research goal remains
active and submission readiness is not achieved.
