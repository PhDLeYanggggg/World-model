# Cross-Moment Risk Ordering: Two Controlled Comparisons

## Material Passport

Previous turn: progress, completed supported-pair fitting and verification
(b6a1046e). More pairs did not establish neural superiority. A constructed
counterexample exposed a possible mismatch between realized share ordering and
conditional moment risk. This registration precedes all new real fitting.

Hypothesis: weighting pairwise supervision by signed cross-moment costs improves
ordering for deployment risk, and a fitting-fixed loss normalizer avoids the
additional random-ratio distortion of per-batch normalization. This is an
experimental target repair, not a novel ranking architecture or safety theorem.

## Two Ordered Controls, No Outcome Selection

1. Batch mode: replace observed-share difference weights with absolute
   D_ij=H_i B_j-H_j B_i. Compare with the frozen supported-pair head.
2. Fitting mode: retain these cross-moment weights but divide by a fixed
   training-only scale instead of the current batch's sum. Compare with the new
   batch-mode head. Both modes start from the original matched initialization.

This isolates target weighting first, then normalization. Each comparison has
36 new heads and 216 full/common/matched-count views: three folds, seeds
17/29/43, neural/damping candidates, all/easy risk events. Total: 72 new Torch
heads, 144,000 updates, 432 views. Retain every arm; never choose a winner from
new outcome metrics. Both complete decision banks must exist before either
mode can run new evaluation. The fitting-mode decision check uses the batch
mode's causal decision archive, not its outcome report.

Keep 355 causal features, width 64, 22,979 parameters, initialization, optimizer,
2,000 updates, minibatch draws, fitting-only feature scalers, forecasts, utility
heads, risk coefficient 1, log epsilon 1e-6 and the 2% deployment budget fixed.
Original moment MSE, occurrence BCE and positive-severity MSE do not change.
There is no extra training, new dynamics fit, threshold search or calibration fit.

## Target and Scale

Use supported-event-first cyclic pairs within each original minibatch/locality.
Supported means B+H>0. Undefined rows stay in the original losses. Compute
cross-products in float64 to avoid float32 product overflow. Skip exact zero
cross-products; reject negative, NaN or infinite targets before pairing. Minor
tie differences from the old float32 realized share are numerical consequences
to report, not an extra label filter. Never clip large pair weights silently.

Rank the same predicted log(Hhat+epsilon)-log(Bhat+epsilon). The loss numerator
is sum abs(D_ij)*softplus(-sign(D_ij)*(score_i-score_j)). Batch mode divides by
the current sum of absolute cross-products. Fitting mode divides by the mean
batch sum from the first 40 unchanged fitting batches, generated using a cloned
sampler RNG. The scale is computed before fitting and held fixed in checkpoints.
No new draws are consumed by optimization. An empty current pair set contributes
zero ranking loss. A nonfinite or zero fitting scale is an explicit blocker.

Audit fitting-only weight sums, quantiles, largest pair share and effective pair
count before training. Large concentration is reported, not used to choose a
coefficient or clip cutoff. The existing gradient-norm limit of 5 remains. Stop
on nonfinite loss/gradient and preserve the checkpoint; do not silently change
the registered objective. Original and fixed fitting-batch traces are retained.

Under independent conditional draws, E[D_ij|x_i,x_j] has the sign of
E[H|x_i]/E[B|x_i]-E[H|x_j]/E[B|x_j] when expected denominators are positive.
The fixed normalizer avoids dividing by a random realized batch weight, but
neither observation proves consistency here: repeated/overlapping rows, correlated
agents, event-conditioned pairing, finite scale estimation and model limitations
remain. No IID, calibrated-risk, physical-safety or novelty guarantee is claimed.

## Evaluation and Verification

For each mode, retain original control/treatment policies, common-pool anchors,
treatment at control counts and control at treatment counts. Frozen causal
decisions precede all new outcome evaluation. Compare all/easy/hard ADE, FDE,
tail/worst-locality errors, zero-reference harm, intervention count, predicted
budget violations and direct neural-versus-equally-protected-damping performance.
Use both ordering/count decompositions with the same CV denominator. Report
3,000 paired locality-bootstrap resamples with the unchanged seed.

All 36 controls in each mode must exactly reproduce their predecessor. Replay
all 72 checkpoints on the first 4,096 excluded-index rows each; replay all 432
metric views. Separate scalar sorting and coordinate arithmetic must verify the
full matrix. Fixed-scale normalization and batch draws are checked against their
fitting-only receipts. Tests cover the analytic counterexample, finite gradients,
locality boundaries, support/unknown labels, unit scaling, resume and the barrier
requiring both decision banks before new readout.

Success is consistent held-out-from-fitting ordering improvement and neural
advantage over equally protected damping, with positive-easy and zero-reference
preservation. A lower fitting loss, favorable seed or weakened damping control
is not sufficient. Count-forced views are offline diagnostics, not deployable
policies. No model promotion is authorized by source-development results alone.

## Execution and Limits

Native arm64 CPU4/inter-op1/workers0. Run a real 100-step pilot in each mode,
then resume to the full registered budget. Atomic checkpoint and heartbeat every
200 steps, private logs and process IDs. Prior whole fitting phases took minutes;
local execution is proportionate. Disk below 10 GiB or nonfinite arithmetic is
a hard stop, not a reason to relabel an incomplete run. No remote job is changed.

All twelve European Squares localities are opened development. Each fit uses
four fitting/eight complete-chain-excluded localities, not independent final
confirmation. Three seeds and locality bootstrap are conditional, dependent and
unadjusted for multiplicity. Detector-track image pixels, obs8/pred12 rawstride12;
not historical t50, seconds, metric, human gold, physical safety, true 3D or
foundation evidence. Historical Stage37 is not recertified. Reserved roles stay
closed. Deployment unchanged, Stage5C off, SMC off, submission readiness false.
