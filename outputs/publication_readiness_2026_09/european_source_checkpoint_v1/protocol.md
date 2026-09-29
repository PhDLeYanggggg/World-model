# Source-Internal Validation Checkpoint Control

## Question and Status

Registered before fitting. Does a checkpoint chosen on the original training
source's held recordings reduce transferable decision-cost error compared with
the final checkpoint of the same training run? This follows the frozen diagnosis
of easy-harm underestimation already on fitting sources, worsened by transfer.
It does not assume early stopping is safe or that transfer shift is the sole cause.

Keep the same 12 already opened development localities, three seeds, 18 producer/
controller contexts and 4/4/2/2 role exclusions. There are 72 single-source fits
and 216 dependent directional readouts. The other fitting locality and both
outer-held localities cannot select checkpoints. Independent model-selection,
calibration and confirmation sources remain closed. This is not a formal primary
replacement, independent calibration or a new trajectory forecaster.

## Only Intended Contrast

Each 380-feature nonlinear head uses the existing 32-wide SiLU architecture,
bounded five-moment decoder, proper loss, optimizer and full 2,000 updates.
Both policies use the same optimization data, initialization, sampled queries,
normalization, support cutoff and complete training trajectory. Compare final
step against the checkpoint with lowest query-balanced validation signed-score
MSE, checked at steps 0,200,...,2000. Ties retain the earlier checkpoint.
Do not stop training early or tune action thresholds. Keeping step0 permits a
negative learning result to be visible, not declared a learned improvement.

Within the original training locality, sort canonical recording identifiers by
SHA-256 of `source-checkpoint-v1|source|recording`; use the first floor(0.7*n)
recordings for optimization, clipped to retain at least one on each side.
Recordings are entirely disjoint. If only one recording exists, use its fixed
70th-percentile unique-current-frame cut with a purge: optimization current frame
< cut-144, validation current frame >= cut+84. This separates the full obs8/pred12
footprints at stride12. An empty side or lack of known labels is a hard preflight
failure, not permission to delete the source or choose another favorable split.
All preprocessing, target scales, RMS weights and support limits use optimization
rows only. Validation outcomes supervise checkpoint choice only. Future validity
is never an inference eligibility feature.

## Readout

Commit all 72 completed checkpoints before freezing all 216 causal action hashes;
commit action hashes before outcome readout. Recompute and compare every action
hash during evaluation. Report final and validation-selected independent policies
and their same-query count-matched policies, using the same original action rule:
positive predicted benefit-minus-harm, all/easy predicted excess <=0, motion and
training support eligibility. Keep the 2% selected-reference risk denominator.
The two arms may have different eligible pools; count matching is not pure ranking.

Primary: validation-selected minus final normalized signed-score MSE, lower is
better. Secondary: matched ADE improvement over final, selected all/easy positive
harm, whole-population easy ADE degradation, hard gain, unknown interventions,
intervention rate, per-locality and seed summaries. Use 3,000 paired locality
bootstrap draws after within-locality averaging. These nominal intervals do not
correct adaptive use of the 12 localities or shared producers. No p-value or
population-safety guarantee. Empty denominators remain undefined, not passes.

## Execution and Evidence

Native arm64 CPU4/interOp1/workers0. Run a 100-update first-fit feasibility pilot,
then resume the same checkpoint to its full budget. Save optimizer, RNG, best
model, final model and validation traces every200 updates. Replay the complete
first fit independently and compare state/selection excluding elapsed time.
Verify the disk reserve before each fit; estimated checkpoint storage is small
and no new row cache is materialized. Preserve completed work after any failure.

Inputs/forecasters are cached_verified; fitting, decisions and readouts are
fresh_run only when actually executed. Frozen-boundary CREATE replication remains
a separate pending verification task, not a training prerequisite or claimed
completed result. Retain all negative results. No new deployment based on this
development control alone. No metric/seconds, true3D, foundation, human-gold,
Stage5C or SMC claim or execution.
