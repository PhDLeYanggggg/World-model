# Why Proposal-Conditioned Supervision Did Not Repair Risk

## Evidence Scope

fresh_run: 108 real neural cost critics, 216,000 primary updates and a fresh
development readout, with deterministic inference replay. cached_verified:
forecasts, public routing models and all-action critics. not_run: new H,
independent final testing, new trajectory dynamics or deployment. The source
is already exposed MovingDroneCrowd++ SOURCE, native 8/12 annotation order and
image pixels. No metric, seconds, true3D or foundation claim. Stage5C/SMC off.

## What Changed

The cost critic was supervised only on actions proposed by a frozen router
trained on a different source partition. The third partition was excluded
from both fits. Forecasts, inputs, model capacity, initialization, optimizer,
normalization, query sampling, 2,000-update budget and 2% risk rule were fixed.
Training proposals were generated before accessing critic-part outcomes.
Unknown selected outcomes retained the original envelope-bound supervision.

This intervention changes effective label exposure. The fraction of fitting
rows with a nonfloor proposal ranges from 19.20% to 97.51%, median 72.81%.
It does not supply more informative history or independent risk calibration.

## Findings

1. **Optimization ran, but lower loss did not imply safer selection.** Across
   108 fits the median recorded minibatch objective falls from 0.60657 at step1
   to 0.14517 at step2000. These are training objectives on different sampled
   batches, not validation loss or a calibration guarantee. All optimizer
   states, own-source preprocessing and initialization/sampling controls pass.

2. **The proposed remedy did not establish a utility gain.** New veto minus old
   veto is -0.2880pp known utility, 95% source CI [-0.8643,+0.2825]. The direct
   missing-outcome lower contrast is -1.0764pp [-2.0174,-0.2208]. The latter
   excludes a positive conservative certificate; it does not identify the true
   unobserved-outcome effect as necessarily negative.

3. **The original risk requirements remain badly violated.** Full support/risk
   passes decrease from 38/696 to 28/696 repeated recording views. Known all-risk
   violations increase 454 to 462; known easy-risk violations 483 to 486.
   Undefined selected-easy views increase 144 to 156. Worst defined signed
   recording easy-net degradation improves from 469.93% to 421.98%, but this
   isolated reduction is not a safe policy. It is not a dataset-average error.

4. **Selection-conditional optimism survives proposal conditioning.** On the
   exact same public proposals, realized/predicted easy-harm mass is 0.0771 for
   the old critic and 0.0682 for the new critic: both overpredict aggregate harm.
   On each critic's admitted actions it is instead 540.55 and 758.34: both
   severely underpredict admitted easy-harm. Overall admitted harm ratios are
   105.88 and 89.65, respectively. These pool repeated diagnostic views, not
   independent observations or estimated harm probabilities. The proposal-pool
   change does not itself calibrate the subset selected by the critic's veto.

5. **Reference-error estimation also matters.** On newly admitted known rows,
   predicted reference-error mass is 107.84 million versus realized 37.37
   million, about 2.89 times as large. A veto based on H <= 0.02R can be too
   permissive when H is underestimated and R overestimated. These are pooled
   pixel-error masses for diagnosis, not a metric-space physical risk measure.

6. **More or fewer interventions is not the explanation by itself.** The new
   source-macro intervention rate is 29.98%, old 29.35%. Against its own query-
   matched count/amplitude control, the new full-strategy secondary effect is
   -0.8753pp [-1.7854,-0.3053] known utility. Both matched and veto policies fail
   original risk rules; this is not evidence that removing the veto is safe.
   Raw pooled selected-row counts and source-macro percentages have different
   weights and should not be interchanged.

7. **Missing support is a separate unresolved barrier.** Eight of276 required
   source views have empty matched utility controls. Thirty of696 repeated
   recording views have undefined full-known easy-net ratios. Those metrics
   remain unavailable. Changing model weights cannot make a missing reference
   denominator observed. There are also hundreds of defined known-risk
   violations, so missingness alone cannot explain the model failure.

8. **Source heterogeneity remains.** Known new-minus-old utility ranges from
   +2.626pp in source055 to -3.962pp in source078. The 3,000-resample analysis
   retains all23source units after context/seed/orientation averaging. No bad
   source, seed, empty view or unknown label is removed to improve the result.

## Interpretation, Not Established Causation

The evidence is consistent with a low-predicted-harm tail being preferentially
admitted despite substantial realized harm, combined with reference-cost
misestimation. A mean-square cost objective and reduced action supervision do
not enforce reliable predictions on that selected tail. Reduced label diversity,
model misspecification and source shift are plausible contributors, not isolated
causal explanations established by this single treatment.

The next useful model test is to target the admitted-tail failure directly,
using only fitting-part crossfit proposals, conservative error uncertainty and
coherent harm/reference estimation. Keep the 2% deployment contract and all
original evaluation/support rules fixed. It must be registered before fitting;
no such next model has been trained in this experiment. Separately, evidence
with actual easy-reference labels is needed to resolve the undefined metrics.

## Decision

Reject this training change as a deployment upgrade. The previous deployed
model is unchanged. Do not call this a new safe dynamics model, independent
generalization evidence or submission readiness. Preserve this negative model
test and the original failed experiments.
