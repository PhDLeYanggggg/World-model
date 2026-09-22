# Fixed Temporal Intervention Versus Displacement-Matched Uniform Shrinkage

2026-09-22. Registered before real training and readout. This is source-only
method development following the failed prefix-risk study, not a new independent
confirmation. No result or calibration guarantee is claimed at registration.

## Hypothesis

The all-prefix guard rejected useful whole trajectories, almost always at the
first predicted step. A less restrictive action may retain the reliable causal
baseline initially and introduce the neural forecast later. Test that temporal
shape against uniform shrinkage with equal causal displacement, rather than
claiming any conservative blend is new or necessarily better.

For the unchanged baseline b and neural forecast p, twelve deterministic weights
are w_k=(k-1)/11, k=1,...,12. Temporal candidate q_k=b_k+w_k(p_k-b_k). First point
is exactly baseline; final point exactly neural. No learned or searched schedule.
The uniform control uses alpha=sum_k w_k ||p_k-b_k|| / sum_k ||p_k-b_k|| and
u_k=b_k+alpha(p_k-b_k); alpha=0 when forecasts coincide. Thus average displacement
from baseline is matched per query, to verified floating point precision.
Alpha uses predictions, never ground truth, observed label length or future mask.

## Unchanged Evaluation and Matched Training

8 observed / 12 predicted annotation steps, SDD raw stride 12, annotation pixels.
Same four development-exposed physical sites, seeds17/29/43, source-excluded held
forecasters and pair-excluded fitting producers. Same complete fitting rows,
source-balanced sampled draws and old frozen 4x emphasis. The loss remains the
existing distance-weighted log composition of benefit, harm and slack. The fixed
strict harm/benefit ratio remains .1, past-stop veto remains, easy ceiling remains
2% for every scene and seed. No new roles, metric, threshold or source admission.

Fit two 45,954-parameter scalar cost heads in each of twelve views: 356 inputs,
hidden128, two outputs, same zero-output seed initialization. 12,000 AdamW updates
per head, lr .001/weight decay .0001, batch256, gradient clip5; CPU4/interop1/
workers0, checkpoints500, heartbeat100. Recompute candidate-derived causal features
and complete-row cost labels. Use the same train-only preprocessing rule for each
arm: means/stds and cost constants may differ because candidate features differ;
CV scale, easy/hard cuts, row support, weights and draw counts must remain identical.
No full forecaster, JEPA or latent rollout training. No test normalization.

## Fixed Readout

Freeze all choices before outcome readout:

1. ramp_strict: temporal candidate's learned terminal-cost strict rule.
2. uniform_strict: matched uniform candidate's learned terminal-cost strict rule.
3. uniform_matched: uniform net-benefit rank at exactly ramp_strict's query count
   per source/seed, causal eligibility only and stable row-id tie break.
4. ramp_at_uniform: temporal candidate at the uniform_strict choice set.
5. uniform_at_ramp: uniform candidate at the ramp_strict choice set.
6. ramp_uncontrolled and uniform_uncontrolled: every past-motion/nonzero-distance
   eligible query. These are diagnostic, not alternate deployment winners.

CV is the no-intervention floor. Also compare the verified unmodified-forecast
scalar-log strict policy, whose preceding ADE gain was 4.18873% with failed easy
protection. Historical Stage26/37 are not recertified by this study.

Primary comparison: ramp_strict minus uniform_strict, equal-site native supported
ADE gain, averaging seed errors (not forecasts). Paired bootstrap3,000 physical
sites. Joint empirical gate requires positive lower primary CI, positive lower
CI versus the old scalar-log strict reference, positive CV gain in each seed,
easy<=2% in every site/seed and seed aggregate, and zero complete exact-zero-CV
harm. Report equal-count and identical-choice crossover comparisons without
promoting secondary winners. Report selected displacement amounts as well as counts;
equal count alone need not equal realized displacement among different queries.

Keep incomplete and unknown labels in their existing support accounting and
full-grid bounds. Report native ADE/FDE, hard/easy, worst site, tails, missing
outcomes, all twelve prefix diagnostic errors, and native second-difference
smoothness including the last two past points. For interactions, reuse the verified
past scene context, radius=median target past scale, threshold=.1 radius, and
baseline-relative proximity cost. Report unsupported context/edges explicitly.
This evaluates a proxy, not joint optimization or physical safety; no fitted
pairwise module or new consistency gate is smuggled into the main comparison.

Four explored sites and repeated seeds do not supply independent calibration or
confirmation. No metric, seconds-level, true-3D/foundation, human-gold, formal
conformal safety or novelty claim. Stage5C and SMC off; deployment false.

## Reproduction

Use `.venv-pytorch/bin/python scripts/run_m3w_temporal_intervention.py` with
`--audit-only`, then `--view coupa_seed17 --arm ramp --stop-at 100`, `--resume`,
`--evaluate`, `--verify`. Run `scripts/verify_m3w_temporal_intervention.py` for
separate label, transform, policy, metric and bound checks. Source/code/config
hashes are fixed before fitting. Large arrays and checkpoints stay local.
