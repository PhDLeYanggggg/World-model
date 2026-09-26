# Failure Analysis: Ranking Survives, Cost Transport Does Not

## Evidence Status

Fresh source-development readouts over cached_verified producer/label
lineage. This is a post-readout diagnosis, not a new confirmatory hypothesis
test, forecast gain or deployment evaluation. Full metrics remain in
`aggregate_metrics.json`; no seed, role or adverse motion-only result is dropped.

## Failure Taxonomy

1. **The expected-cost claim fails.** The primary matched full comparison has
   zero positive intervals out of six. The required original comparison has
   three negative intervals. More favorable secondary ranking does not meet
   the predeclared magnitude objective.
2. **Representation contains information, but the repair is insufficient.**
   Conditional harm-event AUROC improves in five full matched comparisons.
   This does not establish that fractional features contain no useful signal,
   nor does it prove that a larger readout would repair transport.
3. **Training fit does not reliably transfer.** Fractional versus matched
   improves training MSE in 62/72 full views, held MSE in 39/72, and both in
   35/72; 27 improve only on fitting data. Motion-only has 50 fitting versus
   25 held improvements, with 35 fit-only views. These dependent-view counts
   identify a generalization gap, not independent significance tests.
4. **Average coverage hides misplaced cost.** Full coverage medians approach
   one, but held MSE worsens against the original head. Adding predicted mass
   to the wrong rows can fix an average while damaging individual estimates.
5. **Easy-membership spillover persists.** Fractional-readout versus original
   has 42 worsening full views, 36 dominated by outside-easy contributions.
   The largest partition is outside-easy/no-harm in 27, outside-easy/positive-
   all-harm in 9, and easy/positive-harm in 6. Outside-easy rows are not all
   harmless. The positive-excess-share summary is 0.95925, a repeated-view
   descriptive ratio, not a probability or independent estimate.
6. **The control also deteriorates.** Mean-feature refitting worsens 39 full
   views, 34 dominated by outside-easy cost. Both arms' easy-cost learning
   remains vulnerable. A gain relative to a damaged control would be inadequate.
7. **Some localities dominate adverse averages.** For producer0/controller2,
   fractional-versus-original mean gain is -39.789%; locality067 contributes
   -130.868%, while other locality points are -0.869%, -0.128% and -27.290%.
   This post-hoc example identifies heterogeneity, not permission to remove
   that locality or choose a favorable role. Absolute component MSE remains
   available alongside these relative ratios.
8. **No evidence of a runtime explanation.** All real Torch fits completed;
   reference paths, matched sampling and initial losses are checked. No
   resource-induced downscale, NumPy substitution or missing-label training
   explains the negative result. Final replay verifies the actual artifacts.

## What Is Not Established

The nested fraction H_easy/H is not an easy-event probability. Observed
outside-easy squared error does not prove that an explicit easy-membership
model will work. Nor does failure here prove that history, scene imagery or
JEPA/Transformer features are universally ineffective: the frozen features
are hidden units of the existing risk MLP, not a new world-model encoder.

The training/readout comparison changes the parameterization and cost fit
relative to original_mean. Only fractional_features versus mean_features is
the matched representation contrast. Both required contrasts fail to establish
the intended gain, so no new intervention policy is justified.

## Specific Follow-On Diagnostic

Study E = 1(0 < CV_error <= fitting_easy_cut), including easy rows with zero
harm. Measure its predictability and shift separately from conditional harm
inside E. Do not conflate E with the previously attempted event H_easy > 0.
If a new conditional-cost factorization is pursued, use the joint identity
E[H * 1(E) | x] = P(E | x) * E[H | E, x], not a product of marginal easy
probability and marginal harm. This is a next hypothesis, not a proven repair.

Keep the existing endpoint, risk tolerance, source-role separation and
unknown-label rules. No test-driven thresholds, role selection or held-label
features. Independent calibration and confirmation stay closed. Current
image-pixel/annotation-step detector labels do not verify meters, seconds,
physical safety, true 3D or foundation capability. Stage5C and SMC remain off.
