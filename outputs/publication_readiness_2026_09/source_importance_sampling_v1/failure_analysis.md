# Objective Repair, Not A New Dynamics Result

## What Succeeded

The single-factor comparison restores the original expected training loss while
retaining the episode sampler's exact exposure. All 24 fresh heads complete
240,000 updates. Four-fold expectation checks, exhaustive small-batch tests,
exact resumed fitting and real forecast replay pass. Previous models are kept.

Geometry improves from -37.3268% to -0.0321% versus stationary CV; centered
appearance from -54.9917% to -0.2746%. Differences against the uncorrected arms
are +37.2947 and +54.7171 percentage points, with conditional four-site intervals
[+26.0700,+52.3035] and [+39.0737,+79.6961]. This controlled loss intervention
strongly supports objective mismatch as a cause of the large harm increase in
the preceding sampler experiment. It is not a complete explanation of the
remaining representation/generalization failure.

Static-target absolute harm falls from0.576017 to0.000639 annotation pixels for
geometry and from0.765669 to0.007827 for centered appearance. Harm is disclosed,
not rounded to a claimed zero. Percentage degradation against zero-error CV is
undefined and cannot certify a2% gate.

## What Still Failed

All24 new held fits remain negative. Geometry's aggregate interval is
[-0.05775%,-0.01284%]; centered appearance's is[-0.68041%,-0.03179%]. None beats
the baseline. Nonzero-target gains remain negative at -0.00715%/-0.01826%.
Geometry hard-slice gains are negative at all four sites. Centered hard gains
are tiny, mixed-sign values, not a robust dynamics contribution.

Against original uniform sampling, geometry gains only0.03832pp with interval
[-0.00078,+0.10872], crossing zero. Centered appearance reduces harm by0.48757pp
with interval[+0.08462,+1.06884]. It still loses to corrected geometry by0.24254pp,
interval[-0.62477,-0.01895]. Better than a failing visual control is not proof
that image information adds useful prediction.

Per-candidate binary oracle headroom is only0.002304% for geometry and0.064806%
for centered appearance. These future-informed oracles are diagnostics, not
deployable selectors. Another threshold sweep on the same candidates cannot
manufacture a large forecasting benefit. Reusing explored outcomes for selection
would weaken evidence further.

## Optimization Versus Information

Geometry's original uniform training gain ranges from-0.02167% to-0.01013%;
centered training gains range from-0.02265% to+0.01254%. The new loss no longer
silently optimizes the previous reweighted risk. On every saved predictor its
exact corrected expected training loss matches the uniform loss.

Every logged gradient norm is above the unchanged cap5. Logs sample update1
and every100th update, not every gradient. This is an observed conditioning
diagnostic, not proof that clipping is the cause of failure. The unbiased
importance identity covers loss and unclipped gradients, not clipped gradients
or AdamW updates. Importance-weight second moments are1.825-1.989; their
asymptotic ESS fraction0.503-0.548 describes sampling variance, not independent
scene/event support. No clipping threshold or learning rate was changed after
reading these outcomes.

The previous raw audit remains relevant:207 larger-excursion windows correspond
to55 annotation episodes/47 scoped tracks, and206 already have moving neighbors.
Observation availability does not establish a transferable predictive cue.
These results do not prove that stationary-start prediction is unlearnable,
that every visual representation is ineffective, or that training longer can
never help. They do rule out declaring success for this fixed repair.

## Next Evidence Needed

Keep the sampler correction as a verified training option, not a claimed method
contribution. Do not launch more threshold sweeps or repeat this matrix unchanged.
Before another training factor changes, use training-only gradient diagnostics
to separate full-population direction, static/nonzero contributions and
minibatch clipping effects on these frozen models. This audit has not run here.
It should distinguish an optimization bottleneck from a candidate-utility gap;
do not assume that reducing tiny static jitter yields useful dynamics.

A candidate must then demonstrate transferable gain, with its own useful past
information and independent event support, before fitting a new risk head or
opening another evaluation role. New scientific roles, main metrics or risk
budgets are not authorized by a negative source experiment.

## Evidence Boundary

Fresh:24 corrected fits, exact expectation/replay/OOF checks and analysis.
Cached_verified:48 original control heads, image features and event group mapping.
Not_run:main/t+50/external/Stage37 comparison, independent calibration or final
confirmation, new deployment. Four explored sites/shared training folds and
2,000 conditional bootstrap draws are not independent confirmation.

All required processes ended normally. Fitting673.584seconds, main-log span
681.766seconds. Nativearm64CPU4/inter-op1/workers0;37scopedtests pass. Completed
resume adds0updates and preserves84artifacts. Full legacy suite not rerun.
No CREATE job; its old access status was not freshly resolved.

Still offline annotation-step/pixel-space 2.5D research, not sensor-as-of, metric,
seconds-level, true3D, foundation or submission-ready. No Stage5C execution/SMC.
