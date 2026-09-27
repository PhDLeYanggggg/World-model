# Why Better Forecasts Did Not Yield Better Protected Intervention

This is an exploratory source-development readout on the twelve opened European
localities, not independent confirmation. Obs8/pred12 uses raw stride 12 and
detector-derived silver image-local trajectories. No metric, seconds, human-gold,
physical-safety, true-3D or foundation claim follows. Deployment is unchanged;
Stage5C and SMC remain disabled.

## What Improved

The frozen neural predictor retains its previously verified raw ADE gain of
8.4616% over CV, but its easy degradation is 11.1692%. The new risk policy cuts
the neural all-ADE gain to 0.3342% [0.2547%, 0.4224%], while easy ADE now improves
3.9961% [2.7207%, 5.4116%]. All twelve locality means preserve easy performance;
even the worst single producer/controller/seed locality view loses only 0.6108%,
within the empirical 2% screen. None of the four zero-CV-error queries is switched
in any of its six dependent views. This is a real observed safety improvement,
not an independent guarantee about unobserved scenes.

## What Failed

The same controller recipe applied to damping gives 0.6994% ADE improvement over
CV. Neural versus equally protected damping is -0.3707% [-0.6136%, -0.1252%].
All three seed point estimates are negative; two intervals are wholly negative.
Only one of twelve locality means favors protected neural forecasts. The primary
useful-safe screen therefore fails despite positive gain against the weaker
uncontrolled fallback. Hard ADE gain is only 0.1549%, not a restored large hard
benefit or the historical Stage37 result. Training-constant heads mostly fall
back, demonstrating that the learned controller is not exactly constant fallback,
but that is insufficient evidence of a competitive neural policy.

## Failure Taxonomy

1. **Excess veto and lost useful actions.** Among 1,913,814 dependent row-views,
   the neural all-risk screen vetoes 1,163,354 and the easy screen a further
   103,812. Average locality intervention is 13.25%, versus 30.87% for protected
   damping. There are 1,070,475 known beneficial neural row-views that remain
   unselected. These are oracle-labelled diagnostics, not inference features or
   independent samples. Loosening thresholds after seeing them is not justified.
2. **Underestimated harm among accepted actions.** In one locality/view the
   predicted selected harm ratio is 0.68%, while the realized ratio is 12.91%.
   Strong veto and harmful acceptance coexist. The corresponding worst displayed
   damping ratio stays below 1%. Aggregate easy preservation does not certify
   each accepted action or validate the neural harm magnitude model.
3. **Very little identifiable joint-decision support.** Only 41 of 6,912 neural
   query-views have supported nonadditive edges; joint choice changes only 14.
   The verified matched-count joint-versus-independent ADE contrast is about
   -0.000032%, CI [-0.000553%, 0.000637%]. There is no demonstrated neural coupling
   gain. Lower image-proximity cost does not establish accuracy or physical safety.
4. **Solver fallback can masquerade as a geometry effect.** There are 53 neural
   and 145 damping query-views with a numerical solver floor. Damping's apparent
   joint-versus-unary gain is 0.0368% when these remain in the whole-query table,
   but only 0.000382% on verified matched-count queries. Neither establishes a
   competitive joint policy: joint still does not beat independent selection.
5. **Not an I/O, missing-rollout or collapse diagnosis.** All 108 Torch heads
   completed their fixed budget. The 355 causal features already contain complete
   CV and candidate rollouts, observed histories and neighbors. The forecaster
   is frozen. This experiment neither tests new JEPA representations nor shows
   that simply adding more rollout features or training a larger backbone fixes
   conditional risk under scene shift.

## Implementation Finding

The first post-freeze fallback-reason report disagreed on one boundary row:
its float32 comparison differed from the frozen policy's explicit float64
comparison. The diagnostic was corrected to reproduce the frozen rule and a
boundary regression test was added. No score, decision, trained weight, threshold,
split or empirical metric changed. Full decision replay is the relevant check;
the reporting repair is not a new improved policy.

## What to Test Next

The next intervention-learning contrast should target selected-set harm magnitude
and ranking under source shift, not increase thresholds. Keep this improved bank
and equally protected damping controls fixed. Register a controller-fitting-only
loss/support contrast, with a separate controller-validation role or cross-fitting
inside the controller sources, before reading its new policy outcomes. In
particular, distinguish underestimated tail severity from poor gain ranking using
the saved cost labels and fitting-locality residuals. The registered easy tolerance
must not be weakened. Joint modeling should be revisited only where nonadditive
support is measurable; empty matched controls are not proof of joint success.

The final independent calibration/confirmation pools remain closed. This result
does not repair external evidence simply by renaming opened source localities.
