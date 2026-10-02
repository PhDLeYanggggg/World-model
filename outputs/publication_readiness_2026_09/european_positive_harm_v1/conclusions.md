# Positive Harm Reduces Known Violations, But Does Not Pass

All72 registered conditional harm heads completed, each with an exact refit,
serialized inference replay and readout replay. Original and additive controls
were cached/hash-verified; new harm fitting and validation are `fresh_run`.
Transfer and independent confirmation are `not_run`: the advance screen fails.
No deployment changes. This is a cost-estimator experiment, not new neural
trajectory dynamics or a successful world model.

## What Changed

Only total/easy harm is learned from the seven past-quality features. Frozen
tree routes and raw benefit/reference estimates are retained. A conditional
exponential link preserves training-leaf means and avoids clipping negative
harm estimates to zero. Normalized conditional deviance replaces squared loss.
These two changes form a combined hypothesis, not an isolated positivity effect.
The original feasibility projection is shared; it can change projected benefit.

## Registered Comparison

| Measure across72 fixed heads | Original | Two-harm additive | Positive harm |
|---|---:|---:|---:|
| Selected repeated occurrences |95,455|112,456|111,031|
| Selected unknown outcomes |918|1,143|1,050|
| Complete finite-completion support |33|19|37|
| Defined selected easy-risk ratios |43|61|51|
| Easy-risk upper violations |7|42|11|
| Known-label easy-risk violations |4|20|2|
| Worst easy-risk upper |5.406%|1200.168%|18.028%|

The original2% budget is selected positive harm divided by selected reference
error. Undefined/empty support is not a pass. Worst upper bounds are possible
completions of unknown outcomes, not observed error. Whole-easy net degradation
is a separate quantity and does not override selected-risk failure.

The positive head repairs much of the additive control's known-label failure,
without reducing to the original intervention count. Full support gains8 groups
and loses4 relative to original. But aggregate support37 does not cancel11
upper violations or a worse worst-case upper. At same-query matched counts,
support is27 versus33 and upper violations13 versus7 for original.

## Prediction And Utility

Mean over fixed views within each of12 localities, then equal-locality mean;
3000 paired locality bootstrap draws. Nominal exposed-development intervals,
not independent or multiplicity-corrected confirmation.

| Positive minus comparator | Estimate | Nominal95% interval |
|---|---:|---:|
| Original, normalized signed-score MSE |+0.136161|[+0.039763,+0.257608]|
| Original, full utility % |+0.043821|[+0.019681,+0.073053]|
| Original, matched utility % |+0.022976|[+0.006997,+0.043355]|
| Additive, normalized signed-score MSE |+0.168307|[+0.056014,+0.301117]|
| Additive, full utility % |-0.242154|[-0.453519,-0.080416]|
| Additive, matched utility % |-0.043408|[-0.078341,-0.013621]|

Lower MSE is better; both MSE contrasts worsen. Utility percentages divide
conservative net-gain differences by full known reference mass. They are not
ADE/FDE improvements. Positive utility over original remains at equal counts,
but this does not establish safe selection. Additive has more utility and much
worse risk; neither is selected by trading away the registered risk requirement.

## Loss And Numerical Evidence

- Total-harm normalized training loss:1.735280 ->1.447384.
- Easy-harm normalized training loss:2.812297 ->2.363809.
- Loss includes fixed regularization; mean over heads of training-weighted
  leaf losses averaged over trees, not trajectory loss or directly comparable
  to the additive model's squared training loss.
- All912,895 populated tree leaves reconstructed; max15 Newton iterations,
  gradient<=9.99992e-8, training mean error<=1.44329e-14.
- Zero training-harm leaves:648 total-harm and174,352 easy-harm occurrences.
  They were left unchanged, not assigned synthetic labels.
- Positive validation harm/easy-harm predictions contain zero exact zeros;
  exponent guard activations0. Additive has9,514 negative raw coordinates and
  8,814 zero easy-harm predictions after projection.
- Projection changes benefit on23,824 positive-arm repeated occurrences.
  Raw benefit is frozen; projected benefit is not claimed frozen.

Loss decrease and convergence are real, but do not imply validation calibration.
Signed-score error worsens in10/12 locality means and in all three seed means.
Localities110/124 have especially large error deterioration. The precise
component/tail mechanism requires frozen prediction-level decomposition;
loss misalignment is a supported concern, not yet an isolated causal attribution.

## Remaining Failure

Two known-label violations remain in locality112. The other nine positive-arm
upper violations have known risk at or below2%, but fail after unknown-outcome
completion. The worst case, locality082, has3 selected occurrences including
one unknown: unknown envelope0.109184 divided by known easy reference0.605643
gives18.0277%. Observed known easy harm is0; this must not be reported as
observed18% degradation or silently certified safe.

**Decision: no advance to transfer, no deployment or publication-readiness
upgrade.** Positive-link harm learning is a partial mechanism repair with
negative cost-accuracy evidence. Do not simply loosen its threshold or drop
unknown outcomes. See [failure analysis](failure_analysis.md) and
[next action](next_action.md).

Native arm64 full run1322.59s, peak10.087GB.72 owned CREATE checkpoints total
152,397,838bytes; all rehashed.13,248 scalar/parent checks,526 independent
aggregate/bootstrap checks and37 scoped tests pass. No new local numeric
cache or Slurm jobs. Independent selection/calibration/confirmation stay closed.

Image-local detector-silver, obs8/pred12 rawstride12. No metric, seconds,
human-gold, physical-safety, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off.
