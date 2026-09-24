# Query Risk Accounting: Mechanism Found, No Policy Promotion

## Result Passport

This is a complete fixed development comparison, not new predictor training.
All 175,756 past-eligible windows, 20,932 recording/frame queries, three seeds
(17/29/43), and three frozen actions are retained. There are 188,388
query/action/seed instances, seven newly recomputed policies and four retained
controls per action, giving 33 action/policy rows. Fitted risk heads, forecasts,
labels and their producer chain are cached-verified; the new allocation and
readout are fresh. No threshold or model was selected from these outcomes.

Protocol: 8 observed / 12 predicted native SDD annotation steps, stride 12,
annotation pixels. These are not historical t+50 results. The four physical
sites have been used for method development; excluded-site fitting does not
make them independent confirmation. ADE is available for 172,957 rows, including
partially labelled futures; 2,799 rows have no evaluable future. Those rows remain
in the inference population and are not classified as safe.

## Fixed Factorial

`q` denotes predicted expected net easy harm and `r` the predicted easy-CV
denominator. The budget is fixed at 2%. Every optimization stays inside one
recording/frame. Population rules use all targets' denominator; selected rules
use only intervened targets. Clipped rules replace `q` by `max(q,0)`. Clipping an
expected net loss is not the same as predicting expected positive harm.

All gains below are equal-physical-site percentages over CV after averaging
three seeds. Easy degradation is the worst physical-site/seed positive-CV easy
subset, truncated at zero. Zero-CV harm is a count of repeated row/seed instances,
not unique people. [All controls, CIs and switch rates](results.md) are retained.

| Forecast | Risk / denominator | ADE gain % | Hard gain % | Worst easy degradation % | Zero-CV harmed |
|---|---|---:|---:|---:|---:|
| Damping | Signed / population | 1.094416 | 0.852434 | 0 | 0 |
| Damping | Clipped / population | 0.870828 | 0.434909 | 0 | 0 |
| Damping | Signed / selected | 0.755697 | 0.352249 | 0 | 0 |
| Damping | Clipped / selected | 0.556687 | 0.031335 | 0 | 0 |
| Transformer | Signed / population | 2.939203 | 2.434926 | 0.865803 | 3 |
| Transformer | Clipped / population | 2.273613 | 1.435497 | 0.367567 | 2 |
| Transformer | Signed / selected | 2.182812 | 1.628906 | 0 | 1 |
| Transformer | Clipped / selected | 1.395384 | 0.595417 | 0 | 0 |
| EqMotion | Signed / population | 2.826836 | 1.680746 | 1.838357 | 6 |
| EqMotion | Clipped / population | 1.983687 | 0.474680 | 0.769668 | 0 |
| EqMotion | Signed / selected | 2.261437 | 1.218582 | 0.269947 | 5 |
| EqMotion | Clipped / selected | 1.299489 | 0.049721 | 0 | 0 |

## What the Comparison Establishes

**Both mechanisms matter.** Removing negative predicted-risk credit eliminates
EqMotion's six zero-CV harms but leaves two Transformer instances. Removing
unselected denominator reduces, but does not eliminate, either predictor's harm.
Applying both restrictions eliminates observed zero-CV harm in this population.
Population risk accounting is a different estimand from selected-agent risk; it
is not itself future leakage or a software bug. Neither estimand is individual
protection.

**Removing the subsidy also removes useful intervention.** Relative to signed
population allocation, the double restriction loses 1.543818 percentage points
of Transformer ADE gain (nominal paired CI95 [-2.335211, -0.752425]) and 1.527347
points for EqMotion ([-2.599747, -0.569369]). Switch rates fall from 19.5843% to
6.4648% and from 13.7124% to 4.4391%, respectively. Hard-subset gain nearly
disappears for EqMotion. This is not an improved overall method.

**The restricted policy also trails the old strict control.** Transformer loses
1.041443 points versus old strict (CI95 [-2.417534, -0.124792]); EqMotion loses
0.309816 points ([-0.474012, -0.038846]). Old strict's ADE gains are 2.436827% and
1.609305%, with worst easy degradations 1.066900% and 0.452161%, and no observed
zero-CV harm. This comparison does not justify replacing the existing rule.

**Coverage is not the whole explanation.** At exactly the double-restriction's
query-specific intervention count, the signed-population control gains 0.459581
points for Transformer (CI95 [0.175493, 0.756344]) and 0.226183 for EqMotion
([0.084754, 0.387777]) over the double restriction. Counts match in every query.
These controls also avoid observed zero-CV harm; EqMotion retains 0.208981% worst
easy degradation. This shows a tradeoff in which targets are admitted, not just
how many. It does not license selecting this control for deployment after the
readout. All three matched alternatives remain reported.

These intervals use 3,000 paired physical-site resamples. They are nominal,
descriptive intervals on four development-exposed sites, not multiplicity-adjusted
confirmatory tests or evidence of new-scene safety. The earlier negligible
nonadditive interaction effect and lack of proof that neural forecasting is
indispensable remain unresolved.

## Failures and Missing Support

The same-rule recomputation differs from the retained parent in 6/26/16 repeated
row/seed decisions for damping/Transformer/EqMotion, across 3/13/8 queries. Primary
ADE changes are only -0.00005186/+0.00000741/+0.00001291 points. These numerical
differences are disclosed, not interpreted as a modeling mechanism.

Two EqMotion policy solves fail the original-unit feasibility check at the same
`hyang/video4`, frame 4296, seed 43 query and fall back to CV. Neither risk
allowance is relaxed. An outcome-based superset calculation bounds each failure's
possible supported-ADE primary-score effect below 0.000948 percentage points;
this is an impact bound, not an optimal repair or a bound for missing futures.
See [numerical impact](numerical_impact.json).

Across seven new policies there are 74,410 query/policy instances without verified
canonical optimality, including the two failed fallbacks. The remaining cases
are feasible numerical solutions whose optimality for the original floating-point
inequality is not certified. Feasibility, optimality, and statistical safety are
three separate claims. No failed reference or count mismatch is hidden.

Even the double-restricted Transformer intervenes on 143 repeated row/seed
instances with entirely unknown future error and 2,741 with incomplete future
support; EqMotion counts are 73 and 1,498. Consequently, observed zero harm cannot
be read as zero full-population harm. Partial-future gain bounds, complete-label
comparisons, per-site tails and every seed remain in [analysis.json](analysis.json).

## Research Decision

Do not promote a new policy or loosen thresholds. The study explains why pooled
expected-risk accounting can admit both useful and harmful interventions; it does
not solve conditional risk estimation. The exact-past-CV veto rejected in the
[preceding diagnosis](../zero_reference_support_v2/report.md) remains unjustified
as a repair: none of the four uniquely harmed windows has exact past CV.

The next priority is admissible independent physical-site support and a frozen
calibration protocol for a precisely defined loss, including the zero-reference
stratum. Do not add another exposed-outcome threshold sweep or architecture merely
to recover the lost score. More capacity is not supported by this mechanism test.
The [literature note](literature_and_claim_limits.md) distinguishes this empirical
study from conformal guarantees and the remaining novelty requirements.

No new neural training or loss curve is claimed; the reused heads' real losses
remain in the [preceding training report](../net_easy_moment_guarded_v1/training_losses.md).
Independent calibration and confirmation were not run. DroneCrowd stays closed,
DUT remains exposed diagnostic, and HT21/CroHD remains quarantined source audit.
Stage5C/SMC remain off. M3W is still not true 3D, metric, verified-seconds,
foundation-scale, independently safety-certified or submission-ready.
