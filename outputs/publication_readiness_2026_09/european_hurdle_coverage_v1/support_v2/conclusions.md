# Risk Ordering Is Not the Same as Intervention Coverage

Status: completed, fresh frozen-score diagnostic; no new training or deployment.
The study separates ordering, intervention count and unequal eligible support.
It does not establish a stable neural advantage or submission readiness.

## What Was Held Fixed

Forecasts, fitted utility/risk heads, training-only preprocessing, source roles,
the 2% predicted-risk rule and the original policy decisions remain frozen.
Each group uses four fitting and eight complete-producer-chain-excluded
localities. All twelve localities are already opened development sources.
There are three folds, three seeds, two candidates and two risk-event targets:
36 groups. Six views per group give 216 views, including all 72 parent controls.

The event target called `easy` describes the risk head's training event. It is
not an oracle easy label supplied at inference. All/easy/hard evaluation
subsets remain defined using evaluation outcomes only, with fitting-only cuts.

The registered first attempt stopped before new outcome evaluation because
two damping/easy groups had unequal causal support. The [amendment](../amendment_support_v2.md)
was committed before the amended readout. It preserves both full original
policies, both common-pool anchors, and both matched-count rankings. It does
not discard the affected rows or assign arbitrary rankings to zero denominators.

## Fixed-Count Ranking Results

Positive values favor hurdle ordering over product-MSE ordering. The table
counts positive/negative conditional 95% intervals among nine fold-seed
comparisons. Intervals containing zero occupy the remaining cells. Both
count anchors are retained, not chosen from their outcomes.

| Candidate / event target | At product common counts: positive / negative | At hurdle common counts: positive / negative |
|---|---:|---:|
| Neural / all | 3 / 4 | 1 / 1 |
| Neural / easy | 2 / 5 | 3 / 4 |
| Damping / all | 0 / 8 | 0 / 6 |
| Damping / easy | 3 / 1 | 2 / 3 |

These are all-ADE contrasts, in percentage points of CV-normalized improvement.
There is no uniform ranking improvement. For neural/all, the ranking component
ranges from -0.3557 to +0.2221 pp at product counts, and from -0.6177 to +0.4047 pp
at hurdle counts. Neural/easy positive points occur in one fold, not all three.
For hard-subset neural/all, the two anchors give 3/4 and 1/5 positive/negative
intervals respectively. The hard subset does not rescue a stable ordering claim.

## Coverage Explains an Important Part of the Tradeoff

With all-event risk heads, coverage components improve all-ADE in all nine
comparisons for both candidates and along both accounting paths. All these
conditional intervals are positive. This arm usually allows more intervention;
the earlier improvement was not simply caused by switching less.

For neural/easy, all nine full-policy all-ADE contrasts are negative, ranging
from -2.5232 to -1.0494 pp. The reduced-coverage component is also negative in
all nine point estimates along both paths, with six or nine negative intervals.
The safety benefit and lost accuracy therefore cannot be interpreted as a
general improvement in which rows are selected.

| Neural easy-event view | Switch range across nine views | Worst positive-easy degradation | Views above 2% | Views harming zero-CV rows |
|---|---:|---:|---:|---:|
| Original product-MSE | 27.40-53.68% | 17.2546% | 4/9 | 6/9 |
| Original hurdle | 1.62-19.13% | 0.6746% | 0/9 | 6/9 |
| Hurdle ordering at product counts | 27.40-53.68% | 13.7083% | 6/9 | 6/9 |
| Product ordering at hurdle counts | 1.62-19.13% | 1.2078% | 0/9 | 5/9 |

Returning hurdle ordering to the original product intervention count restores
substantial easy-case harm. Conversely, reducing product ordering to hurdle
counts satisfies the positive-easy limit in all nine views, but still harms
zero-CV rows in five. This is evidence that intervention coverage matters for
protection; it is not a unique causal attribution or an independently calibrated
safety guarantee. A view with no zero-CV examples is not proof of protection.

All nine forced high-count neural/easy views violate the hurdle predicted-risk
rule somewhere. Such counterfactuals are deliberately offline controls, not
new deployable policies. Predicted-budget compliance is distinct from observed
error preservation, and neither is physical safety.

## Unequal Support Is Retained, Not Hidden

Only two groups have nonzero full-versus-common support components. Their
all-ADE contributions are +0.004401 and +0.000796 pp; easy contributions are zero.
These contributions are small in this readout, but the support mismatch is real.
The stored product reference score is exactly zero on some rows while hurdle's
is positive. This audit does not prove whether underflow, score saturation or
another numerical mechanism caused those zeros.

## Evidence and Decision

All 216 views replay exactly, including 72 parent controls. A separate scalar
sorting and explicit coordinate-error implementation verifies 216 selection
arrays, 864 metric reductions and 108 additive decompositions. There are
227 passing tests across 36 scoped files; the full legacy suite was not run.
See [completion receipt](completion_checks.json), [all results](results.md),
[figure](matched_ranking.svg), [group summaries](group_metrics.json),
[safety summaries](safety_summary.json) and [failure analysis](failure_analysis.md).

Deployment is unchanged. No matched-count arm is promoted. The earlier hurdle
study's failure to establish stable safe neural superiority over protected
damping remains unresolved; this diagnostic does not retrain the predictor.
Next, use fitting-only cross-fitted data to test an ordering-specific objective
and an explicit support/abstention control, each against a matched frozen
controller. Do not loosen the risk budget or select a favorable fold or count
after readout. Reserve independent calibration/confirmation for a method that
has survived these source-development checks.

Released detector tracks, image pixels, observation 8/prediction 12 at raw stride
12. Not historical raw-frame t50, seconds, metric, human gold, true 3D or a
foundation model. Three seeds and 3,000 locality-bootstrap draws are conditional,
dependent across views and unadjusted for multiplicity. Historical Stage37 is
not recertified. Reserved roles stay closed; Stage5C and SMC remain disabled.
