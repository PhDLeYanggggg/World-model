# Whole-Recording Optimization: Better Fitting, Failed Risk Transfer

## Result

I trained 36 new neural intervention heads, with 72,000 optimizer updates, to
test whether exact recording-level statistics repair the previous stochastic
policy. Full-recording gradients recover useful decisions relative to the
stochastic method, but the complete policy fails the unchanged 2% requirements
on held recordings. There is no deployment change or submission-readiness claim.

This is policy learning between frozen forecasts, not a newly trained trajectory
forecaster, generative world model or scene-joint inference algorithm.

## Controlled Comparison

Each head has 12,385 parameters. Architecture, initialization, causal inputs,
training roles, 2,000 updates, risk budget and final decision threshold are fixed.
Three head seeds and six producer/controller contexts produce 18 fits per arm.
All 36 final states were frozen before any new inner-held scoring.

- **Stochastic:** the verified previous hard-forward policy, retained unchanged.
- **Exact dual:** the same sampled primal row stream, with full-record risk
  statistics for the dual update.
- **Exact gradient, primary:** full-record statistics for both updates.
- The original frozen moment-based policy and no-intervention floor remain
  explicit references. The original policy is not certified safe by this study.

Exact gradient sees 1,789,950,000 training-row gradient occurrences versus
56,832,000 sampled primal occurrences for exact dual and its stochastic control.
The experiment is update-matched, not compute- or example-exposure-matched.
We cannot isolate reduced estimator variance from greater gradient exposure.

## Main Findings

| Policy | Fit budget views | Held budget views | Held known all/easy violations | Held easy-degradation violations |
|---|---:|---:|---:|---:|
| Original | 1/72 | 0/72 | 61/44 | 61 |
| Stochastic | 35/72 | 4/72 | 43/50 | 27 |
| Exact dual | 33/72 | 2/72 | 45/44 | 25 |
| Exact gradient | 53/72 | 0/72 | 64/60 | 49 |

Primary recording-view budgets pass 178/444 on fit and 2/186 on held data.
All three head seeds have 0/24 held view budgets; this is not one bad seed.
Six fit and three held easy-risk ratios are undefined and do not pass.

On held recordings, primary utility versus stochastic is bounded by
**[+0.377395, +1.916356] percentage points** for the full policy and
**[+0.073468, +0.171986]** at the same intervention count. Both are positive,
but the policy fails safety. Against the original reference, full utility is
**[-4.226985, +0.851822]** and matched utility is
**[-1.517747, -0.628100]**. Every head-seed matched upper bound is negative.
Seven primary/original matched views are empty; their risk is not estimable.

These are deterministic partial-label completion bounds, not sampling confidence
intervals or relative ADE/FDE improvements. Each view uses its full known
reference-error mass, followed by equal-view averaging. No favourable secondary
arm substitutes for the primary result.

## What Improved and What Did Not

Exact gradients retain 59.36% of the original known benefit and 28.53% of its
known harm on fit data. On held recordings those fractions become 52.96% and
52.60%: useful harm discrimination does not transfer. Primary held interventions
rise to 56,720, including 869 unknown outcomes. The added activity is not a
safe improvement. Merely fixing the dual estimate does not help: its matched
held utility versus stochastic is [-0.050685, -0.028776] points.

The earlier diagnostic was real: 256/277 positive empirical constraint
occurrences looked nonpositive in at least half of the original 64-row draws.
It motivated a valid repair test, but sampling noise alone does not explain or
repair the failed held policy. See [failure analysis](failure_analysis.md).

## Evidence and Next Experiment

Fresh_run: diagnosis, 36 new fits, 144 fixed readouts and full numerical replay.
Cached_verified: source data, upstream models, original/stochastic policies and
separate receipt arithmetic. Not_run: a new forecaster, independent evaluation
and deployment. Fit has 897,060 row occurrences over 74 source-qualified
recordings; held has 419,766 over 31. Twelve sources and repeated contexts/seeds
are not independent sample counts. The previously exposed inner-held data remain
exploratory; no independent selection, calibration or confirmation role opened.

The next useful test is recording-disjoint utility/constraint learning inside
original TRAIN, with the current shared-record optimizer as a control. First
reconcile the existing cross-fitting assets, then freeze the new comparison.
The question is whether reducing same-record constraint overfitting preserves
useful intervention on other recordings. It is not a new threshold search or a
reason to open independent test data. This follow-up is **not_run**.

100 targeted tests pass; 144 readouts and both frozen controls replay exactly.
The separate reporter checks 5,022 policy summaries and 288 frozen controls.
These verify execution, not independent scientific replication.

obs8/pred12 rawstride12, pixel/local detector-silver. No metric, seconds,
human-gold, physical-safety, true3D or foundation claim. Stage5C and SMC stay off.
