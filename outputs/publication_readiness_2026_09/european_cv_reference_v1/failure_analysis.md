# Reference Repair Failure Analysis

This analysis follows the completed, frozen source-development experiment.
It is not a new fit, an independent risk-calibration result or a threshold search.

## Confirmed and Unresolved Causes

| Cause | Evidence | Conclusion |
|---|---|---|
| Wrong safety reference | Old fitting-selected fallback itself degrades easy by 15.4759% versus CV | Confirmed contract mismatch; changing only how often the old fallback is used cannot remove that baseline error |
| Unconditional objective versus conditional risk | New cost labels are CV-relative, but the budget remains predicted positive harm over all candidates | Reference repaired; easy-event denominator and worst-locality constraints remain unenforced |
| Relevant-event support | All four zero-CV rows are in one excluded source fold whose fit has zero such rows | No fitting evidence for this event in that fold; an estimated zero probability would not establish safety |
| Partial future labels | Four zero-CV rows have 2/12 labels and no final endpoint | Masked-prefix correctness is not full-horizon correctness; do not remove the rows or infer missing futures |
| Strong simple control | Neural pointwise gain versus damping 0.97 is 0.09--0.36%, all intervals crossing zero | No established strong-control superiority |
| Site heterogeneity | Worst easy-locality degradation remains 8.71--12.54% despite mean 1.90--2.38% | Equal-locality mean does not imply per-locality protection |
| Joint contribution | Same-count controls show tiny/undefined gains, despite verified equal counts | Nonadditive predictive value not established; stronger graph complexity not justified yet |
| Teacher-size shift | Nested cost targets use four-site producers; final held forecasts use eight-site producers | Disclosed fitting/readout shift; contribution not isolated in this version |
| Runtime/convergence | All budgets finish with finite losses; some last minibatch losses exceed first losses | No runtime blocker; finite/noisy losses do not establish optimization convergence |

## Why the Budget Is Not the Safety Criterion

Let `L_CV` and `L_N` be supported masked ADE for CV and the frozen neural
forecast, `a` the binary intervention, `H=max(L_N-L_CV,0)` and
`G=max(L_CV-L_N,0)`. For the new policy the realized excess error is exactly
`a*(H-G)`. For the old policy an additional fallback-to-CV error term remained.
That additional term is removed by this reference repair.

For locality `j`, define the positive-easy event `E` using the unchanged
training-derived cutoff and `L_CV>0`. Its signed percentage degradation is

```text
R_j = 100 * E[a*(H-G)*1(E) | j] / E[L_CV*1(E) | j].
```

This is an algebraic decomposition, not a generalization theorem. The report
averages locality ratios and separately inspects the worst one. Constraining
predicted all-candidate positive harm to 2% of a fitting-wide cost scale does
not constrain this event-specific ratio or its worst locality. A small easy
denominator and concentrated harm can violate it even when overall ADE improves.
Prediction errors in the harm head add another gap between estimated and actual
risk. Neither fixed thresholds nor a conformal name would remove those gaps.

At the zero-reference event `Z=(L_CV=0)`, `G=0`, so the criterion requires
nonnegative added harm `a*H` to remain zero. Finite absence of observed harms is
not a population guarantee. The joint pilot observes no `Z` examples at all;
its zero count is an absence of support, not validated protection. The excluded
fold containing all four `Z` cases has no fitting examples of `Z` either.

## What This Version Does Not Establish

- The repaired head is not proven more accurate than the fixed strong damping
  control; the small point estimates are not a deployable margin.
- Neither the raw forecaster nor the CV-reference policy is safe on every source
  locality. No new winner is selected after reading these results.
- Fitting and held-source producer sizes differ. This version does not isolate
  whether matching producer sizes would improve risk prediction.
- A last-velocity-zero inference rule was not tested or selected. The observed
  four cases cannot become a test-dependent exception. Future availability and
  zero-ADE labels must never become inference features.
- Released detector tracks and coordinate timestamps do not by themselves prove
  online as-of annotation causality, common seconds, metric scale or safety.

## Next Repair, Before Any Reserved Readout

1. Version a CV-relative, event-conditional risk objective with training-only
   easy/zero-event support summaries. Retain unknown-label targets and original
   risk tolerances. Compare against the present head on exactly matched sources
   and forecasts before any capacity increase.
2. Test a past-only support-aware abstention rule. If event support is absent,
   do not equate an estimated zero risk with evidence of safety. Any new source
   sampling must be preregistered, restricted to already admitted training
   recordings and chosen without future availability or evaluation errors.
3. Only after a credible source candidate, freeze one model/policy for the
   separately reserved selection and calibration stages. Retain confirmation
   data closed. Quantify calibration support before making a risk claim.

This is the next experiment rationale, not evidence that these repairs already
work. No new independent readout, threshold change, deployment, Stage5C or SMC
is authorized by a positive average source score.
