# Supported-Event Pairing and the Limits of Realized-Ratio Ranking

## Method

The previous risk-ranking auxiliary discarded a pair whenever either realized
event cost had zero total mass. This provided sparse easy-event supervision.
We instead remove only these undefined rows before forming cyclic pairs within
each original minibatch/locality. Their original moment and occurrence/severity
losses remain. The model, initialization, sampler, ranking formula and weight,
optimizer, update budget, forecasts and risk rule are unchanged.

We fit 36 small Torch heads across three source-role rotations, three seeds,
two candidate predictors and two event targets. Every six-view full/common and
matched-count comparison is retained, for 216 views. Forecast-producer exclusion
covers the complete fitted chain. All twelve localities are nevertheless opened
development, not independent final testing. Confidence intervals use 3,000 paired
locality resamples, conditional on fitted models and unadjusted for dependent
comparisons.

## Findings

Neural/easy pair exposure increases from 167,291 to 541,866, and fixed fitting
total loss decreases in all nine heads. However, same-count all-ADE ordering
contrasts remain mixed: one positive and one negative interval at old counts,
two positive and one negative at new counts. Neural/all behavior is exactly
unchanged, consistent with its already-supported labels. Damping/easy ordering
worsens, with six negative and no positive intervals at either count anchor.

All 18 neural all-ADE point contrasts remain below equally protected damping;
17 conditional intervals are negative. The hard subset has no positive interval.
Neural positive-easy degradation stays within 2%, but 11 views still harm cases
with an exact constant-velocity baseline. More pair support is not sufficient to
establish safe neural forecasting superiority.

## Target-Definition Diagnostic

A constructed two-state example shows that a margin-weighted ordering objective
on realized H/(B+H) can prefer the reverse of E[H|x]/E[B|x]. The actual auxiliary
loss is lower when one unsafe state's harm is underpredicted. Original moment
losses still oppose that error; the example does not characterize the optimum of
the full combined objective or attribute real-data failures causally.

For independent conditional draws, E[H_i B_j-H_j B_i] has the sign of the desired
conditional risk difference when denominators are positive. This elementary sign
identity motivates a future controlled target comparison, not a new calibration
theorem. Dependence, normalized stochastic weights and variance remain unresolved.
The [executable counterexample](estimand_counterexample.md) keeps analytic and
real-data evidence separate. No new model uses that proposed target here.

## Limits and Reproducibility

All checkpoint replays, 216 metric views, 36 old controls and separate arithmetic
checks pass; 247 scoped tests pass. This is reproducible computation, not
independent confirmation. Limited source diversity, zero-reference support and
the absence of stable neural advantage prevent deployment or submission-ready
claims. Detector-track image pixels, obs8/pred12 rawstride12; not seconds, metric,
human gold, physical safety, true 3D or foundation evidence. Reserved roles remain
closed; Stage5C and SMC remain disabled.
