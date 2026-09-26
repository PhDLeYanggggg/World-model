# A Large Empirical Floor Is Not A Deployment Repair

## Material Passport

Completed fresh fitting-only accounting on 144 views, four frozen prediction
banks, two weightings and two caps. All 2,304 error identities pass, maximum
observed error 1.1369e-13. Inputs and neural scores are cached_verified against
row, target, source and prediction hashes. No new model was fitted and no
outer held labels were evaluated. No deployment or independent-role change.

## Result

The fixed predicted H cap limits realized-label correction substantially.
The table reports medians across 72 dependent fitting views per entry, not
pooled errors, independent replications, confidence intervals or model gains.

| Inputs / predictor | Uniform floor share (%) | Existing risk-weighted floor share (%) |
|---|---:|---:|
| Full / original in-sample | 63.7546 | 60.9854 |
| Full / inner OOF | 68.2613 | 67.2560 |
| Motion-only / original in-sample | 66.6523 | 62.7160 |
| Motion-only / inner OOF | 79.4729 | 79.4934 |

Full original uniform shares range 37.3452-83.5151%; OOF shares range
15.3145-93.8420%. The separate median above-cap row fractions are 4.33697%
and 4.41632%. Motion-only originals and OOF have median fractions 0.60134%
and 0.68429%, respectively. These are separately summarized view statistics,
not a pooled claim about a single population. All cyclic controls, weighted
summaries and per-view values remain in [results](results.md), the CSV files
and [figure](fixed_cap_diagnostic.svg).

Using the causal disagreement envelope as the diagnostic cap yields zero
projection floor for every bank/view/weighting. That follows the verified
label support bound. It does not mean an envelope-bounded model can perfectly
predict future harm: the offline minimizer is allowed to see each label.

## Why Not Raise The Cap Globally?

Across all 1,152 bank/view/weighting records, none has positive mean(y-H).
The cap can be too small for rare individual outcomes while its average
already exceeds mean easy harm. Full original risk-score bins have positive
mean(y-H) in 0/216 uniform cells and 1/216 weighted cells. Full OOF has 11/216
and 5/216. Motion-only original has 2/216 under each weighting; OOF has 23/216
and 14/216. Cells are dependent fitting summaries, not calibration tests.

Thus the result does not support a blanket upward shift. Nor does it establish
that a learned conditional cap repair is impossible. An exact conditional-mean
predictor can have a large realized-label projection floor; the test suite
contains a concrete 50% counterexample. Keep E[H_E|x] <= E[H|x] intact.

The remaining error includes distance and cross terms, not only examples
whose labels are below the cap. Both rare-tail prediction and transferable
conditional information remain unresolved. The previously failed source-held
MSE and tail/coverage gates are unchanged by this diagnostic.

## Consequence For The Next Experiment

Stop treating more H_E-only residual flexibility or global cap relaxation as
the default repair. First test whether the producer-relative cap-exceedance
event has learnable causal signal and recording-level support under strictly
nested fitting-locality exclusion. This differs from generic any-harm ranking,
which was already weak within positive disagreement in the earlier
[tail-crossfit experiment](../european_harm_tail_crossfit_v1/conclusions.md).

Only supported out-of-fold evidence should trigger a matched joint H/H_E
repair against the original and frozen-H control. Preserve all source roles,
capacity/budget controls and existing primary metrics; do not select a cap,
threshold or model using an outer result. This is a next hypothesis, not a
newly demonstrated improvement or authorization to open reserved data.

Registration 06511693 preceded pilot and full computation. Pilot computation
including source loading was 14.846347 seconds for one view; the full run
was 201.085832 seconds, excluding earlier ancestry checks, plotting and replay.
Native arm64 CPU execution, no new neural or trajectory updates.

Status: diagnostic completed, no model promotion, research goal still active.
Obs8/pred12 native annotation steps and detector pixels only. No metric,
seconds, true-3D, foundation, human-gold, physical-safety or submission-ready
claim. Stage5C and SMC remain off. Exact replay status is in verification.json;
passing implementation checks does not create a scientific improvement gate.
