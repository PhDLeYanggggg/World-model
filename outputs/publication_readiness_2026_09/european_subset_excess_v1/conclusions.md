# Conclusions: Subset Supervision Did Not Repair Safe Selection

## What Was Run

fresh_run:108paired groups,216new25,028-parameter heads,432,000updates.
All initialization, query/row draws and optimizer budgets match both new arms
and the frozen pointwise control. Individual supervision contributes half of
both objectives. Only auxiliary pointwise versus aggregated subset supervision
differs. Registration f39d5aa4 preceded fitting; training freeze36923606 preceded
decisions; action freezeb959c128 preceded outcomes. No held threshold search.

## Predeclared Comparisons

- Matched-count aggregate versus pointwise subset ranking ADE gain:
  +0.000678%,95%development CI[-0.008701%,+0.010486%];5/12localities positive.
- Its all-reference-denominator harm reduction is negative:
  -0.002279percentage points,CI[-0.004717,-0.000135]. At fixed counts it
  increases this harm diagnostic rather than reducing it.
- Aggregate versus pointwise subset joint ADE gain:
  +0.003497%,CI[-0.011473%,+0.017681%];6/12localities positive. This contrast
  is not rate matched. Joint harm reduction is -0.010914pp,
  CI[-0.020107,-0.003440]. Intervention falls by0.135569pp.
- Aggregate rank versus cached pointwise rank has a tiny secondary gain:
  +0.004553%,CI[+0.000081%,+0.009690%];8/12localities positive. It changes
  subset weighting as well as aggregation, does not establish an aggregation
  effect, and does not improve the harm screen. Intervals are nominal across
  the predeclared contrasts, not multiplicity-adjusted confirmatory tests.
- Aggregate joint versus cached pointwise joint is -0.012015%,
  CI[-0.027374%,+0.003311%];2/12localities positive.

Aggregate joint gains0.507865%ADE and0.557981%hard over its protected floor;
pointwise subset joint gains0.504890%ADE and0.538433%hard. These averages do not
make either policy safe. Aggregate joint has84/216observed-risk violations and
16undefined selected-risk ratios, versus82and13for pointwise subset joint.
Worst aggregate-joint easy gain over CV is+0.176646%; zero-CV harm is0.

## Decision

The registered exploratory screen fails. No deployment change, independent
confirmation, formal calibration certificate, Stage5C or SMC. The old selected
risk primary remains structurally incomplete; zero-action views are not filled
with zero risk. The new full-reference diagnostic is not a replacement2%risk
certificate. Source roles remain4producer/4controller/2head-fit/2held across
twelve already-opened development localities.

This rejects the tested fixed-subset aggregation repair, not every possible
selection-aware training method. It does not establish that more training,
more modalities or a larger model would solve the remaining problem.
Technical verification status is recorded separately in verification.json.
The full M3W research/submission goal remains incomplete.

Image-local detector silver, obs8/pred12 raw-frame stride12. Not metric, seconds,
human gold, physical safety, true3D or foundation evidence. Historical exposed
Stage37 t+50 figures are not directly comparable or recertified by this study.
