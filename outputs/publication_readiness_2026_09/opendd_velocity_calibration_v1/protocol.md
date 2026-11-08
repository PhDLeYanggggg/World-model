# Nested Calibration of the New OpenDD Forecast

Use the same exposed OpenDD development cohort, seven sites, 10,475
targets, 8/12 samples and CV fallback. Freeze the 21 existing outer
self-only forecaster checkpoints (seven held sites and three seeds).
Train 63 new inner checkpoints on the five sites remaining after each
of 21 unordered exclusion pairs, three seeds. Reuse each pair fit in
both directions. Model size, data draws and 600 updates are unchanged.
The held outer site does not enter fit, easy-cut estimation or alpha
selection. No DroneCrowd access, official rule revision, deployment,
Stage5C or SMC.

Calibrate two policies separately: (1) a uniform CV-to-network blend and
(2) a blend whose multiplier is reduced by `min(1, (0.05 + D(CV, OLS4)) /
D(CV, network))`, where D is maximum path disagreement and the 0.05 floor
uses publisher coordinates. Both keep only rows with a nonidentity new
forecast and at least one legal option in the existing motion menu, so
paired controls can edit exactly the same recipients. Positive alpha
grid: .025, .05, .075, .10, .15, .25. Select by mean full-grid conservative
gain if strictly positive on at least four of six inner sites, with all
six sites having defined complete-label easy reference and **positive**
easy harm at most 2% of that site's complete easy reference mass. Ties
choose the smaller alpha. If none qualifies, use exact CV everywhere;
fallback is not a policy success or a valid empty matched utility result.
This stricter site-level calibration filter is a development objective,
not a substitute for the original per-recording safety test.

On held sites compare each frozen policy to CV, fixed 25% on the common
roster, and old-bank hindcast and minimum-change policies at exactly the
same edited recipients and blend multipliers. These are coefficient-matched,
not necessarily displacement-matched. Debits for missing steps are computed
between the two actual compared forecasts. Report conservative gain, observed
ADE/FDE, signed and positive easy harm, coverage, and 3,000 paired site
bootstrap resamples after averaging seeds. Complete easy diagnostic cuts are
the 25th percentile of positive CV ADE among complete training rows. Also
replay the original recording risk algebra by flight using the training
cut and all available labels (including partials). Report all flights,
undefined selected-easy reference and defined violations separately. Any
flight lacking full easy reference cannot pass that algebra under any
policy; do not certify zero harm for identity output or change the rule.
