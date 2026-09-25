# Nested Source Calibration With Producer Exclusion

## Material Passport

Mode: registered code experiment, source-development only. Status: registered
before new fitting. Population: the same 12 opened European Squares localities.
Primary source: hash-bound detector-track assets and existing producer bank.
No independent selection/calibration/confirmation role is opened.

## Hypothesis and Necessary Repair

Symmetric risk regression increases neural utility but underestimates harm on
selected rows. A held-source calibration rule may improve protection. It must
be fitted on scenes excluded from the entire prediction and scoring chain.
Existing eight-locality cost/risk heads cannot simply be reused: their nested
forecast producers can include a proposed calibration locality, even when the
final score head has not fitted that row. That would not be independent calibration.

Keep the frozen three source folds, four localities each. For every fitting
fold, retain both assignments of the other two folds to calibration and outer
development readout. This is six ordered role assignments, not six independent
datasets. Pool by the two predefined cyclic orientations; retain all seeds and
both orientations. These roles remain subdivisions of development, never a
renaming of them as untouched test data.

## Producer Bank Phase

Split each fitting fold into two disjoint two-locality halves using past-eligible
row counts and the fixed salt, not future errors. Train a predictor on each half
and predict only the opposite half. Both halves exclude every calibration and
outer locality. These out-of-fold predictions supervise gain/harm/risk heads.
The final four-locality predictor already exists and is hash-verified; it
predicts both calibration and outer localities without refitting on either.

18 new inner Torch predictors (3 folds x2 halves x3 seeds), each4,000 updates:
72,000 updates, with the parent's architecture, optimizer, learning-rate schedule,
normalization, loss, batch64, CPU4 threads and workers0. A100-update runtime
pilot resumes inside the first budget. Existing source sampler semantics are
preserved: future-unknown draws have zero supervised loss and are counted
explicitly, not described as labeled examples. No early stopping or checkpoint
selection. This is not a claim that less training data improves the predictor.

Freeze and replay the new bank before the calibration readout. New predictions
are generated only for the opposite fitting half. Existing final four-locality
predictions cover calibration/readout. New bank need not predict reserved roles.

## Head and Calibration Phase

Fit54 MSE heads:utility, all-event risk and easy-event risk for each candidate,
fitting fold and seed. Two candidates are frozen neural forecasts and fixed
damping0.97. Heads see only four fitting localities, with neural labels produced
by the disjoint two-locality producers. Candidate comparisons match their source
exposure, causal355-feature family, seeds and fixed2,000-update budget. Unknown
labels are excluded from head sampling. Heads then freeze before calibration.

Use the same frozen head and final producer on calibration and readout rows.
Compare three declared pointwise rules, without choosing a winning outer result:

1. No calibration: the fixed2% predicted positive-event-harm rule.
2. Population rescaling: adjust predicted harm upward and denominator downward
   when calibration-locality moment averages show optimism. Never relax the
   original ratio restriction; zero predicted mass with positive realized harm
   yields abstention, not division by zero.
3. Selected-risk grid: evaluate the declared ratio cutoffs0..0.02 on calibration
   only. Require each calibration locality's positive all-event and easy-event
   harm totals to remain below2% of the corresponding full-locality CV mass;
   require no added error on its observed zero-CV cases. Choose highest
   equal-locality net benefit among feasible cutoffs, breaking ties toward fewer
   interventions then stricter cutoff. Always retain explicit all-CV fallback.

Calibration is empirical fitting, NOT a finite-sample conformal guarantee.
Only four calibration localities are available per assignment; overlapping rows
must not inflate independent sample size. Selected-risk thresholds are frozen
before outer readout. No outer outcomes choose settings or new thresholds.

The matched uncalibrated control under this reduced fitting roster is essential:
comparison to the previous eight-locality experiment alone would confound
calibration, fitting data and producer size. Report producer shift and candidate
quality separately. Neither original historical Stage37 nor its claims are
re-certified by these results.

## Readout, Decision and Subsequent Work

Report all declared pointwise views with ADE/FDE, hard/easy, worst-locality
easy, zero-CV, complete-future support, intervention rate, gain/harm and
3,000 locality-bootstrap resamples. Three seeds, both cyclic orientations.
Conditional development intervals are not independent confirmation or
multiple-comparison-adjusted discoveries. No new joint optimization in this
first nested-calibration study: calibrated pointwise risk is not a certificate
for joint decisions. A joint follow-up requires its own frozen calibration map.

Success requires neural gain over matched protected damping with preserved
worst-locality easy and zero-CV outcomes across the declared controls, not just
reduced risk-estimate distance or gains vs CV. All negative/undefined results
stay visible. If calibrated selection degenerates to CV, report lost utility.
Do not open independent data merely to rescue a source-development failure.

## Related-Method Boundary

[Learn then Test](https://arxiv.org/abs/2110.01052) frames calibrated risk control
through hypothesis testing. [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
controls expected monotone loss under its assumptions. This experiment does not
implement either finite-sample guarantee. Four dependent-in-protocol source
localities, unbounded native trajectory costs and empirical threshold fitting
do not automatically inherit those guarantees. Novelty is not established by
adding a calibration scalar or combining existing modules.

## Runtime and Claims

Local arm64 first; fresh pilot establishes fit cost.18 small predictors and54
small heads are expected to fit within local memory. Save atomic checkpoints,
PID/UTC heartbeats and sampling state; explicit resume only. Available disk at
registration is approximately27GiB; new cache budget is below3GiB. Do not
delete unrelated data. CREATE remains an option if resource limits emerge,
not a requirement to displace unrelated scheduled work.

Image pixels,8 observed/12 predicted, raw stride12; not historicalt50, seconds,
metric, human gold, physical safety, true3D or foundation evidence. Stage5C and
SMC remain disabled. No deployment or submission-ready claim.
