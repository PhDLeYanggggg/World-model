# Cutoff-Relative Source Risk Features

## Question and Role

The preceding registered dimensionless experiment improved average intervention
utility but worsened easy-risk estimation and neural population easy degradation.
Its first354 normalized features plus D/past_scale lose absolute motion amplitude,
whereas the supervised easy event still uses a fixed native source error cutoff.
A synthetic scale witness can establish information loss, not that it is the sole
cause of the empirical failure. The new hypothesis is that cutoff-relative scale
context restores useful risk information without hard-coding the coordinate unit.

Only development-exposed SDD source data are used. The main protocol remains
obs8/pred12 native annotation steps, stride12, annotation pixels. No new data-role,
easy definition, cutoff, risk tolerance, predictor, forecast or target is chosen.
No independent confirmation, external prediction error or deployment readout is
authorized. DroneCrowd remains closed and IMPTC remains quarantined.

## Fixed Training

Replace the last two native head features with log(past_scale/cutoff) and
log1p(D/cutoff); keep the first354 features. The training-only cutoff is the same
one already used by each source fold's six target moments and predicted budget.
It never comes from the held site's future errors. Under a coordinate-unit
relabel, scale, D and the cutoff must all change together. Using an SDD pixel
cutoff on another dataset does not magically become valid external calibration.

Fit36 new six-output forests: four excluded source sites, three seeds and three
actions (damping .05, Transformer, EqMotion). Each uses128 trees, maxdepth16,
minleaf64, feature fraction1/3 (118 columns per node), source-only preprocessing
and exactly the frozen768000 draws. Supervision/known masks/draws must match the
previous native and dimensionless heads. CPU4, workers0, save every16 trees.
First run16 trees for coupa/seed17/damping, then exact resume to the entire matrix.
The previous72 fits are cached_verified controls, not fresh training this round.

## Fixed Decisions and Evaluation

Apply the unchanged pointwise, population and selected-denominator rules at
rho=.02. No threshold/model sweep, output-based policy selection or new interaction
term. Freeze all new decisions before the new aggregate readout. Register and
retain contrasts against the matched native arm, preceding dimensionless arm and
old strict for each rule/action. Recompute all39 old/new summary rows and verify
that the30 old reductions reproduce. Source load may materialize previously
exposed label arrays, but they cannot affect fits, preprocessing or decisions.

Report observed-grid ADE/FDE, complete/hard/positive-easy/zero-CV, selection rates,
unknown/incomplete selections, partial-future bounds, six-target training and
held-source MSE, per-site/per-seed effects and3000-resample physical-site bootstrap.
Intervals are nominal four-site development intervals, not independent confirmation.
No overlapping-window independence or multiple-comparison claim.

Utility alone is not success: show easy tradeoffs and zero-CV harms. A favorable
mean with an interval crossing zero is uncertain. Matching the original budget
does not match intervention rate; no new equal-coverage ranking claim is allowed.
The previous12569 failed exact-count controls remain failures, not repaired here.

Metadata unit relabel probes use factors .01 and100 with normalized forecasts
fixed. Report exact feature/prediction agreement and any deviations; a numerical
mismatch invalidates exact unit-invariance, not the otherwise valid source readout.
No prefix-adapter precision repair is mixed into these frozen forecasts. A source
feature-unit check is not end-to-end cross-dataset invariance or calibrated safety.

## Provenance and Recovery

Hash-bind source/parent artifacts, config, code, tests and this registration before
fitting. Use one writer, PID/heartbeats, resumable checkpoints and immutable
decisions/reports. Run decision and aggregate replay, separate arithmetic and
scoped regression tests. No raw data, features or weights enter Git. All outcomes,
including failure, remain public aggregate evidence. Stage5C and SMC stay off.
