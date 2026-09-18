# Source Trajectory Cost Alignment: Fixed Experiment

## Question and Scope

The preceding source-site classifier fits training labels better with RGB but
performs worse on every held physical site's mean Brier score. Its binary target
counts very small annotation changes equally with larger movements. A small set
of larger movements contributes disproportionate trajectory error. This study
therefore tests actual twelve-step candidate trajectories, not another start
classifier or an additional routing threshold grid.

Hypothesis: training directly for mean trajectory error yields more useful
bounded stationary-history forecasts than compressing individual errors with
log1p. A matched RGB/mask contrast tests the incremental role of pixel content.
No new method or useful forecasting result is assumed in advance.

Only approved original SDD train40 is used. Keep the exact preceding stationary
complete-label cohort and all five physical source-site folds. Three seeds and
two input arms by two objectives give60fresh fits,2,000updates each. Incomplete
future labels remain unscored,notnegative. Main11966/native8-to12/equal-site
past-normalized ADE, sealed roles and deployment rules are unchanged. This is
internal source-fit diagnosis, not independent confirmation or new formal split.

## Predictor and Objectives

Reuse the existing CNN/MLP input architecture and past image cache. Replace the
binary output with24 coordinates for12future relative positions. Initialize the
output layer exactly tozero, matching stationary CV. Map each local offset z to
z/(1+norm(z)), then restore with the existing past-only context radius/rotation.
No spatial support means exact baseline fallback; those rows are retained.

The same four observed rotation entries are appended to both arms so that the
unrotated image and canonical vector frame are identifiable to the predictor.
There is no new image-resolution choice, scene ID, absolute frame ID or future
input. This new trajectory task is not a single-factor causal comparison against
the older binary classifier; the new two-by-two comparisons are matched.

For parent-normalized per-window ADE e and c=mean training-complement CV ADE:

- ADE objective: mean(e/c).
- Log-ADE objective: mean(log1p(e/c)).

c is fitted only from each training complement and is a global loss scalar,
never an inference feature. Both objectives restore predictions to the original
coordinate/scale before scoring. Uniform training-window sampling, same seeds
and row draws;AdamWlr.0003/wd.0001,batch64,gradientclip5,checkpoint200. No early
stopping, best-checkpoint search or held-site threshold selection.

## Fixed Evaluation

Report uncontrolled predictions and a fixed0.9 gate using the already frozen,
same-arm/seed/held-site classifier. Its producer excludes the whole held site;
verify producer hashes and row joins. This gate is an uncalibrated diagnostic,
not a safety guarantee or deployment decision. Also report future-informed
binary CV/neural oracle as a diagnostic only. No oracle values enter inference.

Primary source diagnostic: equal physical-site mean of parent-normalized ADE;
report reductions versus CV and matched objective/input controls. Retain all
sites/seeds. Report native pixel ADE/FDE, per-agent/video sensitivity, train-only
90th-percentile-CV hard slice, zero-CV-error easy absolute harm, intervention,
oracle headroom and context-support ceiling. Percentage easy degradation is
undefined for a zero baseline denominator; do not manufacture a passing value.
This does not replace the existing main forecasting primary endpoint.

Use2,000conditional physical-site block resamples and within-site video blocks.
Average seed losses,not ensemble trajectories. Five exposed sites, overlapping
training folds and overlapping windows limit independence. No formal risk or
population claim. Whether any fit improves is reported,not used to select an
unregistered replacement or open the main sealed roles.

## Execution and Limits

Inputs cached_verified;new fits/evaluation fresh_run. Nativearm64CPU4,oneinter-op,
workers0. A100-update training-only pilot estimates cost and is resumed inside
the fixed120k budget. Local expected cost is under the user's12-hour allowance;
CREATE M3W contents remain unknown because the last verified access attempt
ended at public-key/MFA authentication. No remote job is inferred absent or
resubmitted. Exact prediction replay,matched draws,finite training and immutable
completed-resume checks required before completion claims.

This remains offline annotated prediction, not strict sensor-as-of. Labels are
silver; source horizon is+144rawframes, not physical-time-equated to main. No
metric,seconds,true3D,foundation,Stage5C orSMC claim. No model is deployed by this
diagnostic alone. No raw data, images, per-row predictions or weights go to Git.
