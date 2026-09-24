# European Conditional Event-Risk V1

## Material Passport

Source-only experiment, registered before its fits and comparative readout.
Routine research execution is delegated. Previous source results inform the
hypothesis; these sources are therefore development, not independent test data.
The prior completed CV-reference version and all producer identities stay frozen.

## Hypothesis and Factorial Test

CV-reference repair reduced easy harm but did not constrain easy-event risk.
Test whether predicting event-specific positive harm and baseline-error mass
helps, separately from refusing intervention when fitting zero-event support
is absent. Do not increase predictor capacity or open new data roles.

For each nested fitting row let R be CV masked ADE, H=max(neural ADE-R,0),
and E=1(0<R<=the original fitting-only easy cutoff). Train matched two-channel
regressions for (R,H) and (R*E,H*E). Unknown future labels produce unknown
targets in both channels, never zero loss labels. Both arms use exactly the
same causal features, cost scale, source-balanced sampled draws, seeds,
architecture and training budget. Only the event mask changes these targets.
The first channel means baseline-error mass, not benefit or a probability.
The reused asymmetric loss weights harm underestimation 4x; denominator error
is symmetric. Neither channel is a calibrated upper/lower bound.

Utility is always the frozen CV-reference neural cost head's predicted G-H,
normalized by its fitting-only CV cost scale. It is not re-estimated or selected.
Train ridge(alpha0.1) and 64-wide neural regressors for both event choices,
three seeds17/29/43 and three outer folds:18ridge fits and18neural fits,
2,000updates each neural fit. The optional100update pilot resumes within the
36,000update budget. All36heads finish before the first comparative readout.

Each fitted head gets two fixed deployment *diagnostics*: no guard, and
source-zero guard. The latter abstains throughout an outer fold if its fitting
portion has no supported R=0 rows. This is a training-support ablation, not
a new guarantee when a few events exist. It is not a detector fitted on held
zeros and does not inspect held labels, site error, future availability or
last-velocity-zero outcomes at inference. Zero-event predictions are not used
to infer future labels. It may sacrifice useful coverage or collapse to CV.

## Decisions and Matched Controls

Pointwise candidates require positive frozen utility, observed historical motion,
positive predicted denominator and predicted H_event <=0.02*D_event. Joint
query budget is sum(a_i*H_event_i)<=0.02*sum(D_event_i). All query candidates,
including unknown-future candidates, contribute their *predicted* denominator.
This is not a calibrated guarantee of conditional realized easy risk. It may
still borrow risk mass across agents or locations and may fail in new domains.

Use the previous fixed1,152queries/6,116targets for joint controls; full-pointwise
readout retains all318,969targets. Compare no intervention/CV, training-selected
baseline, fixed damping0.97, frozen neural output, prior CV-reference policy,
independent risk decisions, whole-query uniform, joint, exact-count unary and
exact-count joint. Match actual counts against independent decisions before
looking at labels. Pair proxy0.1, past-edge radius3current median box widths,
forecast proximity0.5widths, solver2seconds/query stay fixed. These proxies do
not constitute metric geometry, physical collisions or interaction ground truth.

## Producer and Data Boundaries

All18source forecasts are reused, with their exact nested producer assignments.
The outer excluded source fold is absent from features' learned preprocessing,
cost targets' teacher training, prototypes and all risk-head fitting. Original
easy/hard cutoffs and source query/row eligibility remain unchanged. The final
8-site producer versus4-site cost-target producer shift remains disclosed,
not silently repaired in this experiment.

Only the12previously opened training localities are used. No reserved selection,
calibration or confirmation data, no DroneCrowd confirmation or new external
source is opened. No new split, horizon, unit, risk tolerance or eligibility
decision. Source-level zero-event support is calculated from fitting rows only.
Support absence is not a scientific success even if resulting fallback is safe.

## Readout and Stop Rules

Report every seed, event, head and support control. Primary comparisons are
against the strong fixed controls and preceding CV-reference policy. Report
ADE/FDE, all/hard/easy, worst easy locality, zero-CV harm, missing label counts,
complete-future sensitivity, tails, switch rate, harm over CV, conditional-moment
prediction errors and same-count joint contrasts. No post-readout winner or
threshold selection. Bootstrap3,000locality resamples, seed39271, conditional
on source data and fitted models; do not treat overlapping windows as iid.

Observed positive-easy degradation<=2% and zero-CV added harm0 remain unchanged;
inspect worst locality as well as the mean. Empty zero-event or contrast support
is unavailable, never a safety certificate or a fabricated zero contrast.
Prediction improvement with failed safety does not permit deployment.

Local native arm64 Torch, CPU4/inter-op1/workers0; atomic checkpoints with
optimizer/RNG/draws and heartbeat every200updates. Prior actual local fit costs
support local placement. No new HPC job, remote scan, authentication change or
action on protected simulation jobs. Long healthy work is not downgraded.

Fresh fits/first readout are fresh_run; frozen source forecasts and complete
cached metric replay are cached_verified; new checkpoint inference is separately
fresh_run. No new dynamics-training claim. No seconds/meters, verified online
sensor labels, human gold, true3D, foundation, physical safety or submission-ready
claim. Stage5C and SMC remain disabled; the overall research goal stays active.
