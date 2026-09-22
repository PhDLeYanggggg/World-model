# Frozen Cost-Head Transfer from Transformer to EqMotion

Registered 2026-09-22 before computing the transferred scores or policy outcomes.
The previous predictor readout is known: EqMotion improves average source ADE
more than Transformer but harms easy/exact-CV paths. The previous cost study is
also known: the fraction-loss secondary arm had a favorable, unconfirmed tradeoff.
This new experiment is therefore exploratory method development, not independent
replication, and does not relabel that previous secondary arm as its primary.

## Question and Scope

Does the already learned baseline-relative benefit/harm representation transfer
unchanged to a stronger prediction family? Test this before spending on a new
EqMotion-specific nested producer and cost-head training matrix. A failed frozen
transfer is not proof that a correctly retrained EqMotion head cannot work.
No new fitting, normalization, calibration, thresholds, model selection or data
roles. The original closed val/test/main/external/bookstore roles remain closed.

Reuse all twelve verified native EqMotion K=1 predictions and all36 frozen Torch
cost heads (direct-native, bounded-native, bounded-fraction), one of each arm per
site/seed. Retain the four design-exposed SDD sites and seeds17/29/43. Source cost
heads retain their original pair-excluded Transformer training lineage; their
own outer source site is excluded from fitting and preprocessing. No EqMotion
evaluation costs become training targets. Fit-only standardization remains fixed.

The inputs are the same356past/candidate features, including full requested-grid
forecast disagreement. Only the candidate prediction changes. Future positions,
future-validity masks and outcome-defined easy/hard labels are not decision inputs.
All transferred scores/choices must be frozen before reading target arrays.

## Fixed Controls and Contrast

Controls: CV floor, uncontrolled EqMotion, and EqMotion with a last-observed-step
stop veto. Each of the three frozen heads gets the old positive-net-gain rule and
the old strict rule (predicted harm <=0.1*predicted benefit); stopped histories or
identical candidate/CV forecasts cannot switch. No search over those constants.

Primary new contrast: bounded-fraction strict minus direct-native strict in
equal-site ADE gain. The joint empirical criterion requires its conditional95%
interval lower bound >0, positive ADE gain versus CV in every seed, zero observed
complete exact-CV harms in every seed and <=2% positive-easy degradation in every
seed. An all-reject policy cannot pass just by beating a harmful comparator.
Also report each site/seed, hard gain,
tails and missing outcomes; this empirical criterion is not a safety guarantee.

Matched-count diagnostic: within each site/seed, take the intervention count of
the fraction strict rule, then rank eligible samples by each arm's predicted net
gain at that exact count. Break ties by stable queryID. This is an offline batch
mechanism comparison, not a prospective fixed-threshold deployment policy.
Report fraction-minus-direct and fraction-minus-native contrasts for both strict
and matched-count policies. Keep all controls and all seeds, regardless of score.

## Readout and Interpretation

Eight observed/twelve requested annotation steps, SDDstride12, annotation pixels.
Do not mix with historical raw-frame t50 scores. Equal-site native ADE gains are
primary, FDE and training-CVq75 hard/training-positive-CVq25 easy are diagnostic.
Keep the existing cost-training subset thresholds, not new evaluation quantiles.
Report effective switches, complete zero-CV absolute harms, unknown/incomplete
selected outcomes, conditional predicted versus realized harm and feature-shift
diagnostics. Include full-grid absolute-gain partial-identification bounds for
missing outcomes; these are not confidence intervals or percentages.

Use3,000paired physical-site bootstrap draws after seed-error averaging, not
overlapping-window bootstrap. Four explored sites are few and not independent
confirmation. Full score/decision replay and a separate arithmetic verifier are
required. Keep a PID heartbeat and immutable identities/artifacts. Cached neural
forecast inference need not be rerun after its already verified full replay.

Failure should identify predictor-feature shift, conditional risk underestimation,
ranking versus intervention-count effects, or missing outcome support. Success
would support a frozen method-transfer hypothesis, not establish a novel world
model, a calibrated policy, formal independence or publication readiness.

No new deployment, risk tolerance, metric/seconds claim, true3D/foundation claim,
Stage5C execution or SMC. Raw data, caches and checkpoints remain outside Git.
