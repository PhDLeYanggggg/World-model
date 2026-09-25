# European Risk-Protected Motion Control V1

## Material Passport

Registered source-development comparison, before fitting new control heads or
reading their comparative outcomes. Prior source results motivated this control;
none of these localities becomes independent test data by renaming the task.
Routine implementation, testing and execution are delegated.

## Question and Single Scientific Factor

Does the frozen neural trajectory candidate provide predictive value beyond a
simple fixed damping-0.97 candidate when both use the same CV reference,
candidate-specific utility/risk learning, source support and intervention rules?

Change the candidate and its necessary rollout-derived features and supervised
cost targets, not the training population, feature schema, budget or risk rule.
Fixed damping 0.97 is an already disclosed strong source accuracy control, not
a coefficient selected on new held outcomes and not itself a certified floor.
CV remains the decision reference for both candidates.

Reuse the verified neural utility and all/easy moment heads from the completed
CV-reference/event studies: nine neural utility heads and 36 moment heads.
Train the matching damping heads: nine neural utility, 18 ridge moment and
18 neural moment heads, 54,000 neural updates total. Three seeds 17/29/43,
three outer source folds, 64-wide heads, 2,000 updates, batch 256, learning rate
0.0003, harm-underestimation weight 4, ridge alpha 0.1. A real 100-update pilot
resumes inside this budget. Cached neural fits are not called fresh training.

Utility predicts positive gain/harm relative to CV; fixed utility is predicted
gain minus predicted harm divided by the same fitting-only CV error scale.
Risk targets are (CV error, positive added harm) or these moments multiplied by
the positive-easy event defined using the original fitting-only cutoff. Unknown
future labels remain unknown in both channels, not zero. For moment fitting,
equal candidate/reference rollouts imply zero harm, not zero baseline-error mass.

All compared heads share training row ids, known-label support, source-balanced
draws, architecture and update budget. Feature means/std and target initializers
are fitted separately because the candidate-derived features/targets differ.
The CV scale and draw sequence must match exactly. No candidate-specific tuning.
Deterministic damping needs no teacher training; the neural four-source-site
cost-target versus eight-source-site final producer shift remains disclosed.

## Decisions and Population

Evaluate all 48 fixed candidate/event/head/support views. Pointwise decisions
cover all 318,969 source targets, including those without observed futures.
Joint decisions use the same fixed 1,152 queries / 6,116 targets. Neither missing
labels nor hindsight difficulty chooses inference eligibility.

Pointwise positive utility and predicted harm <=0.02*predicted event error mass.
Joint sum(selected predicted harm)<=0.02*sum(predicted event error mass) over
all targets in a query. Keep no guard and fitting-zero-support guard: the latter
abstains throughout folds with no fitting zero-CV events. It does not identify
held zeros or certify safety when a few training events exist.

Compare CV/no intervention, raw candidates, training-selected baseline,
pointwise, independent query-budget, whole-query uniform, joint, exact-count
unary and exact-count joint. Actual counts must match within each candidate's
independent/unary/joint comparison. Between candidates the risk budget is the
same but realized intervention counts can differ; do not claim otherwise.
Pair weight 0.1, edge radius three current median box widths, proximity threshold
0.5 widths, two-second solver limit remain unchanged.

Use the separately tested exact-infeasibility pruning for both candidates before
risk division: an individual nonnegative cost exceeding the whole query budget
cannot belong to any feasible subset. Re-execute neural decisions as well as
damping decisions; do not silently substitute old numerically failed calls.
Pruning changes no mathematical feasible set and relaxes no safety tolerance.
Report any changed decisions versus old V1, including solver/tie effects.

## Predeclared Readout

All 45 new heads complete before comparative readout. Report every seed and
control; do not choose a winner or tune a threshold after reading outcomes.
Primary contrast: neural versus damping at the same event/head/support and
decision rule, on the same sample population. Also retain raw and CV-relative
ADE/FDE, hard, positive-easy mean/worst, zero-CV harm, tails, complete-future
sensitivity, intervention rates and within-candidate matched-count contrasts.

Bootstrap 3,000 locality resamples, seed 39271; equal-locality percentage changes,
not iid overlapping windows. Intervals are conditional source-development
evidence with shared fitted producers, not independent calibration. Retain
undefined contrasts and their unsupported localities. Positive-easy <=2% and
zero-CV added harm 0 remain unchanged; no deployment promotion in this study.
The joint pilot has no zero-CV cases and cannot validate that protection.

Only the 12 already opened European source localities are used. Reserved model
selection, risk calibration, confirmation and DroneCrowd remain closed. Native
arm64 CPU4/inter-op1/workers0 with atomic model/optimizer/RNG checkpoints and
200-update heartbeats. No resource probing or multiprocessing. Local cost is
supported by preceding real fits; no new CREATE job or protected simulation
job change. No new trajectory/JEPA training or latent rollout.

Released detector tracks, image-pixel obs8/pred12 raw stride12 only; not raw t50,
verified online sensor labels, seconds, meters, physical safety, human gold,
true 3D, foundation or submission-ready evidence. Stage5C and SMC remain off.
