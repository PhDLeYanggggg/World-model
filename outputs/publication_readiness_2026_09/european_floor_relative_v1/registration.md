# Reference-Consistent Gain/Harm Learning

Registered before new fitting or new held-out development readout, 2026-09-25.
This is a source-development target ablation, not independent confirmation.
The preceding default-action diagnosis is binding; no seed/mode winner is chosen.

## Hypothesis

The controller should learn whether a neural candidate improves the action it
actually replaces. Compare CV-relative and protected-floor-relative benefit/harm
targets under identical causal inputs, model families and update budgets.
Keep the protected damping policy fixed on the eight excluded localities and
keep both registered cross-moment modes, three seeds and two event targets.

## Honest Floor Targets

Each fit uses four localities, split into the same two pre-existing halves of
two localities. Fit the floor's ordinary utility head and cross-moment risk head
on one half, predict only the opposite half, and reverse. All eight outer
localities are excluded from this producer chain. Damping itself is deterministic
and causal. Shared inner utilities:18; inner risk heads:36 per mode. The risk
heads use their own half's CV easy cutoff and training-only preprocessing.

Use these out-of-source floor decisions on the four fitting localities. Use the
unchanged full-four-source frozen floor on the eight held-out development
localities. Record the two-to-four producer-size shift; do not mistake it for
identical training distributions. No in-sample floor target is admitted.

## Matched Target Experiment

For each fold/seed/event/mode, fit two utility heads and two risk heads:
CV-target utility/risk and floor-target utility/risk. All use the same380 causal
features: past geometry, neural candidate, protected-floor rollout, explicit CV
rollout and causal floor decision. Both targets use the same training-only CV
cost scale, supported rows and source-balanced sampler. Use the same union
rollout envelope max(distance(CV,neural),distance(floor,neural)) in both arms.
This holds feature information and capacity fixed while changing targets.

Utility has an envelope-bounded two-cost MSE objective. Risk uses the same
cross-moment hurdle objective and mode-specific batch/fitting normalizer.
Initialization uses each target's fitting prior by the same rule; the first
layer seed, architecture and source sampler are matched, but output priors are
not falsely described as identical. Fixed normalization uses the first40 batches
from a cloned fitting sampler. Unsupported normalization fails explicitly.

Easy-event membership remains positive CV error below the original fitting-only
CV easy cutoff. Floor-target risk predicts floor error mass and incremental
neural harm within that unchanged event; it does not redefine easy membership.
Evaluation easy/hard cutoffs remain those of the original four-source fit.
Zero-CV cases remain separately reported. This one-factor target experiment
does not pretend to add a new zero-support certificate or relax the2% limit.

Each head has width64,2,000 updates, batch256, learning rate0.0003, gradient clip5.
Total new heads234:90 inner-floor heads and144 main heads,468,000 updates.
No new trajectory predictor is trained. Store every checkpoint, trace, sampler,
preprocessor, replay and score bank. Pilot100 updates, then resume the same head.
CPU4/inter-op1/workers0 in native arm64;10GiB free-space guard.

## Frozen Readout

Four factorial policies share the actual protected-floor fallback:
CV utility/CV risk; floor utility/CV risk; CV utility/floor risk; floor utility/floor
risk. The old frozen rebased policy is a fifth control. All require observed
motion, positive predicted utility, positive predicted event mass and predicted
harm at most2% of predicted event mass. No threshold is selected by readout.
Fit both modes and freeze both complete decision banks before either new
evaluation. No reserved selection/calibration/confirmation role is opened.

Report all/easy/hard/complete ADE against the protected floor, matched CV-target
control and CV; endpoint FDE, switch rate, unknown-label interventions,
zero-reference harm, tails and worst locality. Reproduce the preceding rebased
control. Use3,000 paired locality-bootstrap draws, seed39271, on the fixed eight
localities per fold. Report undefined support rather than discarding localities.
Three seeds and repeated folds are dependent development views, not independent
confirmations. No winner selection or deployable-model claim from this readout.

## Boundaries

EuropeanSquares released detector tracks, image pixels, obs8/pred12 rawstride12.
No future endpoint in inference, test goals, central velocity or held-out
normalization. No metric, seconds, human gold, physical safety, true3D or foundation
claim. Historical Stage37 is not recertified. Stage5C and SMC remain off.
Source-cross-fit targets and code checks do not themselves prove efficacy.
