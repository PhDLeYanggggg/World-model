# Frozen Cost-Bank Factorization: Fitting-Only Evidence

## Material Passport

Code experiment / fixed inference control. Status: real run and separate exact replay complete.
Fresh: current fitting inference, controls and metrics. Cached_verified: frozen models,
causal feature packets and prior negative reports. Not_run: new training, held or
independent evaluation, world-dynamics learning and deployment.

## Execution

Run PID 17568, exit 0, completes 144 groups in 262.978860s; peak RSS 6,547,881,984 B.
Separate PID 18293, exit 0, exact replay takes 262.283743s.
All original policy choice hashes and complete metrics match the preceding frozen
fitting readout. Mixed-with-itself neutral costs, choices and ranks are exact.
No optimizer step or new tree. Three head seeds do not create three forecasting pipelines.

## What Stops The Neural Candidate?

| Cost-estimator family | Eligible neural query views | Mixed neural gate passes | Mixed neural choices | Lost to physical competition | Dedicated neural expert passes | Dedicated passes rejected by mixed gate |
|---|---:|---:|---:|---:|---:|---:|
| forest | 894975 | 437 | 271 | 166 | 85544 | 85117 |
| neural | 894975 | 35733 | 22246 | 13487 | 23993 | 13953 |

Among dedicated-expert passes rejected by the mixed estimator, overlapping causes:

| Family | Nonpositive gain | Zero reference | Zero easy-reference | All-harm gate | Easy-harm gate |
|---|---:|---:|---:|---:|---:|
| forest | 103 | 0 | 0 | 82224 | 85112 |
| neural | 10836 | 0 | 55 | 7636 | 2102 |

Counts overlap; do not sum them as exclusive causes. Fixed-order exclusive
counts and full co-occurrence bit patterns are retained in summary.json. Gate
diagnostics use predicted costs and causal availability only, never future labels.

## Fixed Policy Results

| Family | Policy | Known ADE gain vs floor (%) | Known FDE gain (%) | Required recording risk support | Undefined easy views | Selected (unknown) |
|---|---|---:|---:|---:|---:|---:|
| forest | physical_neural | +0.000111 | not_estimable | 330/444 | 104 | 7026 (80) |
| forest | mixed_neural_restricted | -0.000086 | not_estimable | 113/444 | 297 | 437 (6) |
| forest | physical_only | +0.000243 | not_estimable | 333/444 | 106 | 5764 (70) |
| forest | neural_only | +0.071258 | not_estimable | 165/444 | 209 | 85544 (836) |
| forest | factorized | +0.071514 | not_estimable | 285/444 | 104 | 88680 (883) |
| forest | suppressed_neural_expert | +0.071271 | not_estimable | 159/444 | 219 | 85117 (830) |
| forest | competition_lost | -0.000003 | not_estimable | 67/444 | 359 | 166 (6) |
| forest | neutral | +0.000111 | not_estimable | 330/444 | 104 | 7026 (80) |
| forest | floor | +0.000000 | not_estimable | 0/444 | 444 | 0 (0) |
| neural | physical_neural | +0.678064 | not_estimable | 77/444 | 49 | 70561 (853) |
| neural | mixed_neural_restricted | +0.268009 | not_estimable | 124/444 | 81 | 35733 (380) |
| neural | physical_only | +0.671048 | not_estimable | 73/444 | 50 | 77771 (901) |
| neural | neural_only | +0.579900 | not_estimable | 126/444 | 95 | 23993 (314) |
| neural | factorized | +1.122806 | not_estimable | 57/444 | 42 | 89911 (1076) |
| neural | suppressed_neural_expert | +0.442730 | not_estimable | 144/444 | 114 | 13953 (189) |
| neural | competition_lost | +0.088324 | not_estimable | 176/444 | 117 | 13487 (138) |
| neural | neutral | +0.678064 | not_estimable | 77/444 | 49 | 70561 (853) |
| neural | floor | +0.000000 | not_estimable | 0/444 | 444 | 0 (0) |

Factorization uses two estimators: 256 trees instead of 128, or 52,618 neural cost
parameters instead of 26,309. It changes learned costs, capacity and inference work,
not just competition. Forecasters, paths, normalization and the 2% gate remain fixed.

## Paired Factorized Minus Mixed

| Family | Full known utility (pp), 95% CI | Full completion-lower (pp), 95% CI | Matched known utility (pp), 95% CI |
|---|---|---|---|
| forest | +0.071403, [+0.018321, +0.137631] | +0.069396, [+0.017579, +0.133797] | +0.003750, [+0.001153, +0.007120] |
| neural | +0.444742, [+0.147436, +0.817331] | +0.324532, [+0.019772, +0.695868] | +0.164349, [+0.038002, +0.354595] |

These are 3,000 locality-resample fitting-resubstitution intervals after context
and controller-seed averaging, not held generalization or independent confirmation.
Completion utility is normalized by fixed known floor-error mass, not a fully
observed population-error denominator. Count matching is utility-only. All unknown
outcomes, required recordings and empty risk estimates remain.

## Interpretation Boundary

For the same mixed estimator, neural gate passes that lose to physical actions
identify competition exactly. Differences between the dedicated neural expert
and mixed estimator demonstrate bank-dependent learned decisions, not a unique
causal attribution to class imbalance, architecture, objective or data sufficiency.
Even a positive fitting effect does not promote deployment or justify a new held
threshold search. Earlier failed selected-set recalibration remains negative.

The experiment uses EuropeanSquares SOURCE detector-track midpoint pixels,
obs8/pred12/stride12 raw frames, not SDD human gold, metric/seconds, true 3D or
foundation evidence. No future endpoint, central velocity, test goals or future
support masks enter inference. Independent roles and DroneCrowd remain closed.
No Stage5C execution or SMC. Submission readiness remains false.
