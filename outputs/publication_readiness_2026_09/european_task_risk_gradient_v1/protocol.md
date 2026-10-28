# TRAIN Constraint Gradient Diagnostic

## Material Passport

Run mode: post-hoc mechanism diagnosis of the completed failed decision-risk
experiment. Frozen TRAIN packets and 216 final heads are verified before use.
No independent selection, calibration, confirmation or development evaluation
input is read. This is not a new model-quality comparison.

## Fixed Calculation

For each of the 72 identities and three arms, replay the original fixed 128-query
known-TRAIN monitor using the original seed. Verify original final primary loss,
regret, both excess constraints and soft intervention before computing gradients.
Report the norms of primary, regret, all-risk and easy-risk parameter gradients,
their weighted ratios and cosines, plus soft versus original hard-rule actions.
Do not take an optimizer step or alter saved models.

Measure a candidate positive TRAIN-only constraint scale: 0.02 times mean reference
cost (all and easy respectively) in frozen TRAIN cost units. Average agents within
query and then queries equally, with equal locality weights. Use all known TRAIN
queries for this constant, not the monitor or evaluation data. A zero reference
mass is unsupported, not replaced by a convenient epsilon. Dividing an empirical
constraint by a fixed positive number preserves its feasible set but changes
optimization conditioning; it does not establish risk calibration or safety.

The complete 216-view audit is descriptive. Gradient imbalance, conflict and
soft/hard disagreement are possible mechanisms, not identified causal effects.
Any subsequent matched training repair needs its own fixed protocol. Existing
2% risk definitions, unknown-label bounds, development advancement requirements
and deployment remain unchanged. Raw data and model weights remain out of Git.

## Execution

Native arm64 CPU, four threads, zero loader workers. Reuse the already verified
streaming TRAIN path from CREATE with no local packet cache; no remote job or
write is needed. Save one immutable lightweight receipt per head and heartbeat
per identity; explicit resume reuses completed diagnostics after identity checks.
Original training and inference sources stay unchanged. Stage5C and SMC stay off.
