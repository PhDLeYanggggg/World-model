# Temporal Target Viability Audit

Registered before the pilot or new real-data readout. Parent: completed
TRAIN-identity quality extension, which repairs predominantly unselected MSE
without repairing selected risk. Its four strong controls remain authoritative;
this diagnostic does not supersede them with a weaker deployable comparator.

## Fixed Scientific Question

Do causal, frozen leaf assignments predict the time structure of reference and
neural-minus-reference errors beyond a global temporal mean and a leaf-local
whole-trajectory mean? Does averaging per-step positive harm silently change
the existing whole-trajectory risk target?

For each observed step, d(t) = neural error - reference error. Original harm is
max(mean_observed(d), 0). Mean_observed(max(d,0)) equals original harm plus a
nonnegative cancellation term. It is not an authorized replacement for harm.
The original B/H/R/ER/EH labels, selected actions and 2% risk rule remain fixed.

## Data and Estimators

Use the exact original 72 forests, three cost-head seeds and whole-recording
TRAIN/validation partitions within 12 exposed European development localities.
Reuse only the previously hashed source-training arrays and predictor lineage.
No independent selection, calibration, confirmation or transfer readout.

On TRAIN only, fit query-balanced analytic means of the 12-step two-channel
target (d(t), reference_error(t)), normalized by frozen TRAIN reference scale.
Routing is frozen. Missing steps never enter means. No new split, scaler,
hyperparameter or cutoff. These are diagnostic conditional-mean probes, not
neural training or fitted deployment policies.

Controls:
1. temporal_leaf: per-step weighted means within frozen leaves, then tree average.
2. rowmean_leaf: weighted row-mean target within the same leaves, repeated over time.
3. global_temporal: TRAIN weighted per-step mean without causal leaf conditioning.

A leaf without observed TRAIN labels at a step uses that TRAIN global mean;
report this fallback explicitly. A globally unobserved step remains unsupported
and NaN, never a safe zero. Inference accepts causal features, envelope and fitted
tables only, not future labels/masks. Future masks are for loss and offline score
support only. Unknown rows stay in the original policy readout/completion bounds.

## Readout and Decisions

Score two-channel and per-channel MSE on observed validation steps, with equal
row weight within each recorded query and the inherited source/recording/query
weights. Report unsupported observed steps separately. Report TRAIN, validation,
full-label validation, and frozen selected validation without selecting any model
or using future completeness to filter a policy. Store sufficient scalar fields
for independent aggregate verification and the hash of every reconstructed table.

Report locality-block paired 3,000-bootstrap intervals (seed 20261005), not
independent-window intervals. The advance-to-supervised-auxiliary-design screen
requires both temporal-leaf MSE contrasts to have upper CI below zero, including
the signed-error channel and the complete-label validation cohort. This is a
diagnostic screen only; empty/unsupported contrasts fail. No policy promotion,
safe-risk claim or world-model success follows even if it passes.

Do not infer that a better temporal target predictor necessarily reduces
whole-trajectory selected harm. If the diagnostic fails, retain the negative
result and separate sparse support, source shift and target information failure.

## Execution and Reproduction

Native arm64 Python, four compute threads, no DataLoader multiprocessing.
First pilot one complete head; run all 72 if memory/runtime are feasible. Refit
each analytic probe exactly and replay predictions. Check frozen original
forecasts, targets, actions and completion statistics against sealed receipts.
Resume by immutable per-head reports and recomputation, not by replacing runs.
Checkpoint report progress and heartbeat; cap 12 hours and 20 MiB light output.
No new numerical cache below the existing disk reserve. Remote CREATE timeout
is an observation blocker, not evidence that remote jobs failed; no remote work
or new model checkpoint is needed for this local diagnostic.

The separately implemented masked auxiliary loss is unit-tested engineering,
not a trained neural result. It preserves the original primary objective and
uses query-balanced observed-step supervision, with absent labels detached and
zero gradient only in loss masking, never fabricated outcome labels.

All findings are image-local detector-silver obs8/pred12 at rawstride12;
not seconds, metric, human gold, physical safety, true 3D or foundation evidence.
Intervals are nominal exposed-development diagnostics, not search-adjusted.
Stage5C and SMC remain off.
