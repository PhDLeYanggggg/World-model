# Motion Runner Heartbeat Repair

The v1 registered input build completed. The100-update real pilot saved its
checkpoint but failed when combining duplicate `pid` keyword arguments in the
heartbeat callback. A continuation saved step400 and hit the same callback error.
Neither invocation reached held-fold prediction or scored a model.

This is an implementation failure, not negative predictive evidence. Preserve
v1 source, registration, input cache and partial checkpoint. The v2 runner changes
only heartbeat dictionary construction. A regression test includes an existing
training PID. Register v2 before any completed prediction; all model/features,
folds, seeds, objectives and update budgets are unchanged. Use a fresh v2 output
directory; do not relabel the failed partial fit as a completed trial.

The extraction is repeated under v2 identity, although its numerical definition
is unchanged. Require array hashes to match the v1 build. No source images or
checkpoints are committed. This is not an OpenMP/runtime-architecture failure.
