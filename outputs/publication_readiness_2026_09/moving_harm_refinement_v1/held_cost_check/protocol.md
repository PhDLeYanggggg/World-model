# Post-decision held-cost observation during exact replay

This is diagnostic instrumentation, not another model or deployment policy.
All54refiners are already frozen and the complete held primary is not changed.
The same-region fitting audit motivates separating fitting reliability from
recording-level held cost errors. Register this observer against the completed
held summary before the separate replay. No hyperparameters or thresholds are
selected from its outputs.

The original registered verification first produces all causal cost outputs
and verifies the saved allocation bindings and feasibility. Only after each
allocation is fixed does the observer read evaluation labels, compute known
and bounded H/EH/reference masses, and retain every causal policy/recording.
The original choices and report are returned unchanged; their hashes and the
prediction hash must match before/after observation. Labels are not passed to
predict, preprocessing, eligibility, allocation or a learned component.

Retain all72model/job cells, four causal policies and all repeated recording
views. Pooled ratios are diagnostics, not independent calibration or per-record
safety certificates. Zero denominators remain undefined. The original replay
still checks all72cost outputs and all original policy scalars; the observer
adds no MILP rerun or claim of bitwise reoptimization. Its timing includes the
extra measurements. No additional training, reserved role, deployment,
metric/seconds claim, Stage5C or SMC.
