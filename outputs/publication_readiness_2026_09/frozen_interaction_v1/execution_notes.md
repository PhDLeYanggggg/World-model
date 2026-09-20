# Frozen Interaction Comparison: Execution And Evidence Boundaries

## Provenance

The design was committed as `2fa64646` before new control scoring. It was
specified after seeing historical v6 outcomes, so it is a post-hoc development
diagnostic, not an independently preregistered confirmatory experiment.

- `cached_verified`: original native-arm64 Transformer/EqMotion checkpoints,
  cost heads, fitted preprocessing, plans and development exports. Original
  artifact lineage, split contract and source hashes remain checked.
- `fresh_run`: new risk-only, unary-geometry and full-joint decisions on the
  existing frozen forecasts, their future-label scoring and independent
  row/aggregate checks. These decisions precede the separate label API.
- `not_run`: new model fitting, new primary-metric training, independent risk
  calibration, new-scene confirmation and physical-scene bootstrap intervals.
  The comparison uses two previously explored recordings of one physical site.

The fixed task observes eight supplied annotation steps and predicts twelve.
The supplied offline historical annotations can themselves contain interpolation
using later annotation controls. Past-indexed access therefore does not establish
strict sensor-as-of availability. Explicit future targets, target validity and
endpoints do not become inference features. No goals are rebuilt from test data.
Dataset-local coordinates and raw annotation indices have no new physical-unit
or seconds interpretation.

## Recovered Original Code, Not A Waived Check

Preflight correctly refused an altered model-code binding before any inference.
The current supervised backend contains a later eight-line architecture dispatch
addition. The runner recovers its exact required bytes from commit `5f59acf2`
into an ignored private runtime mirror, checking SHA256 before writing/loading.
It leaves the current checkout untouched. Only that verified code location is
relocated in the contract; data roles, ancestors, protocol and required hashes
are unchanged. Tests reject wrong bytes, outside-root paths and data relocation.

Every newly admitted query must exactly reproduce the old agent identity,
requested horizon, scale, label eligibility and floor/uncontrolled forecast
error. Any mismatch is an error, not a silently accepted "reproduction". Legacy
control differences are counted separately; new and old solver versions must
not be conflated. All three new arms share the same new numerical checks.

## Runtime And Recovery

CPU threads are four, inter-op threads one, and DataLoader workers zero. The
Transformer retains CPU; EqMotion retains MPS. No NumPy or device fallback is
enabled. Both 128-query pilots are part of their full runs, not replacement
small studies. A receipt follows each completed batch, and interruptions resume
only from hash-verified batches. Completion binds all receipts and the report.

EqMotion first failed at MPS initialization in the restricted sandbox, before a
new query completed. The message referred to unsupported macOS, although the
host reports macOS 15.3.1 and arm64. The same command outside the sandbox then
completed 128 actual MPS queries in 27.60 seconds with exact historical errors,
and the full run resumed those receipts. This observed permission/runtime
phenomenon is distinct from the old x86_64 Conda/MKL/OpenMP problem. No source
data, precision policy or predictor was changed to make the replay pass.

The progress observer never kills or restarts the main job. A permission-limited
PID check is recorded as unknown, not dead or stuck. Completion requires the
main runner's terminal result and verified receipts. Completed resume rebuilds
and compares aggregates without new inference or rewriting completed artifacts.

## Independent Arithmetic Check

The first new aggregate checker incorrectly required a float64-level `1e-12`
identity between the displayed full objective and its decomposition. Historical
cost-head gain means retain float32, while coefficient sums accumulate in
float64. A real Transformer batch exposed this reporting mismatch; the maximum
over that completed family is `2.3283064365386963e-08`. A small regression fixture
reproduces the different reduction paths. The independent checker now bounds
only this display discrepancy using a float32 summation/division roundoff bound
and the fixed nonnegative selected-gain condition. It separately checks the
coefficient objective and original-unit primal/dual certificate at the original
tolerance. Forecasts, choices, solver code, policies and saved results are not
changed. This is not a license to accept an inferior optimum or evidence of a
predictive difference. Coefficient-level proxy advantages remain distinct from
actual ADE/FDE contrasts in the analysis.

## Statistical Limits

Each fixed combination repeats the same development population. The 24
combinations are not 24 independent scenes. Overlapping agent windows are not
independent calibration samples. Three training seeds describe fit variability
only. Scene-level uncertainty remains unavailable with one physical site, and
no window bootstrap is substituted.

An exact selected-agent count is not necessarily an equal changed-forecast count,
equal scored-label coverage or equal realized risk. Missing futures remain in
the decision population and are disclosed separately from scored-label cohorts.
The algebraic pair-proxy bound concerns the constructed objective, not true
collision probability, realized forecasting harm or physical safety.

All fixed combinations, including negative and zero contrasts, must be retained.
No result selects a new model, threshold, metric or deployment. Stage5C and SMC
remain disabled. The overall research goal is not completed by this diagnostic.
