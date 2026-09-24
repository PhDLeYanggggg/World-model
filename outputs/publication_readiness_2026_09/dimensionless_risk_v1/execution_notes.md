# Execution and Verification Record

## Actual Execution

The local arm64 `.venv-pytorch` environment completed the full matrix. No remote
job was submitted. The current CREATE queue was not queried in this experiment;
the earlier saved authentication failure is not evidence that remote jobs are
absent. No simulation-project process, environment or storage was modified.

Environment: Python arm64, NumPy 2.4.6, PyTorch 2.12.0, scikit-learn 1.8.0,
four compute threads, one interop thread, no DataLoader workers. This is actual
scikit-learn risk-head training with frozen neural forecasts, not new Torch
trajectory training or a fallback passed off as neural training.

| Phase | PID | Observed result |
|---|---:|---|
| 16-tree pilot | 94843 | Clean exit at the registered resumable checkpoint |
| Full fit / decisions / readout | 94947 | 72 fits, 188,388 query/action/seed instances; exit 0 |
| Full decision replay | 97707 | All arrays and decision records exact; exit 0 |
| Separate arithmetic | 97713 | 72 fits, 36 pairs, 753,552 constraints, 600 reductions; exit 0 |
| Full aggregate replay | 98518 | Entire analysis exact; exit 0 |

The full main-process event interval is 2,478.776 seconds (41.313 minutes).
Cumulative fitting-loop time, including the resumed pilot, is 1,666.767 seconds
(27.779 minutes). Full decision replay takes a 729.778-second event interval;
aggregate replay takes 57.166 seconds. These are observed event/fit times, not
end-to-end cold-start or online deployment-latency benchmarks. Memory checks
showed continuing CPU work and roughly 8 GB resident memory, not a measured peak.
There was no runtime hang, model-budget reduction or quick-run substitution.

The solver emitted internal diagnostic lines but the processes completed normally.
Returned feasibility and optimality statuses, including all fallbacks, are kept.
No failure status was silently relabelled as an optimum.

## Identities

- Experiment identity SHA256:
  `bfd417125a50724e03e4dac9fb70ed23aa54a00a34930adb959b3fde50033a5f`.
- Frozen decision manifest SHA256:
  `7cd1af74a55af48598b86219f85e0cd80dad663bef8bebec2a7a2f1fc60aeee2`.
- Aggregate SHA256:
  `122a42cdf729e52970e902847996f608d352c8ac12ad9c4d761bf78b13f35808`.

Training source files, registration, configuration, old predictor/head receipts,
source inputs and producer dependencies are hash-bound. Every new fit has the
registered 128 trees and 768,000 source draws, with zero unknown-label draws.
The paired arms have the same supervision, draws, known-label mask and source
sites. An outer source site is excluded from each relevant producer chain.

## Checks That Passed

- Deterministic 16-to-128-tree resume regression.
- Full decision replay, including metadata-unit relabel prediction checks.
- Full aggregate replay, with identical analysis hash.
- Separate same-executor arithmetic: model identities, 118 features considered
  per tree split in both arms, paired supervision/draws, four primary risk
  constraints per query, count-failure truth and scene reductions.
- The 41-test training/risk/normalization batch.
- The 63-test native-metric/prefix/intake/precision batch.
- State-to-analysis hash/count/rounded-metric consistency.
- Tradeoff figure rendered and visually inspected; text and axes are readable.

The test batches overlap. They are not 104 unique tests and the unchanged full
legacy suite was not rerun. Separate arithmetic is not an independent researcher,
an exhaustive solver proof, or a fresh independent predictive test. Exact replay
checks implementation reproducibility, not scientific validity by itself.

## Result Sources and Remaining Limits

Risk-head training, fixed decisions, readout and separate arithmetic are
`fresh_run`. Frozen input/predictor/target assets and exact replay are
`cached_verified`. New trajectory-predictor training, external predictive
readout, independent calibration, confirmation, and exhaustive global solver
optimality are `not_run`, with their scope/reasons in the conclusions.

Observed aggregate utility is better under dimensionless population policies,
but neural easy preservation fails. The restrictive Transformer result is only
a development signal; its contrast with old strict includes zero. No policy is
selected for deployment from this readout.

All four SDD physical sites remain development-exposed. Source exclusion is not
independent confirmation. IMPTC remains quarantined with outcomes unread;
DroneCrowd remains closed. Unit/protocol claims remain annotation pixels and
native annotation steps, with no metric/seconds, true-3D, foundation-model,
Stage5C or SMC claim.

## Recovery and Repository Scope

Reproducible commands and required local private assets are listed in
[operation_zh.md](operation_zh.md). This is not a claim that a clean public clone
contains the private data/checkpoints for one-command retraining.

Source/feature/decision/weight caches remain under the ignored private data tree.
Only code, aggregate reports/metrics and the generated aggregate SVG are submitted.
The existing unrelated staged changes are preserved. The aggregate JSON is about
2.35 MB and contains metrics and fit receipts, not trajectory tensors or weights.
The numerical-precision repair is a separate experiment and is not mixed into
these fits. All required fitting and replay processes finished before close-out.
