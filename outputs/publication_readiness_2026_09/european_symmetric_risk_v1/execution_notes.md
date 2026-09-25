# Execution Evidence

## Provenance

Registration commit: `e7f839e1`. Scientific runner, model helper, config and
matrix stay unchanged after registration. Only reporting and plotting were
added after readout. All reports preserve the fixed matrix without selecting
a winner. Local date: 2026-09-25; times below are UTC.

`fresh_run`: 36 neural event-risk heads, 72,000 updates, their new decision
banks and source-development readout. `cached_verified`: 18 frozen symmetric
utility heads, 36 old neural-risk controls, 36 ridge-risk controls and the old
ridge decision banks; all referenced hashes checked. `not_run`: independent
reserved calibration/confirmation, deployment, new trajectory/JEPA training,
CREATE training, Stage5C and SMC. Disabled components are not results.

## Runtime

Native arm64 `.venv-pytorch/bin/python`, Torch2.12.0, NumPy2.4.6. CPU4 threads,
inter-op1, workers0; architecture checked before Torch import. No NumPy model
fallback, multiprocessing or GPU resource probing. Atomic checkpoints and
heartbeat every200updates. Training losses are available in
[training_losses.md](training_losses.md), not substituted for held-out metrics.

| Process | PID | Observed completion |
|---|---:|---|
| Real 100-update pilot, resumed inside budget | 34577 | 03:13:10, exit0 |
| Complete fitting and 48-view readout | 34631 | 03:13:59 to03:20:09, exit0 |
| Metric reconstruction and checkpoint replay | 35181 | 03:20:38 to03:22:39, exit0 |

Summed fitting-loop time is63.07495seconds; this excludes source assembly,
normalization, inference, bootstrap, solver and reporting overhead. It is not
the elapsed end-to-end run time. Observed main-process RSS was approximately
6.1GB, not a sampled high-water guarantee. Small risk-head fitting justified
local execution. No CREATE job was submitted. The last inspected remote
access record (September24) is an authentication failure, not evidence that
remote jobs stopped; no simulation project job or credentials were changed.

## Verification

- [verification.json](verification.json): complete metric reconstruction.
- [checkpoint_replay.json](checkpoint_replay.json):36 new checkpoints x4,096
  inference rows exact;512,000 matched supervised draws each, zero unknown draws.
- [accounting_audit.json](accounting_audit.json):144 causal pointwise decision
  banks reconstructed; joint budgets, counts and frozen control hashes verified.
- 185 tests across25 scoped files pass in2.47seconds; not the full legacy suite.
- No unknown, negative, undefined or failed-solver result is replaced by zero
  or removed to improve the reported conclusion.

`analysis.json` remains a local row-detailed artifact and is not committed.
Public `summary_metrics.json`, `risk_reliability.json`, reports and SVG contain
aggregate evidence only. Raw data, predictions, checkpoints, PNG and environment
remain excluded. Reproduction requires the licensed local source assets and
prior hash-bound artifacts; a Git checkout alone is not a complete dataset.

The unrelated staged index contained3019 entries before this update and must
remain byte-identical after the scoped commit. Its raw-diff SHA256 is
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

## Interpretation

Complete execution does not mean a successful safety repair. Neural easy-event
accuracy improves but all seeds violate the worst-locality easy gate. New
selected-risk diagnostics identify underestimation, not a completed calibration
fix. Source-development image pixels and raw frames only; no promotion, formal
submission, Stage5C, SMC, metric/seconds or foundation claim.
