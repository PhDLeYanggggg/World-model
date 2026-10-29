# Verification and Evidence Status

## Fresh Execution

| Phase | PID | Exit | Scope | Seconds |
|---|---:|---:|---|---:|
| Pre-repair frozen Jacobian diagnosis |50837|0|72 fit views|61.038337|
| Rational real-data pilot |51761|0|200 updates, direct/resume equivalence|12.075372|
| Rational continuations |51994|0|18 fits,36,000 updates|190.943200|
| Fixed evaluation |52493|0|144 fit/held views, eight policies|122.317541|
| Post-repair Jacobian verification |53106|0|72 fit views|64.016528|

All listed processes reached terminal completion; no active training was left
behind. Whole TRAIN packets streamed read-only from existing approved CREATE
assets with identity/SHA checks. No new scheduler job. The original held
partition was discarded before both gradient diagnoses, while the registered
model readout used already exposed internal held recordings only after all
new final checkpoints froze. No independent data role was opened.

## Passed Engineering Checks

119 scoped regression tests passed in5.58s. They cover fixed-cohort accounting,
unknown-outcome retention, gradient behavior, extreme anchors, causal model
inputs, bounds, initial/control matching, exact checkpoint resume, protected
outputs, readout arithmetic and negative-report preservation. An extreme-anchor
unit test initially detected quotient-gradient cancellation; the equivalent
stable tail formula fixed it before registration and any real-data training.

The complete legacy suite was not run: some older tests train and overwrite
frozen research outputs. This report does not claim all repository tests pass.
Research-state JSON parses; scoped whitespace validation passes. The final
private comparison figure was visually inspected and is legible without overlaps.
Its known-violation counts are not safety rates; undefined views are explicit
in the result table and failure analysis.

All18 real initial objective values, source/query sampler hashes, parameter
counts and protected base tensors match archived controls. All seven old
predictors reproduce their old prediction hashes, policies and scores. No old
fit is called fresh training. Gradient and inference check commands passed:

```bash
.venv-pytorch/bin/python -m scripts.audit_m3w_harm_jacobians check
.venv-pytorch/bin/python -m scripts.report_m3w_rational_harm --check
.venv-pytorch/bin/python -m scripts.check_m3w_rational_gradients check
```

The report verifier invokes the complete rational readout/checkpoint validator.
Check mode replays immutable evidence; it does not claim another training run.

## Evidence Hashes

- Pre-repair Jacobian summary: `0e1e6a488ac904db64900c5cf6c3f1da7757611f5d25a9903b764c9e4911c8e8`.
- Rational registration: `e16fac539928dd8c92ceff3d3cbe7fa533337130fd9235ffe71ab9cf04adf875`.
- Rational readout summary: `1fe3a21b1cabfdf8d5a4c8f3771b7d454628e9bda913476dad93e66ad1cff623`.
- Post-repair gradient summary: `90f83c41814e576243b03be9eff5664965addbd96fa486ebefd6da630f70019c`.

## Scientific Gate and Storage

The registered advancement gate FAILED. Numerical repair and complete execution
are not independent validation, deployment safety or submission-candidate status.
Three-thousand-draw intervals are exploratory locality bootstrap, not an IID
window analysis or multiplicity-adjusted formal confirmation.

Public output is aggregate JSON/Markdown and source code. No public file in
these new result directories exceeds1MiB. Raw data, caches, images and model
checkpoints remain private and untracked. The unrelated staged-change fingerprint
was preserved: `c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Only explicitly owned paths are committed; no add-all/reset/revert operation.

Obs8/pred12, raw stride12, detector-track pixel/dataset-local evidence. No
metric, seconds, human gold, physical safety, true3D or foundation claims.
Historical Stage35/37/43/44 evidence remains exploratory. Stage5C/SMC disabled.
