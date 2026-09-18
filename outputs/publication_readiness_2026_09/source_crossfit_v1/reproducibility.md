# Reproducing the Training-Side Cross-Fit Experiment

## Fixed Scope

Pre-fit registration: `a89299fd`, `configs/m3w_source_crossfit_v1.json`.
Registration SHA256:
`f1ea040f58610ecb7f8f2380f72c92dd7a745767d05855292e572421e0dd0a14`.

Four internal source-site folds, seeds 17/29/43, twelve cold-start models and
120,000 total optimizer updates. Each model uses 2,000 constant-rate updates
followed by 8,000 cosine-decay updates. The terminal step is fixed, not selected
from held scores. The 100-update pilot is included, not additional training.
Training data and code are reused after hash checks; old fitted parents are not.

The 15,430 query population contains 29 recordings and 545 recording-scoped
agents at four historically explored sites. Query windows overlap. Each
candidate producer, normalizer, loss scale and hard-label cutoff excludes its
query's physical site. Bookstore and all main evaluation roles are excluded
from inference and fitting. Shared dataset construction loads verified existing
assets; this is a role-use restriction, not a claim of zero file access.

The experiment uses complete stationary-history source queries, mask/geometry
inputs, eight observed and twelve predicted annotation steps, stride 12.
It does not rerun the full mixed-motion main benchmark or raw-frame t+50.
The model predicts a bounded deterministic trajectory, not latent generation.

## Commands

Run from the repository root using the native arm64 environment:

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_crossfit.py --registration configs/m3w_source_crossfit_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_crossfit.py --registration configs/m3w_source_crossfit_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_crossfit.py --registration configs/m3w_source_crossfit_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_crossfit.py --registration configs/m3w_source_crossfit_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_crossfit.py --registration configs/m3w_source_crossfit_v1.json
.venv-pytorch/bin/python scripts/diagnose_m3w_crossfit_costs.py --registration configs/m3w_source_crossfit_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_crossfit.py tests/test_m3w_source_modality_continuation.py tests/test_m3w_source_cost_dynamics.py tests/test_m3w_recording_diagnostic.py tests/test_m3w_crossfit_cost_diagnostic.py
```

An interrupted training invocation resumes from the existing registered
checkpoint with optimizer, sampler and Torch RNG state. Do not run a second
copy concurrently. Running again after completion must perform zero new updates.
`--replay` reforecasts both the training complement and held queries and demands
exact array equality. The verifier checks fold lineage, label construction,
bounds, target exclusion from input and completed-resume immutability.

The entry path rejects an incompatible Apple Silicon runtime before importing
Torch. The recorded environment is Torch 2.12.0, NumPy 2.4.6, CPU threads 4,
inter-op threads 1 and DataLoader workers 0. Checkpoints and heartbeat events
are written every 200 updates. This experiment does not require a CREATE job;
current CREATE resource availability is not established by this local run.

## Evidence and Files

- `input_checks.json`: data identities, per-fold training/held hashes and role guards.
- `report.json`: twelve training receipts and three aligned OOF-label archive hashes.
- `replay.json`: exact prediction replay receipts; not an accuracy claim.
- `analysis.json`: registered conditional estimates and descriptive controls.
- `verification.json`: independent lineage checks and immutable artifact hashes.
- `cost_attribution.json`: explicitly post-hoc training-side error decomposition; no model or policy selection.
- `results.md`, `site_metrics.csv`, `comparison.svg`: lightweight aggregate outputs.

Private files remain under `data/stage_cvpr2027_experiments/source_crossfit_v1/`:
logs, random-initialization parents, terminal checkpoints, per-query predictions
and OOF labels. They are not distributed in Git. Reproduction requires the
matching locally acquired source assets and admitted data manifests. A public
repository clone alone is not a self-contained dataset/checkpoint reproduction.

## Interpretation

Primary gain is the ratio of equal-site mean past-normalized ADE, not a mean of
site percentages. Seed averaging is averaging errors, not forecast ensembling.
Window-weighted results are a sensitivity analysis. The 2,000-resample intervals
are descriptive: four explored sites and shared training folds do not provide
independent confirmation. The full-four-site in-sample reference is
`cached_verified`; differing fit populations and normalizers confound that
comparison. A future-informed binary oracle is a diagnostic, never an input.

Easy percentage degradation is undefined where baseline ADE is zero; absolute
annotation-pixel harm must remain visible. Finite bounded outputs do not prove
physical validity or a safety guarantee. Loaded-target poisoning does not
establish strict real-time availability of historically interpolated annotations.

No risk head is fitted here. A subsequent risk-head validation split needs its
own upstream producer exclusions; see [method and limits](method_and_limits.md).
No independent calibration, new deployment, metric/seconds, true-3D, foundation
or human-gold claim. Stage5C execution and SMC remain off.

## Completed Receipt

Training PID 48956 exited zero: 119,900 updates in the main invocation plus the
100-update pilot, 120,000 total. Main log span 5,784.63657 seconds, summed fit
5,725.36250 seconds. Replay PID 57231 exited zero and reproduced all twelve
models; analysis, verifier and post-hoc attribution also exited zero. The
verifier's completed-resume child PID 57374 added zero updates and preserved
55 artifacts. Ninety-six loaded-target poisoning checks passed. The figure was
visually inspected. Twenty-eight focused tests pass; full legacy suite not run.
Font-cache warnings did not prevent analysis or figure completion. A locale
error in a standalone hash command was resolved with `LC_ALL=C`; no training
or result file was changed by that command.

| Artifact | SHA256 |
| --- | --- |
| report.json | 64f38a57ea213f8c416a6d56948e5d7a58d7bd1609bed7f3e80b2848b3b5f785 |
| analysis.json | d7ca8d2e74b41ee05ab30fd638fdc43b4274327ab62be9ec2e5307e8ff758fdb |
| verification.json | 91af2f9fec199ae323042bf8f2db75b1dc2f6398dd281bbb8b44ffad66908b2b |

The main registration and its ten frozen dependency hashes remain unchanged.
The post-hoc attribution is new diagnostic code, not a revision of that analysis.
