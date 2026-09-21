# Completed Cost-Head Resume: Dependency Integrity Repair

Status: engineering repair with synthetic training regression and read-only
verification of six real frozen heads. No new real-world training or inference,
metric amendment, independent calibration, test selection or deployment change.

## Reproduced Defect

The completed branch of `scripts/train_m3w_oof_cost_head.py --resume` checked
the head checkpoint hash, producer OOF exposure and source recording integrity,
then returned `already_complete_cached_verified`. Unlike its incomplete-fold
branch, it did not recheck the OOF archive bytes, fold receipts, feature identity,
report provenance or the completed head manifest.

A temporary synthetic training experiment reproduces this exact behavior:
two eight-update Torch forecasters produce OOF inputs for a ridge cost head.
After fitting is complete, a numeric training target in an archive is changed. The old entrypoint returns
success; the v2 entrypoint rejects the changed dependency. This is a completed
resume defect, not evidence that partial-fold recovery or all neural training
checkpoints are broken. The existing incomplete-ridge and neural-cost recovery
regressions also pass separately.

The defect is not evidence that the published six real heads were corrupted.
Their newly checked dependencies match the earlier frozen evidence. It does
not explain their conditional harm underprediction, nor rehabilitate any
negative forecasting result.

## Repair Without Rewriting Frozen Code

The original training entrypoint and bound training/evaluation implementation
bytes remain unchanged. Historical protocols bind those files, so silently
editing them would invalidate old provenance.

New `scripts/train_m3w_oof_cost_head_v2.py` delegates new and incomplete training
to that exact trainer. On successful completion it verifies and binds the
dependencies in `completion_v2.json`. On completed resume it verifies those
bindings without importing Torch, fitting, forwarding a model or rewriting
outputs. It preserves the native arm64 guard and sets numerical-library thread
limits before importing NumPy or Torch.

The new verifier checks:

- original arguments, protocol, full producer manifest and training source hashes;
- complete held-fold membership, recursive OOF exposure and original source data;
- every OOF archive/receipt pair, producer identity and feature/target schema;
- row roles, physical-scene labels, duplicate queries and feature identity;
- final report, head manifest, row counts, dimensions and checkpoint bytes;
- finite coefficients and exact reconstruction of fit-only mean/scale;
- validator source identity and dependencies again before reporting completion.

This is file/provenance consistency checking, not proof that a regressor is
optimal, generalizes, is calibrated, or has complete truthful historical-use
declarations. It is not a defense against an adversary rewriting all trusted
snapshots and code together.

The old entrypoint and old orchestrators remain available for exact historical
replay. Their completion status alone is **not** the strengthened check. New
registered ridge jobs should use v2. No old orchestration protocol was silently
migrated in this repair.

## Target-Identity Caveat

The earlier `oof_feature_identity` deliberately excludes targets. A changed
target archive accompanied by a rehashed receipt can remain feature-consistent.
The new completed-run receipt binds the whole archive before later resumes.
An explicit regression checks this distinction.

For old runs with no v2 receipt, the new training entrypoint refuses completed
resume rather than minting trust from whatever files are currently present.
The separate legacy verifier requires a previously retained, hash-pinned source
snapshot. It does not write v2 receipts into old training directories.

If an interruption occurs after the old trainer finishes but before the v2
receipt is written, the next invocation also fails closed. Preserve the outputs;
do not delete them, rerun automatically or make a receipt from unchecked current
bytes. Recover using independently retained bindings or an explicitly reviewed
reproduction. This narrow completion-boundary recovery limitation is retained,
not hidden as a fully automatic resume capability.

## Real Frozen Assets Checked

`scripts/verify_m3w_frozen_cost_completions.py` uses the already published
Transformer/EqMotion fit-forensics identities and report hashes as its anchors.
It checks seeds 17, 29 and 43 for both families: six ridge heads, each with
11,966 fitting row identities, 306 features and 13 direct dependency files.
All six exact normalizer checks pass and all target archives match their
previously retained bindings. The runner checks 142 shared source bindings.
The repeated rows are not additional independent observations.

Result source: `fresh_run` integrity verification on `cached_verified` fitting
assets. Cached training labels are read for shape/finite/schema checks; no new
outcome score or target prediction is computed. The audit neither deserializes
Torch checkpoints nor imports Torch. The exact old model source is read from
the verified mirror, not executed or substituted into the checkout.

Fresh execution: PID 53188, exit 0, 1.762 seconds. Completed replay: PID 53309,
exit 0, 1.555 seconds, identical analysis. These are integrity checks, not
training-speed measurements. No CREATE job was submitted; this work fits local
resources comfortably. All required sessions are terminal.

Analysis SHA-256:
`0bb9a0cdccb4349bbf3fb83e8e651f61cdae1cf4f55450d8becf5e77b9e7482e`.

## Reproduction

```bash
.venv-pytorch/bin/python scripts/verify_m3w_frozen_cost_completions.py --resume
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_completion.py tests/test_m3w_cost_validation_lineage.py tests/test_m3w_experiment_contract.py -q
.venv-pytorch/bin/python -m pytest tests/test_m3w_supervised_intervention.py::test_cost_head_cli_fold_checkpoint_resume_and_tamper_detection tests/test_m3w_neural_gain_harm.py::test_neural_cli_reuses_ridge_oof_and_resumes_across_processes -q
```

59 scoped tests pass in 6.54 seconds, including 13 new tests. Two additional
existing cross-process recovery tests pass in 17.35 seconds. No skips. The full
historical report-writing integration suite was not rerun. Tests use temporary
synthetic recordings and do not alter real research outputs.

The primary metric/target decision remains pending. Independent cost validation
still needs clean nested producers; calibration/confirmation data remain a
separate issue. This repair is necessary reproducibility work, not a neural
contribution or submission result. No Stage5C, SMC, metric/seconds-level,
true-3D or foundation-model claim.
