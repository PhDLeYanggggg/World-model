# Engineering Verification and Test Isolation

Date: 2026-09-16. Result source: fresh_run. This is not an independent forecasting evaluation.

## Results

- Full suite: **1,870 passed, 1 failed**, 4,393.92 seconds (73 minutes). Exit status 1; not all green.
- Final focused suite: **32 passed**, 1.86 seconds. Includes the two new scene-membership tests and the Stage42 output-isolation repair added after the full run was collected.
- New-reader checks: 142,402 indexed views checked, 144 real-window future-corruption cases and 36 scene-membership/input corruption cases passed. This is not a complete experiment no-leakage certificate.
- Seven WorldCore checkpoint hashes and twelve direct input-cache hashes still match the earlier audit. Identity does not establish independent evaluation.

Commands and counts are also recorded in `verification_record.json`. The full suite was not rerun after the two added scene tests; their final version was covered by the focused run. No training process was stopped because it was slow.

## Remaining Full-Suite Failure

`tests/test_worldcore_data_lake.py::test_goal_completion_audit_keeps_active_ingest_not_complete` also fails alone. Its temporary fixture lists 31 blocked, not-yet-uploaded datasets in the download manifest but does not create a manual handoff covering them. The transfer guard returns `failed_safe_transfer_policy_crosscheck`, specifically `manual_handoff_missing_blocked_not_uploaded_dataset_ids`; the test expects a non-failing transfer-policy status.

This is in the pre-existing staged data-lake work, not the new M3W reader/runtime code. That code/test was left unchanged. The follow-up is to resolve whether the fixture should construct the required handoff or assert the fail-closed result, without relaxing the production transfer guard. No upload or download was executed by this diagnosis.

## Legacy Test Side Effects

Several legacy integration tests write reports and research state in the repository rather than a temporary directory. One also runs three training seeds, five epochs and bootstrap evaluation. Those replays use historical data and are not new confirmatory evidence.

The Stage42 source-level integration test now runs in a temporary directory and verifies that root research state is unchanged. The remainder of the legacy suite is not yet fully isolated.

After the full run, 94 changed report/log files were preserved in ignored local quarantine and the prior report versions recovered. Six identified historical research-state sections and their current-stage markers were recovered while retaining this round's new research record. The two pre-existing Stage43 workspace changes were retained using pre-test snapshots, not reset to Git. All files in the data-lake snapshot remain unchanged. Checkpoint contents did not differ from their Stage43 pre-test snapshot; the seven Stage44 checkpoint identities were also rechecked.

Local recovery manifests are under `data/stage_cvpr2027_causal/pytest_generated_reports/` and `pytest_generated_snapshot_outputs/`; the test-written root ledger is preserved as `pytest_generated_root_state.json`. These artifacts, raw data, caches and checkpoints are not committed.

The unrelated staged diff fingerprint remains `c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`. A scoped commit must preserve it.

## Research Boundary

Runtime probes and regression tests do not establish a new deployable model, corrected external improvement or submission readiness. The primary scientific protocol and independent confirmation remain pending. Dataset-local/raw-frame claims only; Stage5C and SMC remain disabled.
