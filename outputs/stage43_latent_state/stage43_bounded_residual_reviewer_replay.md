# Stage43-AO Bounded Residual Reviewer Replay

- source: `fresh_stage43_ao_bounded_residual_reviewer_replay`
- result_source: `fresh_replay_from_frozen_policy_artifact`
- gate: `11 / 11`
- verdict: `stage43_ao_bounded_residual_reviewer_replay_pass`
- reviewer replay passed: `True`
- policy hash match: `True`
- replay max abs diff: `0.00000000`

## Replayed Metrics

- all: `38.00%`
- t50: `26.96%`
- t100 diagnostic: `0.00%`
- hard/failure: `37.71%`
- easy degradation: `0.00%`
- switch rate: `63.08%`

## Replay Diff

| metric | expected | replayed | abs diff |
| --- | ---: | ---: | ---: |
| all | `38.00%` | `38.00%` | `0.00000000` |
| endpoint | `44.21%` | `44.21%` | `0.00000000` |
| t50 | `26.96%` | `26.96%` | `0.00000000` |
| t50_endpoint | `35.59%` | `35.59%` | `0.00000000` |
| t100 | `0.00%` | `0.00%` | `0.00000000` |
| hard_failure | `37.71%` | `37.71%` | `0.00000000` |
| easy | `0.00%` | `0.00%` | `0.00000000` |
| switch_rate | `63.08%` | `63.08%` | `0.00000000` |

## Boundary

- This is a reviewer replay from the frozen policy artifact, not a new threshold search.
- Dataset-local/raw-frame 2.5D only.
- Future labels are eval/loss only; no metric/seconds claim; no Stage5C; no SMC.

## Gate

| gate | passed |
| --- | --- |
| frozen_policy_artifact_present | `True` |
| policy_hash_recomputed | `True` |
| feature_schema_matches_checkpoint | `True` |
| cache_row_hashes_match_prior | `True` |
| checkpoint_hash_matches_freeze | `True` |
| stage43_m_report_hash_matches_freeze | `True` |
| checkpoint_not_tracked_by_git | `True` |
| replay_metrics_exact | `True` |
| replayed_policy_safe | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
