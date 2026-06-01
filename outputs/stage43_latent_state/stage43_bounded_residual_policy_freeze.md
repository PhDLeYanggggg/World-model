# Stage43-AN Bounded Residual Policy Freeze

- source: `fresh_stage43_an_bounded_residual_policy_freeze`
- result_source: `fresh_freeze_from_statistically_confirmed_stage43_am_candidate`
- gate: `11 / 11`
- verdict: `stage43_an_bounded_residual_policy_frozen`
- policy frozen: `True`
- policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
- checkpoint tracked by git: `False`

## Frozen Policy Metrics

- all: `38.00%`
- t50: `26.96%`
- t100 diagnostic: `0.00%`
- hard/failure: `37.71%`
- easy degradation: `0.00%`
- switch rate: `63.08%`

## Hashes

- stage43_m_report_sha256: `4cef03834308250a6e80b98188f2c8c88d367fef91ecb52ec8a9df683c7ca539`
- stage43_m_checkpoint_sha256: `a21a621d9bd5061d728dc80fd43a4d4bf302489b6395cf5562810fabf7c4318f`
- stage43_al_report_sha256: `7381dbd9b13ed9b7f61a47ef019e0dd96aae7ba2b3aa078a10fd862e7a8a88b0`
- stage43_am_report_sha256: `2ec9a668520f49a5784d1b360a4a6b8944fc621304bb0fb2f06d266f8b697d80`

## Boundary

- Global floor is not removed.
- Dataset-local/raw-frame 2.5D only.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| stage43_al_candidate_passed | `True` |
| stage43_am_statistically_confirmed | `True` |
| policy_hash_present | `True` |
| policy_artifact_written | `True` |
| checkpoint_not_tracked_by_git | `True` |
| replay_diff_zero | `True` |
| bootstrap_ci_supports_policy | `True` |
| frozen_metrics_safe | `True` |
| global_floor_not_removed | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
