# Stage42-GL Source Conversion Contract

- source: `fresh_stage42_gl_source_conversion_contract`
- generated_at_utc: `2026-05-27T13:07:09.884651+00:00`
- git_commit: `a83420c`
- input_hash: `121b05f66e319071f0806370ea494a37aefc0a397b83a0d3c737b7476c9c8067`
- gate: `16 / 16`
- verdict: `stage42_gl_source_conversion_contract_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GL 是 source/legal/calibration 到 guarded conversion 的合约检查，不下载、不转换、不训练、不评估。
- post-confirmation calibrated subset candidates 只有在用户确认 official terms、allowed use、local path、source identity 后才可能进入 guarded conversion。
- blank intake、prefill suggestion、parseability、technical dry-run 都不等于 permission、conversion 或 evaluation。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_gl_source_conversion_contract`
- intake_datasets: `5`
- manifest_ready_targets: `0`
- manifest_blocked_targets: `5`
- contract_ready_now: `0`
- guarded_launcher_queue_count: `0`
- post_confirmation_calibrated_candidate_datasets: `2`
- post_confirmation_calibrated_source_rows: `5`
- calibrated_t50_windows_after_terms: `10060`
- calibrated_t100_windows_after_terms: `5696`
- download_executed: `False`
- conversion_executed: `False`
- training_executed: `False`
- evaluation_executed: `False`
- next_required_action: `user fills official terms/path/source identity, then reruns validator and guarded launcher`

## Contract Table

| rank | dataset | domain | status | missing fields | local path found | after-terms calibrated sources | t50/t100 after terms |
| ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | `ucy_crowd_original` | `UCY` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False | 3 | 9554 / 5605 |
| 2 | `eth_biwi_original` | `ETH_UCY` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False | 2 | 506 / 91 |
| 3 | `aerialmpt_or_other_topdown` | `other_topdown` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False | 0 | 0 / 0 |
| 4 | `opentraj_toolkit` | `OpenTraj` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False | 0 | 0 / 0 |
| 5 | `trajnetplusplus_official` | `TrajNet` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False | 0 | 0 / 0 |

## Interpretation

- `post_confirmation` rows are only opportunity rows. They are not permission, not converted data, and not evaluated evidence.
- `contract_ready_now = 0` means no future converter may run yet.
- The next meaningful user action is to fill official terms/path/source identity in the intake template and rerun the validator.
- Even after conversion, any metric/time wording must be limited to the restricted source-specific subset that passes the metric/time claim guard.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `intake_loaded` | True |
| `manifest_loaded` | True |
| `gh_loaded` | True |
| `ej_loaded` | True |
| `required_fields_enforced` | True |
| `blank_or_incomplete_rows_not_ready` | True |
| `contract_ready_matches_launcher_queue` | True |
| `post_confirmation_candidates_recorded` | True |
| `calibrated_opportunity_recorded` | True |
| `no_download_conversion_training_eval` | True |
| `post_confirmation_candidates_not_claimed_as_data` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
