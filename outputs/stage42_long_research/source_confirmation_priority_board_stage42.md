# Stage42-GN Source Confirmation Priority Board

- source: `fresh_stage42_gn_source_confirmation_priority_board`
- generated_at_utc: `2026-05-27T13:22:32.549827+00:00`
- git_commit: `431e72f`
- input_hash: `341474d3bfd0987ea6abd7b81c699303b88bbc19d85017bc4ef4efb2f0551526`
- gate: `14 / 14`
- verdict: `stage42_gn_source_confirmation_priority_board_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GN 只排序 source/legal/calibration unblock 优先级；不下载、不转换、不训练、不评估。
- contract_ready_now=0 时，任何 post-confirmation opportunity 都不能写成 converted dataset 或 evaluation result。
- 用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。
- dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_gn_source_confirmation_priority_board`
- targets_ranked: `5`
- ready_now: `0`
- blocked_now: `5`
- top_priority_dataset: `ucy_crowd_original`
- top_priority_domain: `UCY`
- top_priority_value_class: `calibrated_t50_t100_unlock`
- total_t50_after_terms: `10060`
- total_t100_after_terms: `5696`
- calibrated_t50_after_terms: `10060`
- calibrated_t100_after_terms: `5696`
- contract_ready_now_from_gm: `0`
- gm_conversion_executed: `False`
- download_executed: `False`
- conversion_executed: `False`
- training_executed: `False`
- evaluation_executed: `False`
- next_best_user_action: `confirm UCY crowd official terms/local path/source identity first, then ETH/BIWI; TrajNet remains useful but h100-limited unless longer official raw sources are provided`

## Ranked Source Confirmation Queue

| rank | dataset | domain | value class | score | t50/t100 after terms | calibrated t50/t100 | raw path candidates | missing user fields |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | --- |
| 1 | `ucy_crowd_original` | `UCY` | `calibrated_t50_t100_unlock` | 3652.680 | 9554 / 5605 | 9554 / 5605 | 1 | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| 2 | `eth_biwi_original` | `ETH_UCY` | `calibrated_t50_t100_unlock` | 1676.800 | 506 / 91 | 506 / 91 | 2 | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| 3 | `opentraj_toolkit` | `OpenTraj` | `low_or_diagnostic_unlock` | 300.000 | 0 / 0 | 0 / 0 | 1 | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| 4 | `trajnetplusplus_official` | `TrajNet` | `low_or_diagnostic_unlock` | 300.000 | 0 / 0 | 0 / 0 | 2 | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| 5 | `aerialmpt_or_other_topdown` | `other_topdown` | `low_or_diagnostic_unlock` | 25.000 | 0 / 0 | 0 / 0 | 1 | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |

## Interpretation

- `UCY` is first because it has the largest post-confirmation t50/t100 and calibrated subset opportunity already visible locally.
- `ETH_UCY` is second because it has source-specific calibration value but much smaller unlocked row count.
- `TrajNet` remains important for diversity, but current local material is short-snippet / h100-limited and cannot repair the long-horizon blocker by itself.
- `AerialMPT / other_topdown` and `OpenTraj toolkit` are lower priority until official terms/source identity and parseable trajectory scope are confirmed.
- This board is an unblock queue only: no data was converted, no evaluation ran, and no metric/seconds claim is allowed.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `contract_loaded` | True |
| `harness_loaded` | True |
| `calibrated_plan_loaded` | True |
| `calibration_manifest_loaded` | True |
| `all_contract_rows_ranked` | True |
| `blocked_rows_not_marked_ready` | True |
| `opportunity_windows_preserved` | True |
| `top_priority_actionable_after_user_confirmation` | True |
| `no_download_conversion_training_eval` | True |
| `user_action_written` | True |
| `no_converted_or_metric_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
