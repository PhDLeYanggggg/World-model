# Stage42-GM Guarded Conversion Harness

- source: `fresh_stage42_gm_guarded_conversion_harness`
- generated_at_utc: `2026-05-27T13:13:27.697719+00:00`
- git_commit: `748ec29`
- input_hash: `6a150731c36023e325aa2f6e0c8e64411d71224082ba55ef269d154881d6a1eb`
- gate: `14 / 14`
- verdict: `stage42_gm_guarded_conversion_harness_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GM 是 guarded converter execution harness；默认 dry-run，并且当前 contract_ready_now=0 时必须拒绝转换。
- 本阶段不下载、不转换、不训练、不评估；没有生成新 feature store。
- prefill、terms hints、parseability、technical dry-run、contract opportunity 都不等于 legal converted data。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_gm_guarded_conversion_harness`
- execute_requested: `False`
- contract_source: `fresh_stage42_gl_source_conversion_contract`
- contract_verdict: `stage42_gl_source_conversion_contract_pass`
- contract_ready_now: `0`
- blocked_contract_rows: `5`
- execution_plan_count: `0`
- conversion_refused_reason: `contract_ready_now_is_zero`
- download_executed: `False`
- conversion_executed: `False`
- feature_store_built: `False`
- no_leakage_audit_executed: `False`
- source_cv_executed: `False`
- training_executed: `False`
- evaluation_executed: `False`
- next_required_action: `fill terms/path/source identity and rerun validator, GL contract, then GM harness`

## Execution Plan

- No execution plan because `contract_ready_now = 0`. Conversion is correctly refused.

## Blocked Contract Rows

| dataset | domain | status | missing fields |
| --- | --- | --- | --- |
| `ucy_crowd_original` | `UCY` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| `eth_biwi_original` | `ETH_UCY` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| `aerialmpt_or_other_topdown` | `other_topdown` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| `opentraj_toolkit` | `OpenTraj` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |
| `trajnetplusplus_official` | `TrajNet` | `blocked_until_user_terms_path_source_confirmation` | terms_accepted_by_user, terms_acceptance_date, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user |

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `contract_loaded` | True |
| `contract_gate_passed` | True |
| `dry_run_default` | True |
| `no_ready_refuses_conversion` | True |
| `blocked_rows_preserved` | True |
| `no_download_conversion_feature_store` | True |
| `no_no_leakage_or_source_cv_claim` | True |
| `no_training_eval_claim` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `no_converted_data_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
