# Stage42-EH Source Terms Confirmation Intake Package

- source: `fresh_source_terms_confirmation_intake_from_stage42_ef`
- generated_at_utc: `2026-05-27T02:14:13.640488+00:00`
- git_commit: `adb6f07`
- input_hash: `35ef1b12d0b79925bd88cff0cc7c00d46fe39e4a6cac460d34bdd719dd77c8c5`
- gate: `14 / 14`
- verdict: `stage42_eh_source_terms_confirmation_intake_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EH turns the Stage42-EF source-terms blocker into a fillable confirmation/intake package.
- 本阶段不下载、不转换、不训练、不评估。
- 只有用户确认 official terms、allowed use、local path 和 source identity 后，未来阶段才允许转换。
- local path、parseability、technical dry-run 都不等于 legal conversion readiness。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets: `5`
- high_priority_after_terms_targets: `2`
- conversion_ready_now: `0`
- converted/evaluated now: `0` / `0`
- estimated_t50/t100_windows_after_terms: `10060` / `5696`
- top_unblock_targets: `['ucy_crowd_original', 'eth_biwi_original', 'aerialmpt_or_other_topdown']`
- intake_template: `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`
- schema: `outputs/stage42_long_research/source_terms_confirmation_schema_stage42.json`
- validator_command: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`

## Intake Table

| rank | dataset | domain | t50 after terms | t100 after terms | source-CV after terms | user fields required | agent may fill |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| 1 | `ucy_crowd_original` | `UCY` | 9554 | 5605 | True | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False |
| 2 | `eth_biwi_original` | `ETH_UCY` | 506 | 91 | True | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False |
| 3 | `aerialmpt_or_other_topdown` | `other_topdown` | 0 | 0 | False | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False |
| 4 | `opentraj_toolkit` | `OpenTraj` | 0 | 0 | False | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False |
| 5 | `trajnetplusplus_official` | `TrajNet` | 0 | 0 | False | terms_accepted_by_user, terms_acceptance_date, official_terms_url, accepted_terms_version_or_access_date, allowed_use, redistribution_allowed, derived_data_allowed, local_path, source_identity, confirmed_by_user | False |

## How To Use

1. Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
2. For each dataset you want to unblock, manually verify the official terms/source page and fill every required field.
3. Do not let the agent infer acceptance, allowed use, or source identity.
4. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.
5. Only if the validator reports conversion-ready targets should a future guarded conversion/no-leakage/source-CV stage run.

## Gate

| gate | pass |
| --- | ---: |
| `ef_input_passed` | True |
| `schema_written` | True |
| `intake_template_written` | True |
| `required_fields_present` | True |
| `ucy_priority_preserved` | True |
| `eth_present` | True |
| `all_targets_require_user_confirmation` | True |
| `no_conversion_or_eval_claim` | True |
| `legal_blocker_preserved` | True |
| `validator_command_recorded` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
