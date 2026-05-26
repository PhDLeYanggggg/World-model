# Stage42-CG Source Terms Confirmation Validator

- source: `fresh_stage42_cg_source_terms_confirmation_validator`
- generated_at_utc: `2026-05-26T17:16:15.755309+00:00`
- git_commit: `b4ef6a7`
- input_hash: `09d9e5847343e1842c23ce1ae7b3e1e2a919c4f232e800e043e6e826d10dae59`
- gate: `11 / 11`
- verdict: `stage42_cg_source_terms_confirmation_validator_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CG 只验证 source terms confirmation，不下载、不转换、不训练、不评估。
- 空白模板、local path、parseability 都不等于 legal permission。
- conversion_ready 需要 terms accepted、allowed_use、local_path、source_identity 和 CF source-CV blockers 同时通过。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C 未执行，SMC 未启用。

## Summary

- targets_validated: `5`
- terms_accepted_targets: `0`
- conversion_ready_targets: `0`
- conversion_allowed_now_count: `0`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`

## Validation Table

| dataset | terms accepted | conversion ready | CF blockers | confirmation blockers |
| --- | ---: | ---: | --- | --- |
| `ucy_crowd_original` | False | False | manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `eth_biwi_original` | False | False | manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `trajnetplusplus_official` | False | False | manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `opentraj_toolkit` | False | False | no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |
| `aerialmpt_or_other_topdown` | False | False | local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, local_path_confirmation_missing, source_identity_missing |

## Interpretation

- Stage42-CG validates terms-confirmation readiness; it still performs no conversion.
- The current CF-generated template is blank, so every source remains blocked.
- Future conversion must use a filled confirmation file plus a separate no-leakage/source-CV conversion stage.
