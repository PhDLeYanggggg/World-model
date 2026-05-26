# Stage42-CF Source Conversion Legal Gate

- source: `fresh_stage42_cf_source_conversion_legal_gate`
- generated_at_utc: `2026-05-26T17:10:31.595849+00:00`
- git_commit: `944ec30`
- input_hash: `5da3baeb5063f1959cdc573ab4b0f37842e0d99ff803cd4308f056e9c9dbd7a2`
- gate: `13 / 13`
- verdict: `stage42_cf_source_conversion_legal_gate_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CF 是 source conversion legal gate，不下载、不转换、不训练、不评估。
- local path found / schema_possible 不等于 legal permission。
- terms confirmed 必须是用户或官方条款确认后的显式记录，不能由脚本自动伪造。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C 未执行，SMC 未启用。

## Summary

- targets_checked: `5`
- local_paths_present: `4`
- schema_possible_targets: `4`
- targets_with_t50_files: `3`
- targets_with_t100_files: `3`
- source_cv_ready_now: `0`
- conversion_allowed_now_count: `0`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`

## Target Decisions

| target | local path | schema | t50 files | independent t50 | legal blocked | conversion allowed now | blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ucy_crowd_original` | True | True | 6 | 0 | True | False | manual_terms_or_application_required, no_independent_t50_candidate |
| `eth_biwi_original` | True | True | 3 | 0 | True | False | manual_terms_or_application_required, no_independent_t50_candidate |
| `trajnetplusplus_official` | True | True | 0 | 0 | True | False | manual_terms_or_application_required, no_independent_t50_candidate |
| `opentraj_toolkit` | True | True | 270 | 0 | False | False | no_independent_t50_candidate |
| `aerialmpt_or_other_topdown` | False | False | 0 | 0 | True | False | local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate |

## Interpretation

- Stage42-CF intentionally allows zero conversions right now.
- This is a guardrail: future conversion requires explicit terms confirmation plus independent source identity.
- The generated confirmation template is not permission; it is a checklist the user must fill after official terms/path verification.
- No metric/seconds-level, true-3D, Stage5C, or SMC claim is introduced.
