# Stage42-HM Restricted Metric/Time Source-Level Terms Intake v2

- source: `fresh_stage42_hm_restricted_metric_time_terms_intake_v2`
- generated_at_utc: `2026-05-27T18:11:24.658424+00:00`
- git_commit: `e0c706e`
- input_hash: `28b38416c0d3149245113db342fca1539a4e778ebe3ccf05d9df9b9582aa96e2`
- gate: `15 / 15`
- verdict: `stage42_hm_restricted_metric_time_terms_intake_v2_pass_blocked_until_user_confirmation`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HM 是 restricted metric/time source-level terms intake v2，不下载、不转换、不训练、不评估。
- 本阶段把 Stage42-HJ/HK 的 source-level UCY/ETH/ETH-Person 候选转成用户可填写的 terms/source identity template。
- 空白 template、local file present、parseability、technical dry-run 都不等于 legal conversion readiness。
- future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- restricted seconds/metric wording 仍需 user terms confirmation、guarded conversion、no-leakage、source-CV、final test。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source_level_candidates: `11`
- source_cv_usable_after_terms_candidates: `9`
- conversion_ready_candidates_now: `0`
- blocked_candidates_now: `11`
- after_terms_domains_with_source_cv_candidate_count: `{'UCY': 3, 'ETH_UCY': 6}`
- ready_now_domains_with_source_cv_candidate_count: `{}`
- after_terms_total_t50/t100_windows: `14457` / `7129`
- ready_now_t50/t100_windows: `0` / `0`
- template_path: `outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_template_stage42.json`
- manifest_path: `outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_manifest_stage42.json`

## Candidate Table

| rank | source | domain | target | t50 after terms | t100 after terms | source-CV usable after terms | ready now | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `UCY_students03` | `UCY` | `ucy_crowd_original` | 6491 | 3413 | True | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 2 | `UCY_zara02` | `UCY` | `ucy_crowd_original` | 2823 | 2095 | True | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 3 | `ETH-Person_seq0_assc_gt` | `ETH_UCY` | `eth_person_local_candidates` | 1040 | 465 | True | False | official_terms_url_requires_user_verified_official_source, official_terms_url_missing, terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 4 | `ETH-Person_sunnyday_assc_gt` | `ETH_UCY` | `eth_person_local_candidates` | 626 | 406 | True | False | official_terms_url_requires_user_verified_official_source, official_terms_url_missing, terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 5 | `ETH-Person_bahnhof_assc_gt` | `ETH_UCY` | `eth_person_local_candidates` | 1897 | 348 | True | False | official_terms_url_requires_user_verified_official_source, official_terms_url_missing, terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 6 | `ETH-Person_jelmoli_assc_gt` | `ETH_UCY` | `eth_person_local_candidates` | 561 | 126 | True | False | official_terms_url_requires_user_verified_official_source, official_terms_url_missing, terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 7 | `UCY_zara01` | `UCY` | `ucy_crowd_original` | 240 | 97 | True | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 8 | `ETH_seq_eth` | `ETH_UCY` | `eth_biwi_original` | 291 | 91 | True | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 9 | `ETH_seq_eth` | `ETH_UCY` | `eth_biwi_original` | 273 | 88 | True | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing |
| 10 | `ETH_seq_hotel` | `ETH_UCY` | `eth_biwi_original` | 215 | 0 | False | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing, source_cv_not_usable_even_after_terms |
| 11 | `UCY_zara03` | `UCY` | `ucy_crowd_original` | 0 | 0 | False | False | terms_not_accepted_by_user, terms_acceptance_date_missing, accepted_terms_version_or_access_date_missing, allowed_use_missing_or_unknown, redistribution_allowed_unknown, derived_data_allowed_unknown, local_path_missing, source_identity_missing, confirmed_by_user_missing, source_cv_not_usable_even_after_terms |

## Interpretation

- UCY and ETH_UCY both have source-level candidates that could support restricted metric/time source-CV after user terms confirmation.
- Current ready-now count is zero because the template is intentionally blank and ETH-Person official/source terms still require user verification.
- This is a source-level intake and validator artifact, not a conversion/evaluation result and not a metric/seconds claim.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `hj_input_passed` | True |
| `hk_input_passed` | True |
| `hl_input_passed` | True |
| `source_level_candidates_present` | True |
| `ucy_and_eth_ucy_after_terms_present` | True |
| `after_terms_t50_t100_support_present` | True |
| `ready_now_zero_until_user_confirms` | True |
| `template_written` | True |
| `manifest_built` | True |
| `blocked_candidates_preserved` | True |
| `no_download_conversion_training_eval` | True |
| `no_metric_seconds_claim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
