# Stage42-BT ETH_seq T50 Support Dry-Run

- source: `fresh_eth_seq_t50_support_dry_run_terms_unverified`
- generated_at_utc: `2026-05-26T15:07:20.693255+00:00`
- git_commit: `0408506`
- input_hash: `717b7343ec0cdd08c17e6ac3fccc63d741ae236c0f720ed9f795265cad64938a`
- gate: `13 / 13`
- verdict: `stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BT 只测试 ETH-Person XML 是否技术上能修复 Stage42-BR 的 ETH_seq calibrated t50 source-support blocker。
- ETH-Person XML terms/license 仍未确认；本结果只能是 technical_dry_run_terms_unverified。
- policy threshold / baseline choice 只从 train/validation source 选择，holdout source 只评估一次。
- future endpoints 只作为 validation/test error labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- candidate_sources: `5`
- eth_person_xml_sources: `4`
- source_cv_folds: `5`
- h50_windows_total: `4397`
- technical_h50_mean_improvement_vs_fallback: `0.4112171834613682`
- technical_h50_min_improvement_vs_fallback: `0.0`
- technical_h50_max_easy_degradation: `0.12464016124653272`
- safe_positive_h50_fold_count: `3`
- eth_seq_holdout_rows: `273`
- eth_seq_h50_improvement_vs_fallback: `0.0`
- eth_seq_easy_degradation: `0.0`
- eth_seq_selected_policy: `constant_position_speed_causal_high`
- eth_seq_t50_support_repaired: `False`
- remaining_blocker: `ETH_seq remains unsupported for calibrated t50 under validation-only safe policy; ETH-Person XML h50 technical positives do not safely transfer to ETH_seq_eth holdout.`

## Fold Results

| holdout | validation | rows | improvement | easy degradation | switch | safe positive | selected policy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH-Person_seq0_assc_gt` | `ETH-Person_sunnyday_assc_gt` | 1040 | 0.433860 | -0.182592 | 0.965385 | True | `constant_position_speed_causal_high` |
| `ETH-Person_sunnyday_assc_gt` | `ETH-Person_seq0_assc_gt` | 626 | 0.788611 | -0.662703 | 0.974441 | True | `constant_position_speed_causal_high` |
| `ETH-Person_bahnhof_assc_gt` | `ETH-Person_seq0_assc_gt` | 1897 | 0.475994 | -0.195992 | 0.920928 | True | `constant_position_speed_causal_high` |
| `ETH-Person_jelmoli_assc_gt` | `ETH-Person_seq0_assc_gt` | 561 | 0.357622 | 0.124640 | 0.976827 | False | `constant_position_speed_causal_high` |
| `ETH_seq_eth` | `ETH-Person_seq0_assc_gt` | 273 | 0.000000 | 0.000000 | 0.000000 | False | `constant_position_speed_causal_high` |

## Interpretation

- ETH-Person XML provides technical h50 signal on several ETH-Person holdouts, but this does not safely repair the actual `ETH_seq_eth` calibrated t50 holdout.
- For `ETH_seq_eth`, validation-only safety selection falls back to constant velocity, so improvement is 0 rather than positive transfer.
- This confirms the Stage42-BR blocker: ETH_seq still needs same-family/source-compatible support, official terms confirmation, or a stronger source-compatible model.
- Because ETH-Person terms are unverified, none of this is official converted/evaluated data or a deployable metric/seconds-level claim.

## Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_labels_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'selection_uses_holdout': False}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'eth_person_terms_confirmed': False, 'official_converted_dataset_claim_allowed': False, 'source_specific_annotation_step_subset_claim_allowed': True, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'positive_eth_seq_t50_claim_allowed': False, 'stage5c_executed': False, 'smc_enabled': False}`
