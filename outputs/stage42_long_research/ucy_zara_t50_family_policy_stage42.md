# Stage42-BS UCY_zara Family-Specific T50 Policy

- source: `fresh_ucy_zara_t50_family_policy`
- generated_at_utc: `2026-05-26T14:59:58.187536+00:00`
- git_commit: `5788582`
- input_hash: `e76c77bcafd86b269a9eed8ded84d2c339b720ae296e49626a4c65b9eabecec6`
- gate: `14 / 14`
- verdict: `stage42_bs_ucy_zara_t50_family_policy_pass_positive`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BS 只针对 Stage42-BR 标出的 UCY_zara calibrated t50 policy/model blocker。
- UCY_zara 有足够同族 source support，因此本步骤不需要新数据许可。
- 训练只用 UCY_zara train source；threshold / alpha / candidate 只用 validation source 选择；holdout source 只最终评估一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- zara_sources_present: `3`
- source_cv_folds: `3`
- rows_total: `51544`
- t50_rows_total: `12750`
- candidate_t50_oracle_headroom_macro_mean: `0.4318861559845056`
- all_improvement_macro_mean: `0.061239639896994214`
- t50_improvement_macro_mean: `0.2471891635815946`
- t50_improvement_min: `0.1509576233445703`
- hard_failure_improvement_macro_mean: `0.06715845661360324`
- easy_degradation_max: `0.012388245928265373`
- policy_selected_fold_count: `3`
- positive_t50_fold_count: `3`
- positive_t50_claim_allowed: `True`
- remaining_blocker: `none`

## Fold Results

| holdout | val | train | rows | t50 rows | oracle headroom | all | t50 | hard/failure | easy degradation | switch | policy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `UCY_zara01` | `UCY_zara02` | `UCY_zara03` | 16103 | 3988 | 0.386743 | 0.036453 | 0.150958 | 0.038941 | -0.013646 | 0.049618 | `ridge_all_lambda_100` |
| `UCY_zara02` | `UCY_zara01` | `UCY_zara03` | 25901 | 6422 | 0.394831 | 0.051920 | 0.205444 | 0.055780 | -0.006847 | 0.066329 | `ridge_t50_lambda_100` |
| `UCY_zara03` | `UCY_zara01` | `UCY_zara02` | 9540 | 2340 | 0.514084 | 0.095345 | 0.385166 | 0.106754 | 0.012388 | 0.152306 | `ridge_t50_lambda_0.1` |

## Interpretation

- BS tests the only calibrated t50 blocker from BR that does not require new data: UCY_zara has same-family support but no safe positive t50 policy yet.
- If BS is positive, UCY_zara can be removed from the policy/model blocker list for calibrated-subset t50.
- If BS falls back or remains non-positive, the blocker is policy/model/feature target quality, not source-support.
- This remains source-specific annotation-step calibrated-subset evidence only; global metric/seconds-level M3W claims remain blocked.

## Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'source_specific_annotation_step_subset_claim_allowed': True, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'm3w_official_metric_seconds_claim_allowed': False, 'positive_t50_claim_allowed': True, 'stage5c_executed': False, 'smc_enabled': False}`
