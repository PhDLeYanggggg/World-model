# Stage42-AW UCY Validation-Support Repair

- source: `fresh_run`
- generated_at_utc: `2026-05-26T10:45:37.276549+00:00`
- git_commit: `7b69da9`
- input_hash: `491a36f9092f65377ea5c5e3f27fe27cdc80db4d96ce6dd2073b4283829c7ed8`
- gate: `14 / 14`
- verdict: `stage42_aw_ucy_validation_support_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AW 修复的是 UCY 在 proposed source-level split 下没有 validation rows 的 blocker。
- 本修复只从 UCY train sources 中切出 internal validation；test source 完全不参与 policy/threshold 选择。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Internal Validation Repair

- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`
- selected_from: `original_train_sources_only`
- uses_test_rows: `False`
- test_rows_unchanged: `True`
- original UCY val rows: `0`
- repaired UCY val rows: `9540`

## Variant Comparison

| variant | val score | slices | global all | global t50 | global hard | global easy | UCY all | UCY t50 | UCY hard | UCY easy | TrajNet all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `family_baseline_rel_only` | 1.732633 | 12 | 0.356806 | 0.289698 | 0.338904 | -0.370489 | 0.374492 | 0.245320 | 0.355073 | -0.418376 | 0.351423 |
| `floor_plus_family` | 1.657185 | 12 | 0.348043 | 0.262663 | 0.330959 | -0.365936 | 0.367467 | 0.209440 | 0.347740 | -0.404442 | 0.342132 |
| `safe_plus_family` | 1.690307 | 12 | 0.336684 | 0.267403 | 0.327287 | -0.342402 | 0.366607 | 0.214561 | 0.346349 | -0.412264 | 0.327578 |
| `baseline_family_all` | 1.702385 | 12 | 0.347951 | 0.277062 | 0.330670 | -0.377073 | 0.369715 | 0.234200 | 0.350277 | -0.421191 | 0.341327 |

## Validation-Best Candidate

- validation_best_variant: `family_baseline_rel_only`
- policy_slices: `['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|100', 'TrajNet|25', 'TrajNet|50', 'UCY|10', 'UCY|100', 'UCY|25', 'UCY|50']`
- summary: `{'source': 'fresh_run', 'ucy_blocker_before': 'no_validation_rows_for_domain_policy_selection_floor_only', 'ucy_val_rows_after': 9540, 'ucy_positive_transfer_after': True, 'trajnet_preserved_after': True, 'paper_claim': 'UCY floor-only blocker is repaired by carving validation support from UCY train sources only; the selected policy is validation-best, test-once, dataset-local raw-frame evidence.'}`

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'internal_val_from_train_only': True, 'test_sources_unchanged': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`
