# Stage42-IX Source-Level Context Repair Trials

- source: `fresh_run_weighted_floor_residual_context_repair`
- generated_at_utc: `2026-05-28T07:56:27.438433+00:00`
- git_commit: `cc7acce`
- input_hash: `97bee01b1ed62d237510f33759162bd8a887a00d1d46bddf4b759463d81784b1`
- gate: `11 / 12`
- verdict: `stage42_ix_context_repair_completed_context_not_proven`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IX 是 source-level context contribution repair trial：它在 Stage42-AO partial/negative 后修改训练目标并重训/重评。
- 本实验测试 history / goal / neighbor context 在加权 hard/t50/t100 和 floor-residual 目标下是否能提供增量。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-AO found standalone context signal but no incremental context gain after baseline-family rollout features.
- Stage42-IX changes the training target instead of just restating the negative result: it tests hard/t50/t100 weighted ridge training and floor-residual targets.
- Test thresholds are still not tuned on test; validation selects lambda and safe-switch policy.

## Trial Metrics

| trial | features | target | feature_count | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_family_absolute_weighted` | `baseline_family` | `absolute_from_current` | 35 | 0.280381 | 0.317359 | 0.143387 | 0.269583 | -0.311860 | 0.659741 |
| `context_only_absolute_weighted` | `context_only` | `absolute_from_current` | 138 | 0.161408 | 0.118182 | 0.011486 | 0.149392 | -0.188295 | 0.484765 |
| `context_only_floor_residual_weighted` | `context_only` | `floor_residual` | 138 | 0.107583 | 0.102137 | 0.069748 | 0.084198 | -0.257056 | 0.301003 |
| `full_absolute_weighted` | `full` | `absolute_from_current` | 166 | 0.233824 | 0.221797 | 0.143650 | 0.227108 | -0.230189 | 0.568060 |
| `full_floor_residual_weighted` | `full` | `floor_residual` | 166 | 0.233840 | 0.221518 | 0.143687 | 0.227061 | -0.230349 | 0.568060 |
| `goal_neighbor_floor_residual_weighted` | `goal_neighbor_context` | `floor_residual` | 22 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

## Context Delta Versus Baseline-Family Reference

| trial | delta all | delta t50 | delta t100 diag | delta hard | delta easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `context_only_absolute_weighted` | -0.118973 | -0.199177 | -0.131901 | -0.120191 | 0.123565 |
| `context_only_floor_residual_weighted` | -0.172799 | -0.215222 | -0.073639 | -0.185385 | 0.054804 |
| `full_absolute_weighted` | -0.046558 | -0.095562 | 0.000263 | -0.042475 | 0.081670 |
| `full_floor_residual_weighted` | -0.046541 | -0.095841 | 0.000300 | -0.042522 | 0.081511 |
| `goal_neighbor_floor_residual_weighted` | -0.280381 | -0.317359 | -0.143387 | -0.269583 | 0.311860 |

## Summary

- best_trial: `baseline_family_absolute_weighted`
- positive_context_repair_trials: `[]`
- context_claim_verdict: `stage42_ix_context_repair_negative_context_still_not_incremental`
- interpretation: Weighted hard/t50/t100 and floor-residual objectives were tested to give context features a fairer repair path. Positive trials can support a limited context contribution claim; if none are positive, the evidence remains that current source-level ridge dynamics are baseline-family dominated.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-IX did not turn history/goal/neighbor context into an incremental source-level ridge contribution; current evidence remains baseline-family dominated.
- This is a retrained repair attempt, not an inference-mask ablation and not a metric/seconds-level or true-3D claim.
