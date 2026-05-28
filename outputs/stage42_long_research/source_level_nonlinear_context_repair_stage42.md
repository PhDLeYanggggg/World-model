# Stage42-IY Source-Level Nonlinear Context Repair

- source: `fresh_run_sampled_extra_trees_context_capacity_repair`
- generated_at_utc: `2026-05-28T08:46:56.602469+00:00`
- git_commit: `7eeaf1a`
- input_hash: `d68e9b96215fac749f9f5d7924aed6773d630b0888d5b841350577df49f94d85`
- gate: `12 / 13`
- verdict: `stage42_iy_nonlinear_context_repair_completed_context_not_proven`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IY 是 Stage42-IX negative result 后的非线性 context repair trial，用 ExtraTrees 多输出 residual 模型测试容量不足假设。
- 训练使用 train split 的 deterministic capped subset；validation 选 safe policy；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-IX showed that weighted ridge and floor-residual targets did not make history/goal/neighbor context incremental.
- Stage42-IY tests whether a nonlinear ExtraTrees residual model can recover context value under the same source-level no-leakage protocol.
- Training uses a deterministic train-only capped subset and records that cap explicitly; validation selects the safe policy; test is evaluated once.

## Trial Metrics

| trial | feature_set | train_used | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tree_baseline_family_residual` | `baseline_family` | 120000 | 0.221602 | 0.246937 | 0.187483 | 0.232718 | -0.125700 | 0.589279 |
| `tree_context_only_residual` | `context_only` | 120000 | 0.207041 | 0.192906 | 0.117467 | 0.200006 | -0.223051 | 0.587593 |
| `tree_full_residual` | `full` | 120000 | 0.229274 | 0.196168 | 0.186525 | 0.230031 | -0.199691 | 0.564204 |
| `tree_goal_neighbor_residual` | `goal_neighbor_context` | 120000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

## Delta Versus Nonlinear Baseline-Family Tree

| trial | delta all | delta t50 | delta t100 diag | delta hard | delta easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tree_context_only_residual` | -0.014562 | -0.054031 | -0.070015 | -0.032712 | -0.097351 |
| `tree_full_residual` | 0.007671 | -0.050768 | -0.000957 | -0.002686 | -0.073991 |
| `tree_goal_neighbor_residual` | -0.221602 | -0.246937 | -0.187483 | -0.232718 | 0.125700 |

## Summary

- best_trial: `tree_baseline_family_residual`
- positive_nonlinear_context_trials: `[]`
- capacity_hypothesis_verdict: `stage42_iy_nonlinear_context_capacity_not_sufficient`
- interpretation: If nonlinear context trials still fail to beat the nonlinear baseline-family tree, the context gap is not explained by simple linear capacity alone. If a context trial wins, that supports a limited nonlinear context contribution claim under source-level split.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-IY did not recover incremental context value with ExtraTrees capacity; current source-level evidence remains baseline-family dominated.
- This is a sampled train-only nonlinear repair test, not a full foundation/full-data claim and not metric/seconds-level evidence.
