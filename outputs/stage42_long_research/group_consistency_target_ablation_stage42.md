# Stage42-ET Group-Consistency Target Ablation

- source: `fresh_stage42_group_consistency_target_ablation`
- generated_at_utc: `2026-05-27T04:02:49.173566+00:00`
- git_commit: `b691057`
- input_hash: `a23c3b418b78e49f5ebddafe495e6ba51ff541afe0abd7bd0a6fdde4bb789418`
- gate: `16 / 16`
- verdict: `stage42_et_group_consistency_target_ablation_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-ET 对 Stage42-ES 选出的 explicit group-consistency target 做 group-schema ablation。
- 本阶段 fresh-reruns source-level full-waypoint repair under alternative group keys；不下载、不转换、不执行 Stage5C、不启用 SMC。
- future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- validation 选择每个 group schema 内的 repair candidate；test 只评一次。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Selection

- selected_by_validation_score: `source_frame_horizon`
- selected_target_for_next_stage: `source_frame_horizon`
- decision: `keep_source_frame_horizon_group_consistency_target`

## Group Schema Comparison

| schema | val score | all | t50 | t100 raw | hard | easy | near@0.05 | unsafe rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_frame_horizon` | 1.910356 | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% | 1.38% | 955 |
| `source_framebucket10_horizon` | 1.910356 | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% | 1.38% | 955 |
| `domain_frame_horizon` | 1.910356 | 24.72% | 22.36% | 14.35% | 23.89% | -25.61% | 1.48% | 1051 |
| `source_frame_no_horizon` | 1.906307 | 24.59% | 22.05% | 14.36% | 23.76% | -25.66% | 4.06% | 874 |
| `agent_isolated_no_interaction` | 1.906243 | 24.58% | 22.02% | 14.37% | 23.75% | -25.66% | 0.00% | 0 |
| `shuffled_source_frame_horizon` | 1.906184 | 24.58% | 22.02% | 14.37% | 23.75% | -25.66% | 1.15% | 59 |

## Contribution Delta

| delta | value |
| --- | ---: |
| `all_increment` | 0.14% |
| `t50_increment` | 0.35% |
| `t100_raw_frame_diagnostic_increment` | -0.02% |
| `hard_failure_increment` | 0.14% |
| `easy_degradation_increment` | 0.03% |
| `near005_reduction_vs_correct_base` | 0.55% |
| `isolated_near005_not_comparable` | 0.00% |
| `p05_min_distance_gain_vs_isolated` | 7.77% |

## Interpretation

- `agent_isolated_no_interaction` is the no-interaction accuracy control: each row is its own group, so group repair cannot use multi-agent proximity; its near@0.05 value is not a valid pairwise collision baseline.
- A positive source/frame/horizon increment over the isolated control supports the group-consistency target as an interaction/occupancy constraint rather than a generic scalar loss artifact.
- This remains protected source-level raw-frame 2.5D evidence, not a metric/seconds-level or floor-free neural claim.

## Gate

| gate | pass |
| --- | ---: |
| `group_schema_variants_evaluated` | True |
| `source_frame_horizon_present` | True |
| `agent_isolated_control_present` | True |
| `source_frame_horizon_all_positive` | True |
| `source_frame_horizon_t50_positive` | True |
| `source_frame_horizon_hard_positive` | True |
| `source_frame_horizon_easy_safe` | True |
| `source_frame_horizon_hard_beats_isolated` | True |
| `source_frame_horizon_near005_repaired_vs_own_base` | True |
| `source_frame_horizon_bootstrap_all_positive` | True |
| `source_frame_horizon_bootstrap_hard_positive` | True |
| `isolated_control_matches_no_interaction_target` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
