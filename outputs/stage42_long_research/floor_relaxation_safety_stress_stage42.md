# Stage42-GT Floor-Relaxation Safety Stress Test

- source: `fresh_stage42_gt_floor_relaxation_safety_stress`
- generated_at_utc: `2026-05-27T14:23:41.903974+00:00`
- git_commit: `16ad229`
- input_hash: `e1bbd7da5c17c848e512997406b79830ef21026e0a0d96d9d01bfa3cfa491075`
- gate: `14 / 14`
- verdict: `stage42_gt_floor_relaxation_safety_stress_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GT 是 Stage42-BY/BZ partial t50 floor-relaxation 的 all-agent safety stress test。
- 本阶段不训练新模型，不下载数据，不转换新数据，不执行 Stage5C，不启用 SMC。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 坐标不能写成 global metric。

## Summary

- selected_variant: `family_baseline_rel_only`
- policy_slices: `['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|100', 'TrajNet|25', 'TrajNet|50', 'UCY|10', 'UCY|100', 'UCY|25', 'UCY|50']`
- target_union_rows: `11538`
- target_union_t50_improvement: `28.97%`
- target_union_hard_failure_improvement: `28.97%`
- target_union_easy_degradation: `-21.41%`
- target_union_near_collision_005_delta: `-0.74%`
- target_union_jagged_rate_delta: `0.00%`
- target_union_safety_pass: `True`
- deployment_decision: `partial_t50_floor_relaxation_safety_supported`
- floor_free_neural_deployable: `False`

## Stress Tests

| slice | rows | groups | t50 gain | hard gain | easy degradation | switch | near@0.05 delta | jagged delta | safety pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_union_t50` | 11538 | 1260 | 28.97% | 28.97% | -21.41% | 89.12% | -0.74% | 0.00% | True |
| `all_test_t50` | 11538 | 1260 | 28.97% | 28.97% | -21.41% | 89.12% | -0.74% | 0.00% | True |
| `TrajNet|50` | 9198 | 597 | 30.21% | 30.21% | -22.95% | 95.26% | -0.95% | 0.00% | True |
| `UCY|50` | 2340 | 663 | 24.53% | 24.53% | -12.64% | 65.00% | 0.13% | 0.00% | True |

## Interpretation

- This stage stress-tests the Stage42-BY/BZ partial t50 floor-relaxation policy at all-agent group level.
- A positive t50 gain alone is not enough; proximity and smoothness must not materially degrade.
- Global floor removal remains forbidden; any supported relaxation is limited to validation-backed t50 slices.
- If `deployment_decision` is diagnostic-only, the BY/BZ t50 relaxation remains paper evidence but not a safety deployment policy.
