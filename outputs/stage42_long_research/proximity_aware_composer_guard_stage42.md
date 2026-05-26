# Stage42-CQ Proximity-Aware Composer Guard

- source: `fresh_validation_selected_proximity_guard_from_stage42_co_policy`
- generated_at_utc: `2026-05-26T19:21:12.883942+00:00`
- git_commit: `8a89b14`
- input_hash: `deb19615f8a037e49a5b26e020af9e35b15f9fa2e87803e40d01831ea74aea0a`
- gate: `19 / 19`
- verdict: `stage42_cq_proximity_aware_composer_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CQ 修复 Stage42-CP 暴露的 composer proximity caveat。
- proximity guard 只使用 endpoint/full-waypoint model rollout 的预测几何，不使用 future labels 作为 inference input。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Validation-Selected Guard

- policy type: `proximity_aware_domain_horizon_full_waypoint_composer`
- min_sep: `0.2`
- margin: `0.005`
- candidate count: `15`
- val guarded_off: `1698` rows
- val vs endpoint all/t50/t100/hard/easy: `1.54%` / `0.90%` / `3.06%` / `1.73%` / `0.25%`
- val near_collision@0.05 delta vs endpoint: `-0.06%`

## Test Once

- test guarded_off: `2435` rows
- test use_full_rate: `16.96%`
- test vs endpoint all/t50/t100/hard/easy: `1.77%` / `1.07%` / `3.48%` / `1.93%` / `0.25%`
- test vs strongest floor all/t50/t100/hard/easy: `22.43%` / `14.57%` / `17.66%` / `21.92%` / `-14.30%`

## Bootstrap CI vs Endpoint-Linear

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 1.50% | 1.77% | 2.05% | 55528 |
| `t50` | 0.59% | 1.06% | 1.52% | 13689 |
| `t100` | 2.91% | 3.48% | 4.08% | 9905 |
| `hard_failure` | 1.63% | 1.93% | 2.22% | 41741 |

## Joint Safety

- near_collision@0.05 delta vs endpoint-linear: `-0.06%`
- near_collision@0.05 delta vs strongest floor: `-0.45%`
- p05 min-distance delta vs endpoint-linear: `-0.0001248494575282963`
- jagged-rate delta vs endpoint-linear: `0.00%`

## Interpretation

- Stage42-CQ turns the Stage42-CP proximity caveat into a validation-selected safety guard.
- The guard gives up some Stage42-CO/CP accuracy gain, but keeps all/t50/t100 raw-frame/hard-failure positive with positive all/t50 bootstrap lower bounds.
- Near-collision@0.05 is no longer worse than endpoint-linear or the strongest floor under this guarded policy.
- This remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC.
