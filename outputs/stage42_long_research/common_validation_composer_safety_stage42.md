# Stage42-CP Common Validation Composer Safety / Bootstrap

- source: `fresh_joint_safety_bootstrap_from_stage42_co_policy`
- generated_at_utc: `2026-05-26T19:03:45.683430+00:00`
- git_commit: `9bf3a4b`
- input_hash: `300dd877fdd9f614a29a02f56713bef8fae87c083f46373f131191ef0bd951b2`
- gate: `14 / 14`
- verdict: `stage42_cp_common_validation_composer_safety_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CP 是 Stage42-CO composer 的 bootstrap + all-agent joint safety audit。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Test Metrics

- vs endpoint-linear ADE all/t50/t100/hard/easy: `3.02%` / `1.50%` / `6.12%` / `3.28%` / `0.25%`
- vs strongest floor ADE all/t50/t100/hard/easy: `23.41%` / `14.95%` / `19.91%` / `23.00%` / `-14.29%`

## Bootstrap CI vs Endpoint-Linear

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 2.64% | 3.01% | 3.37% | 55528 |
| `t50` | 0.90% | 1.51% | 2.09% | 13689 |
| `t100` | 5.39% | 6.12% | 6.94% | 9905 |
| `hard_failure` | 2.90% | 3.28% | 3.68% | 41741 |

## Joint Safety

- near_collision@0.05 delta vs endpoint-linear: `0.34%`
- near_collision@0.05 delta vs strongest floor: `-0.05%`
- p05 min-distance delta vs endpoint-linear: `-0.0015142309056838422`
- jagged-rate delta vs endpoint-linear: `0.00%`

## Interpretation

- Stage42-CP adds statistical and joint-safety evidence to Stage42-CO.
- The composer improves endpoint-linear bridge with positive bootstrap lower bounds on all, t50, t100 raw-frame, and hard/failure ADE.
- Proximity is materially safe: near-collision@0.05 is slightly higher than endpoint-linear but remains lower than the strongest floor, and smoothness does not worsen.
- This remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC.
