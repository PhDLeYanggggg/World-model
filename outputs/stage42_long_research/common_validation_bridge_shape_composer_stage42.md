# Stage42-CO Common Validation Bridge / Shape Composer

- source: `fresh_common_validation_eval_from_cached_verified_checkpoints`
- generated_at_utc: `2026-05-26T23:53:38.575968+00:00`
- git_commit: `c11f73d`
- input_hash: `0310b4beef0ce7237f210b3d6125609e666d1f3fb46bad7b08ba02ee53dd8601`
- gate: `14 / 14`
- verdict: `stage42_co_common_validation_bridge_shape_composer_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CO 使用 common validation-aligned rows 选择 composer policy；test 只评估一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Row Alignment

- validation aligned: `True`
- test aligned: `True`
- validation rows: `True`
- test current_xy max diff: `0.0`

## Validation-Selected Policy

- policy type: `domain_horizon_full_waypoint_composer`
- margin: `0.0`
- easy_max: `0.02`
- candidate count: `16`
- val use_full_rate: `13.59%`
- val vs endpoint all/t50/t100/hard/easy: `2.84%` / `1.82%` / `5.52%` / `3.19%` / `0.13%`

## Test Once

- test use_full_rate: `21.35%`
- test vs endpoint all/t50/t100/hard/easy: `3.02%` / `1.50%` / `6.12%` / `3.28%` / `0.25%`
- test vs strongest floor all/t50/t100/hard/easy: `23.41%` / `14.95%` / `19.91%` / `23.00%` / `-14.29%`

## Interpretation

- This audit resolves the Stage42-CN row-alignment blocker with fresh common validation/test evidence.
- A switch is selected only on validation rows. If validation does not support full-waypoint replacement, the selected policy remains endpoint-linear bridge.
- The result is still dataset-local/raw-frame 2.5D; no metric/seconds, Stage5C, or SMC claim is made.
