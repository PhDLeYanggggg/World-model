# Stage42-DL Group-Consistency Runtime Policy API

- source: `fresh_runtime_api_from_frozen_group_consistency_policy_artifact`
- generated_at_utc: `2026-05-26T22:52:39.008600+00:00`
- git_commit: `9f5ca39`
- input_hash: `26edd12e22c749429c1378f90b71c934670931a874298333f4ff2c3d63d4cf95`
- policy_hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `30 / 30`
- verdict: `stage42_dl_group_consistency_runtime_policy_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DL 把 Stage42-DJ/DK frozen group-consistency full-waypoint policy 变成可调用 runtime policy API。
- runtime policy 只使用 predicted full-waypoint rollouts、train-horizon causal floor rollout、source/frame/horizon group key、agent id、current xy、normalizer。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- DL 不重新训练、不重新选择阈值、不使用 test metrics 调参。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Runtime Policy

- mode: `repel_unsafe`
- min_sep: `0.08`
- margin: `0.0`
- strength: `0.5`
- runtime inputs: `['base_xy_predicted_full_waypoint_candidate', 'floor_xy_train_horizon_causal_rollout', 'pred_xy_model_rollout_diagnostic', 'base_switch_from_validation_selected_policy', 'source_frame_horizon_group_key', 'normalizer', 'agent_id', 'current_xy']`

## Real Batch Replay

- rows: `47458`
- selected_xy_max_abs_diff: `0.0`
- switch_exact_match: `True`
- selected_ade_max_abs_diff: `0.0`
- selected_fde_max_abs_diff: `0.0`

## Replayed Metrics Vs Train-Horizon Causal Floor

- all: `24.72%`
- t50: `22.36%`
- t100 raw-frame diagnostic: `14.35%`
- hard/failure: `23.89%`
- easy degradation: `-25.63%`

## Replayed Group Safety

- base near@0.05: `1.94%`
- final near@0.05: `1.38%`
- floor near@0.05: `2.24%`
- base p05 min distance: `0.07437689768396878`
- final p05 min distance: `0.07770240407545181`

## Interpretation

- Stage42-DL turns the frozen group-consistency full-waypoint repair into a callable runtime policy API.
- The real-batch replay verifies that runtime application exactly matches the original Stage42-DI selected repair on reconstructed test rows.
- It does not execute Stage5C, does not enable SMC, and does not make metric/seconds-level claims.
