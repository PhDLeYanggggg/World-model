# Stage42-CV Batch Runtime Policy Replay

- source: `fresh_batch_runtime_replay_from_frozen_policy_artifact`
- generated_at_utc: `2026-05-26T23:09:18.992450+00:00`
- git_commit: `47b5df5`
- policy_hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- gate: `25 / 25`
- verdict: `stage42_cv_batch_runtime_replay_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CV 在真实 common validation/test rows 上重放 Stage42-CU runtime policy。
- guard 的第二个 proximity 输入是 validation-selected base composer candidate rollout 的 group min-distance，不是 future label。
- runtime batch replay 不重新选择阈值，不使用 test endpoint goals，不执行 Stage5C，不启用 SMC。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Replay Summary

- validation rows: `53256`
- test rows: `55528`
- validation decision exact replay: `True`
- test decision exact replay: `True`
- test selected_xy max abs diff vs CQ: `0.0`
- test selected ADE max abs diff vs CQ: `0.0`
- test selected FDE max abs diff vs CQ: `0.0`

## Test Metrics Vs Endpoint-Linear ADE

- all: `1.77%`
- t50: `1.07%`
- t100 raw-frame diagnostic: `3.48%`
- hard/failure: `1.93%`
- easy degradation: `0.25%`
- switch rate: `16.96%`

## Runtime Reasons On Test

- `base_choice_endpoint_linear`: `43673`
- `base_choice_full_waypoint_geometry_nonfinite_replay_no_guard`: `130`
- `base_choice_full_waypoint_guard_clear`: `9290`
- `proximity_guard_fallback_to_endpoint_linear`: `2435`

## Joint Safety Vs Endpoint-Linear

- near_collision@0.02 delta: `-0.00%`
- near_collision@0.05 delta: `-0.06%`
- p05 min group distance delta: `-0.01%`
- jagged-rate delta: `0.00%`

## Interpretation

- Stage42-CV proves the callable runtime policy exactly replays the original CQ guard decisions on real validation/test rows.
- This is stronger than smoke testing: it exercises the policy on the same common rows used by the bridge/shape composer evidence.
- It does not reselect thresholds, does not use future labels as inputs, and does not add new model scores.
- Claims remain protected dataset-local/raw-frame 2.5D only.
