# Stage42-IU Source-Level UCY Full-Waypoint Specialist Integration

- source: `fresh_composition_from_current_stage42_it_and_cached_verified_stage42_v`
- generated_at_utc: `2026-05-28T06:26:53.977147+00:00`
- git_commit: `6bb9df2`
- input_hash: `5fcc912ce5d934649d7c976865ee81df74a900376b4311cb3f9adb5267c9df44`
- policy_hash: `e8f205c3cfecc21c27f62788805c0ae9cbe746a41fa51cbb476c5ef6486e33c7`
- gate: `17 / 17`
- verdict: `stage42_iu_source_level_ucy_full_waypoint_integration_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IU 是 source-level full-waypoint UCY specialist integration，不是 metric 或 seconds-level 结果。
- Stage42-IU 使用 Stage42-IT current source-level TrajNet slice，并用 Stage42-V UCY slice 替换 Stage42-IT 的 UCY fallback-only slice。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 当前仍未构建单一 merged row-cache artifact；本阶段是 source-level policy-package composition evidence。
- Stage5C latent generative 未执行。
- SMC 未启用。

## What This Adds Beyond Stage42-IT

- Stage42-IT was fresh source-level full-waypoint evidence, but UCY remained fallback-only in that proposed source-level test.
- Stage42-IU keeps the fresh Stage42-IT TrajNet slice and replaces only the UCY fallback-only slice with the cached-verified Stage42-V UCY full-waypoint specialist.
- This is a policy-package integration, not a new single merged row-cache artifact and not a new metric/seconds claim.

## Weighted Package Summary

- rows: `47458`
- CI note: Stage42-IU has no single merged row-cache bootstrap yet. It uses Stage42-IT point metrics for TrajNet and Stage42-V multi-seed UCY specialist statistics; this is policy-package evidence.

| metric | weighted mean | domain min CI low | domain max CI high |
| --- | ---: | ---: | ---: |
| ADE all | 0.305568 | 0.161557 | 0.330147 |
| ADE t50 | 0.284549 | 0.212607 | 0.378387 |
| ADE t100 raw-frame diagnostic | 0.195280 | 0.145073 | 0.282687 |
| ADE hard/failure | 0.302105 | 0.172392 | 0.349043 |
| ADE easy degradation | -0.242171 | -0.303101 | 0.000000 |
| switch rate | 0.679387 | 0.382494 | 0.736062 |

## Per-Domain Metrics

| domain | source | rows | ADE all | ADE t50 | ADE t100 diag | ADE hard | easy degr | switch | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | `Stage42-IT` | 37918 | 0.320593 | 0.281795 | 0.190601 | 0.312518 | -0.303101 | 0.736062 | not_available |
| `UCY` | `Stage42-V` | 9540 | 0.245852 | 0.295497 | 0.213880 | 0.260718 | 0.000000 | 0.454123 | 0.325422 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_policy_tuning': False, 'stage42_it_no_leakage': {'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}, 'stage42_v_no_leakage': {'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_policy_tuning': False, 'train_only_normalization': True, 'val_only_policy_selection': True}}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Verification

- focused pytest: `3 passed in 0.91s`
- related pytest: `11 passed in 0.98s`
- full pytest: `.venv-pytorch/bin/python -m pytest tests -> 1113 passed in 841.55s (0:14:01)`

## Interpretation

- Stage42-IU removes the Stage42-IT UCY fallback-only weakness at policy-package level by importing the Stage42-V UCY specialist slice.
- The integration is positive on TrajNet and UCY for all, t50, t100 raw-frame diagnostic, and hard/failure slices, with easy preserved.
- The limitation is important: this is still not a unified row-level cache with one bootstrap over all selected rows. That remains a next step.
- All claims remain protected dataset-local/raw-frame 2.5D. No true 3D, no foundation, no metric/seconds, no Stage5C, and no SMC claim is made.
