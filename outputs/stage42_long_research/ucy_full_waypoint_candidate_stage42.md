# Stage42-V Strict Pure-UCY Full-Waypoint Candidate

- source: `fresh_run`
- generated_at_utc: `2026-05-26T04:16:48.847524+00:00`
- git_commit: `e58c9d9`
- gate: `11 / 11`
- verdict: `stage42_v_ucy_full_waypoint_candidate_pass`
- deployment decision: `deploy_stage42v_ucy_full_waypoint_candidate`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-V 训练 UCY-aware full-waypoint candidate；不是 endpoint-to-linear bridge 成功包装。
- future waypoints 只作为 train/val supervised labels 和 test eval labels，不作为 inference input。
- policy 只在 UCY validation source zara01 上选择，test zara02/zara03 只评估一次。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Data

- train: `{'rows': 117808, 'sources': ['UCY/students01/students001-trajnet.txt', 'UCY/students03/obsmat.txt'], 't10': 34363, 't25': 31735, 't50': 29112, 't100': 22598, 'hard': 85739, 'easy': 38264, 'valid_waypoint_rows': 117808, 'all_waypoints_valid': 83442}`
- val: `{'rows': 16103, 'sources': ['UCY/zara01/obsmat.txt'], 't10': 4580, 't25': 4284, 't50': 3988, 't100': 3251, 'hard': 14187, 'easy': 1352, 'valid_waypoint_rows': 16103, 'all_waypoints_valid': 11523}`
- test: `{'rows': 35441, 'sources': ['UCY/zara02/obsmat.txt', 'UCY/zara03/crowds_zara03.txt'], 't10': 10283, 't25': 9523, 't50': 8762, 't100': 6873, 'hard': 28250, 'easy': 8193, 'valid_waypoint_rows': 35441, 'all_waypoints_valid': 25158}`

## Summary By Trial

| trial | ADE all | ADE t50 | ADE t100 diag | hard | easy degr | FDE t50 | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ucy_full_waypoint_balanced` | 0.181347 | 0.240403 | 0.147630 | 0.191324 | 0.000000 | 0.249837 | 0.372967 |
| `ucy_full_waypoint_long_horizon` | 0.171968 | 0.234800 | 0.114838 | 0.179829 | 0.000000 | 0.272894 | 0.392925 |
| `ucy_full_waypoint_t50_hard` | 0.220755 | 0.290332 | 0.147461 | 0.229484 | 0.000000 | 0.334459 | 0.427791 |

## Best

- best trial: `ucy_full_waypoint_t50_hard`
- ADE all: `0.220755`
- ADE t50: `0.290332`
- ADE hard/failure: `0.229484`
- easy degradation: `0.000000`
- FDE t50: `0.334459`

## Interpretation

- UCY full-waypoint candidate deployable: `True`
- next action: If deployable, integrate as a UCY candidate source into Stage42-R/S combo; otherwise train a stronger waypoint-shape bridge or add UCY scene/goal/context features.

## No-Leakage / Claim Boundary

- no leakage: `{'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_policy_tuning': False, 'train_only_normalization': True, 'val_only_policy_selection': True}`
- claim boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
