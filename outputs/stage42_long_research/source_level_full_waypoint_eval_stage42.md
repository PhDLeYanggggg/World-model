# Stage42-AM Proposed Source-Level Full-Waypoint Evaluation

- source: `fresh_run`
- generated_at_utc: `2026-05-28T04:49:25.419437+00:00`
- git_commit: `cd619fb`
- input_hash: `d58853ad5efe0062b68f81cd6184ec4e5200a0843862541ba4285626a4d5c2e8`
- gate: `12 / 12`
- verdict: `stage42_am_source_level_full_waypoint_eval_pass_positive`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AM 是 proposed source-level split full-waypoint fresh evaluation，不是 metric 或 seconds-level 结果。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## What This Fixes From Stage42-AL

- Stage42-AL found that locked-policy stress rows were not a full proposed source-level split evaluation.
- Stage42-AM evaluates the proposed source-level test split directly: TrajNet `37918` rows and UCY `9540` rows.
- ETH_UCY remains train/val in this proposed split; ETH_UCY stress rows are not counted as source-level test evidence here.

## Split And Labels

- split_stats: `{'train': {'rows': 266745, 'domains': {'ETH_UCY': 134695, 'TrajNet': 75287, 'UCY': 56763}, 'scenes': 9, 'sources': 13, 't10': 83107, 't25': 73802, 't50': 64551, 't100': 45285, 'hard': 198685, 'failure': 96981, 'easy': 80855}, 'val': {'rows': 23788, 'domains': {'ETH_UCY': 16103, 'TrajNet': 7685}, 'scenes': 2, 'sources': 2, 't10': 7045, 't25': 6459, 't50': 5873, 't100': 4411, 'hard': 21872, 'failure': 10450, 'easy': 4454}, 'test': {'rows': 47458, 'domains': {'TrajNet': 37918, 'UCY': 9540}, 'scenes': 2, 'sources': 3, 't10': 15402, 't25': 13470, 't50': 11538, 't100': 7048, 'hard': 35076, 'failure': 19056, 'easy': 11192}}`
- label_stats: `{'rows': 337991, 'full_waypoint_rows': 233825, 'endpoint_only_rows': 104166, 'missing_track_rows': 0, 'test_rows': 47458, 'test_full_waypoint_rows': 32056}`
- floor: `{'source': 'fresh_run', 'type': 'train_horizon_selected_safe_causal_baseline', 'strongest_by_horizon': {10: 0, 25: 2, 50: 1, 100: 1}, 'geometry_diagnostics': {'safe_endpoint_max_abs_fde_error': 1.7858156996196362e-06, 'safe_endpoint_mean_abs_fde_error': 6.232531667367801e-08}}`
- feature_count: `166`

## Candidate Metrics On Proposed Source-Level Test

| candidate | rows | all ADE improvement | t50 ADE improvement | t100 raw-frame diag | hard/failure | easy degradation | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-horizon causal floor | 47458 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |
| ungated ridge diagnostic | 47458 | 0.443417 | 0.402571 | 0.457972 | 0.436217 | -0.266068 | 1.000000 |
| protected ridge source-level | 47458 | 0.245788 | 0.220171 | 0.143652 | 0.237494 | -0.256627 | 0.588099 |

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.242554 | 0.245853 | 0.248923 | 47458 |
| `t50` | 0.215923 | 0.220124 | 0.224522 | 11538 |
| `t100_raw_frame_diagnostic` | 0.137653 | 0.143621 | 0.150097 | 7048 |
| `hard_failure` | 0.233887 | 0.237527 | 0.240825 | 35076 |
| `easy_degradation` | -0.359604 | -0.345323 | -0.331098 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.320593 | 0.281795 | 0.190601 | 0.312518 | -0.303101 | 0.736062 |
| `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-AM closes the Stage42-AL proposed source-level evaluation gap with a positive protected ridge full-waypoint probe.
- This remains dataset-local raw-frame 2.5D evidence, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.
