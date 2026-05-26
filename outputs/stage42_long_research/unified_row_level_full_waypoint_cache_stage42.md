# Stage42-X Unified Row-Level Full-Waypoint Cache

- source: `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions`
- generated_at_utc: `2026-05-26T04:38:59.967645+00:00`
- git_commit: `deaa13b`
- input_hash: `32df67bd086ca4748f9c9fcbf3708725ff37125964e9c47b3ea1b32325e24d4c`
- cache_hash: `ffa31b2525fa1a10db356ac5b1ef78602e44bc6f065c63cfc05ac29083e08937`
- gate: `16 / 16`
- verdict: `stage42_x_unified_row_level_full_waypoint_cache_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-X 构建统一 row-level full-waypoint cache：ETH_UCY/TrajNet 来自 Stage42-S，UCY 来自 Stage42-V。
- future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。
- UCY rows 通过 source/scene/horizon/current/future/waypoint alignment 校验后替换；Stage42-V ETH_UCY slice 不重复计入。
- 统一 bootstrap 基于 merged row-level arrays，而不是 domain-level weighted summary。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Row-Level Merge

- ETH_UCY and TrajNet rows keep the Stage42-S row-cache combo outputs.
- UCY rows are replaced by Stage42-V strict pure-UCY full-waypoint predictions.
- Stage42-V ETH_UCY rows are not imported, preventing double counting.
- Fallback UCY rows use the exact global Stage42-S floor errors to avoid tiny normalizer mismatches.

- cache_dir: `data/stage42_unified_full_waypoint_cache` (not committed)
- UCY alignment: `{'global_ucy_rows': 9540, 'stage42v_ucy_rows': 9540, 'source_file_order': True, 'scene_id_order': True, 'horizon_order': True, 'current_xy_match': True, 'future_xy_match': True, 'waypoint_xy_match': True, 'waypoint_valid_match': True, 'normalizer_max_abs_diff': 0.05018138885498047}`

## Seed Summary

| metric | mean | ci_low | ci_high |
| --- | ---: | ---: | ---: |
| ADE all | 0.090014 | 0.082449 | 0.097579 |
| ADE t50 | 0.061094 | 0.053671 | 0.068517 |
| ADE t100 raw-frame diagnostic | 0.081533 | 0.052781 | 0.110285 |
| ADE hard/failure | 0.093746 | 0.086531 | 0.100961 |
| ADE easy degradation | 0.001102 | -0.001058 | 0.003262 |
| FDE t50 | 0.153762 | 0.122384 | 0.185141 |
| switch rate | 0.232591 | 0.216137 | 0.249046 |

## Row Bootstrap Over Seed-Mean Arrays

| slice | rows | mean | ci_low | ci_high |
| --- | ---: | ---: | ---: | ---: |
| `all` | 55528 | 0.043083 | 0.041525 | 0.044623 |
| `t50` | 13689 | 0.029896 | 0.027880 | 0.031819 |
| `t100_raw_frame_diagnostic` | 9905 | 0.086146 | 0.078598 | 0.093690 |
| `hard_failure` | 41741 | 0.054453 | 0.052389 | 0.056498 |
| `easy` | 16739 | 0.000027 | -0.000164 | 0.000255 |

## Per-Domain Stress

| domain | rows | ADE all | ADE t50 | ADE hard | easy degr | FDE t50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.042817 | 0.017093 | 0.044320 | 0.001614 | 0.059435 |
| `TrajNet` | 20087 | 0.102635 | 0.097465 | 0.108920 | 0.004552 | 0.235413 |
| `UCY` | 9540 | 0.196091 | 0.122892 | 0.207360 | 0.000000 | 0.292236 |

## Interpretation

- Stage42-X upgrades Stage42-W from a domain-level package to a row-level merged cache with unified bootstrap.
- This is stronger full-waypoint branch evidence across ETH_UCY, TrajNet, and UCY.
- It remains protected and dataset-local raw-frame 2.5D evidence; it is not Stage5C, SMC, metric, seconds-level, or true 3D.
