# Stage42-IV Source-Level Row-Cache Full-Waypoint Integration

- source: `fresh_run_current_source_level_row_cache_and_cached_verified_stage42v_ucy`
- generated_at_utc: `2026-05-29T04:58:57.306037+00:00`
- git_commit: `e043235`
- input_hash: `715efb270a4c0334b568fcdb1cd1dc57b3cfcd23f57fb54093b16b745e86f380`
- cache_hash: `ff0213ae55c873796c8ed6e374a20729ec60db1ea2662b9ca703ace0d809a87f`
- gate: `20 / 20`
- verdict: `stage42_iv_source_level_row_cache_integration_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IV 把 Stage42-IT/Stage42-IU source-level full-waypoint evidence 升级为单一 row-level merged cache。
- TrajNet rows 来自当前 Stage42-IT source-level full-waypoint rerun；UCY rows 来自 Stage42-V UCY full-waypoint specialist。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## What This Adds Beyond Stage42-IU

- Stage42-IU was a source-level policy-package composition: TrajNet from Stage42-IT and UCY from Stage42-V.
- Stage42-IV exports one row-level source-level cache for that same TrajNet+UCY test protocol and reruns bootstrap over merged row arrays.
- UCY replacement is accepted only because horizon/current/future/waypoint geometry aligns row-by-row. Source text ids differ, so geometry alignment is the claim support.

## Cache And Alignment

- cache_path: `data/stage42_source_level_full_waypoint_cache/stage42iv_source_level_merged_cache.npz` (not committed)
- source_level_test_domains: `{'TrajNet': 37918, 'UCY': 9540}`
- alignment: `{'source_ucy_rows': 9540, 'stage42v_ucy_rows': 9540, 'horizon_order': True, 'current_xy_match': True, 'future_xy_match': True, 'waypoint_xy_match': True, 'waypoint_valid_match': True, 'source_file_text_match': False, 'scene_id_text_match': False, 'normalizer_max_abs_diff': 73.75215673446655, 'strict_geometry_alignment_pass': True, 'text_id_note': 'source_file/scene_id text may differ across Stage42-IT and Stage42-V derivations; geometry and waypoint alignment are the required row-level evidence.'}`
- source_level_best: `{'lambda': 100.0, 'score': 1.836828805896121, 'policy_slice_count': 8, 'val_metric': {'rows': 23788, 'all_improvement': 0.4408939341861645, 't10_improvement': 0.6928791809089825, 't25_improvement': 0.3042341730290602, 't50_improvement': 0.47362173836075094, 't100_raw_frame_diagnostic_improvement': 0.33025201024339657, 'hard_failure_improvement': 0.43658325454875535, 'easy_degradation': -0.2874272837926436, 'switch_rate': 0.8334874726752984, 'harm_over_fallback': -0.2385891923291905}}`

## Merged Row-Level Metrics

| metric | value |
| --- | ---: |
| ADE all improvement | 0.291543 |
| ADE t50 improvement | 0.247045 |
| ADE t100 raw-frame diagnostic improvement | 0.196335 |
| ADE hard/failure improvement | 0.287273 |
| ADE easy degradation | 0.000000 |
| FDE t50 improvement | 0.287678 |
| switch rate | 0.702832 |

## Bootstrap CI Over Single Merged Row Cache

| slice | rows | mean | ci_low | ci_high | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 47458 | 0.291543 | 0.288551 | 0.294631 | 2000 |
| `t50` | 11538 | 0.247045 | 0.242930 | 0.251388 | 2000 |
| `t100_raw_frame_diagnostic` | 7048 | 0.196335 | 0.189983 | 0.202854 | 2000 |
| `hard_failure` | 35076 | 0.287273 | 0.283919 | 0.290560 | 2000 |
| `easy_degradation` | 11192 | 0.000000 | 0.000000 | 0.000000 | 2000 |
| `fde_t50` | 11538 | 0.287678 | 0.283393 | 0.291692 | 2000 |

## By Domain

| domain | rows | ADE all | ADE t50 | ADE t100 diag | ADE hard | easy degr | FDE t50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.320593 | 0.281795 | 0.190601 | 0.312518 | 0.000000 | 0.286366 |
| `UCY` | 9540 | 0.196091 | 0.122892 | 0.213880 | 0.207360 | 0.000000 | 0.292236 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Verification

- focused pytest: `.venv-pytorch/bin/python -m pytest tests/test_stage42_source_level_row_cache_integration.py tests/test_stage42_source_level_row_cache_mechanism_audit.py -> 6 passed in 57.12s`
- full pytest: `.venv-pytorch/bin/python -m pytest tests -> 1193 passed in 857.42s (0:14:17)`

## Interpretation

- Stage42-IV removes the Stage42-IU single-row-cache limitation for the TrajNet+UCY source-level full-waypoint package.
- The row-level merged bootstrap remains positive on all, t50, t100 raw-frame diagnostic, and hard/failure slices, with easy preserved.
- This is still protected dataset-local/raw-frame 2.5D evidence. It is not true 3D, not a foundation model, not metric/seconds-level evidence, and does not execute Stage5C or SMC.
