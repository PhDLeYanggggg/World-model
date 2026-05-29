# Stage42-JV Source Slice Evidence Matrix

- source: `fresh_stage42_jv_source_slice_evidence_matrix_from_cached_verified_row_cache`
- generated_at_utc: `2026-05-29T06:18:11.087685+00:00`
- git_commit: `4156692`
- input_hash: `889dc9330c1d3951f0c098ce9dca5a1914a20397064f809b20dab7e72358a13f`
- gate: `18 / 18`
- verdict: `stage42_jv_source_slice_evidence_matrix_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JV 使用 cached_verified Stage42-IV row-level cache 做 fresh source/domain/horizon/slice 分解。
- Stage42-JV 不训练、不调 threshold、不下载、不转换，不把 slice synthesis 当新模型结果。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Cache And Input Status

- cache: `{'path': 'data/stage42_source_level_full_waypoint_cache/stage42iv_source_level_merged_cache.npz', 'exists': True, 'rows': 47458, 'source': 'cached_verified_stage42iv_row_level_cache'}`
- inputs: `{'iv_verdict': 'stage42_iv_source_level_row_cache_integration_pass', 'iw_verdict': 'stage42_iw_row_cache_mechanism_audit_pass', 'ju_verdict': 'stage42_ju_current_reviewer_replay_package_pass', 'iv_rows': 47458, 'iw_rows': 47458}`

## Core Slices

| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 47458 | 0.291543 | 0.278634 | 0.000000 | 0.702832 | 0.675460 |
| `hard` | 35076 | 0.287273 | 0.277157 | 0.000000 | 0.729644 | 0.684000 |
| `failure` | 19056 | 0.363413 | 0.331826 | 0.000000 | 0.806623 | 0.428159 |
| `hard_or_failure` | 35076 | 0.287273 | 0.277157 | 0.000000 | 0.729644 | 0.684000 |
| `easy` | 11192 | 0.256627 | 0.218765 | 0.000000 | 0.412616 | 0.734811 |
| `switched` | 33355 | 0.372722 | 0.361226 | 0.000000 | 1.000000 | 0.634957 |
| `fallback` | 14103 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.771254 |
| `full_waypoint_available` | 32056 | 0.220934 | 0.231506 | 0.000000 | 0.660688 | 1.000000 |
| `partial_waypoint_available` | 15402 | 0.549989 | 0.544622 | 0.000000 | 0.790547 | 0.000000 |

## Domain Metrics

| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.320593 | 0.288179 | 0.000000 | 0.736062 | 0.674508 |
| `UCY` | 9540 | 0.196091 | 0.247036 | 0.000000 | 0.570755 | 0.679245 |

## Horizon Metrics

| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 15402 | 0.549989 | 0.544622 | 0.000000 | 0.790547 | 0.000000 |
| `100` | 7048 | 0.196335 | 0.193457 | 0.000000 | 0.432605 | 1.000000 |
| `25` | 13470 | 0.233241 | 0.219655 | 0.000000 | 0.664662 | 1.000000 |
| `50` | 11538 | 0.247045 | 0.287678 | 0.000000 | 0.795372 | 1.000000 |

## Domain x Horizon Metrics

## TrajNet

| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 12342 | 0.609766 | 0.608102 | 0.000000 | 0.785043 | 0.000000 |
| `25` | 10770 | 0.294623 | 0.276856 | 0.000000 | 0.831291 | 1.000000 |
| `50` | 9198 | 0.281795 | 0.286366 | 0.000000 | 0.811155 | 1.000000 |
| `100` | 5608 | 0.190601 | 0.170014 | 0.000000 | 0.322218 | 1.000000 |

## UCY

| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 3060 | 0.368956 | 0.352371 | 0.000000 | 0.812745 | 0.000000 |
| `25` | 2700 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| `50` | 2340 | 0.122892 | 0.292236 | 0.000000 | 0.733333 | 1.000000 |
| `100` | 1440 | 0.213880 | 0.266308 | 0.000000 | 0.862500 | 1.000000 |

## Source File Metrics (rows >= 200)

| slice | rows | ADE improvement | FDE improvement | easy degradation | switch rate | full waypoint rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_data/OpenTraj/datasets/TrajNet/Test/crowds/students002.txt` | 765 | 0.656027 | 0.628648 | 0.000000 | 0.992157 | 0.444444 |
| `external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` | 9540 | 0.196091 | 0.247036 | 0.000000 | 0.570755 | 0.679245 |
| `external_data/OpenTraj/datasets/TrajNet/Train/crowds/students003.txt` | 37153 | 0.310346 | 0.279473 | 0.000000 | 0.730789 | 0.679245 |

## Diagnostics

- negative_or_weak_source_files: `[]`
- weak_positive_source_files: `[]`
- interpretation: Positive aggregate evidence is decomposed by source/domain/horizon/slice; weak or negative source files must not be hidden.

## Gate

| gate | pass |
| --- | ---: |
| `cache_exists` | `True` |
| `row_count_matches_iv` | `True` |
| `two_external_domains_present` | `True` |
| `all_metric_positive_easy_safe` | `True` |
| `both_domains_positive_easy_safe` | `True` |
| `t50_positive_both_domains` | `True` |
| `t100_raw_positive_both_domains` | `True` |
| `hard_failure_positive` | `True` |
| `easy_safe` | `True` |
| `switch_slice_positive` | `True` |
| `fallback_slice_exact_or_nonharmful` | `True` |
| `full_waypoint_available_positive` | `True` |
| `source_file_metrics_recorded` | `True` |
| `negative_source_slices_reported` | `True` |
| `horizon_metrics_recorded` | `True` |
| `no_metric_seconds_or_3d_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Claim Boundary

- no_leakage: `{'future_endpoint_input_absent': True, 'future_waypoint_input_absent': True, 'central_velocity_absent': True, 'test_endpoint_goals_absent': True, 'test_threshold_tuning_absent': True, 'future_labels_eval_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'slice_synthesis_not_new_training': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-JV supports a stronger reviewer-facing statement that the protected row-cache/full-waypoint result is not only aggregate-positive: it is positive and easy-safe across the two current external domains and raw-frame horizons in this cache.
- It still does not prove metric/seconds-level dynamics, true 3D, foundation behavior, or independent scene/goal/neighbor/JEPA/Transformer claims.
- Source-file weak slices remain visible in the JSON/MD diagnostics and must be addressed before broader claims.
