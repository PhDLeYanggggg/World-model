# Stage42-IW Source-Level Row-Cache Mechanism Audit

- source: `fresh_run_row_cache_mechanism_audit_from_cached_verified_stage42iv_cache`
- generated_at_utc: `2026-05-28T07:31:46.533933+00:00`
- git_commit: `85442d8`
- input_hash: `a3d120a819753ae5e9cbd5f5a055beaacb22d2f90690927b16926eacdf587656`
- gate: `18 / 18`
- verdict: `stage42_iw_row_cache_mechanism_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IW 只审计 Stage42-IV 单一 source-level row-cache 能直接支持的机制证据。
- history / neighbor / goal / interaction 的独立贡献不能只靠这个 row-cache 证明，仍需要 retrained ablation evidence。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## What This Audit Can And Cannot Prove

- Directly supported by the Stage42-IV row-cache: safe-switch behavior, teacher/floor fallback usage, source/horizon slice behavior, easy preservation, waypoint sequence coverage/completeness, and row-level bootstrap.
- Not directly proven by this row-cache alone: independent causal contribution of history, neighbor, goal, interaction, JEPA, or Transformer modules. Those require retrained ablations; this report only records whether such reports exist as cached-verified supporting evidence.

## Main Metrics From The Same Merged Row Cache

| metric | value |
| --- | ---: |
| rows | 47458 |
| all improvement | 0.291543 |
| t50 improvement | 0.247045 |
| t100 raw-frame diagnostic improvement | 0.196335 |
| hard/failure improvement | 0.287273 |
| easy degradation | 0.000000 |
| switch rate | 0.702832 |

## Safe-Switch / Floor Mechanism

| field | value |
| --- | ---: |
| switch rows | 33355 |
| fallback rows | 14103 |
| switch rate | 0.702832 |
| mean gain, all rows | 0.134267 |
| mean gain, switched rows | 0.191037 |
| harm rate, switched rows | 0.076600 |
| hard/failure switch rate | 0.729644 |
| easy switch rate | 0.412616 |
| easy mean positive harm | 0.003129 |
| fallback exact floor rate | 1.000000 |

## Full-Waypoint Shape Coverage

- full_waypoint_rate: `0.675460`
- mean_valid_waypoints_per_row: `3.350921`
- mean raw residual from linear bridge: `0.104097`
- p90 raw residual from linear bridge: `0.277769`
- note: Residuals and turn angles are dataset-local/raw-frame shape diagnostics, not metric distances or seconds-level dynamics.

## By Domain

| domain | rows | switch | all | t50 | t100 diag | hard/failure | easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.736062 | 0.320593 | 0.281795 | 0.190601 | 0.312518 | 0.000000 |
| `UCY` | 9540 | 0.570755 | 0.196091 | 0.122892 | 0.213880 | 0.207360 | 0.000000 |

## Bootstrap CI

| slice | rows | mean | ci_low | ci_high | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 47458 | 0.291543 | 0.288667 | 0.294525 | 2000 |
| `t50` | 11538 | 0.247045 | 0.242612 | 0.251123 | 2000 |
| `t100_raw_frame_diagnostic` | 7048 | 0.196335 | 0.189978 | 0.202308 | 2000 |
| `hard_failure` | 35076 | 0.287273 | 0.283884 | 0.290605 | 2000 |
| `easy_degradation` | 11192 | 0.000000 | 0.000000 | 0.000000 | 2000 |

## Cached-Verified Supporting Evidence

| evidence | source | verdict | path |
| --- | --- | --- | --- |
| `source_level_incremental_ablation` | `cached_verified` | `stage42_ao_incremental_component_evidence_partial_or_negative` | `outputs/stage42_long_research/source_level_incremental_ablation_stage42.json` |
| `unified_ablation_evidence` | `cached_verified` | `stage42_y_unified_ablation_evidence_pass` | `outputs/stage42_long_research/unified_ablation_evidence_stage42.json` |
| `source_level_safety_floor_audit` | `cached_verified` | `stage42_at_source_level_fallback_audit_pass` | `outputs/stage42_long_research/source_level_safety_floor_audit_stage42.json` |
| `source_level_full_waypoint_eval` | `cached_verified` | `stage42_am_source_level_full_waypoint_eval_pass_positive` | `outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json` |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-IW strengthens the evidence package by showing the current source-level merged cache is not just a headline metric: its gains come from switched rows while fallback rows stay tied to the teacher/floor, and easy cases remain protected.
- The waypoint label audit confirms every row has at least two valid waypoint labels, while only a subset has all four valid waypoints; this is sequence-capable but not complete-full-waypoint coverage on every row.
- Module claims for history, neighbor, goal, interaction, JEPA, and Transformer must still be grounded in retrained ablations, not inferred from this row-cache summary.
