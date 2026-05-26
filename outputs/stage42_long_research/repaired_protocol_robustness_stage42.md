# Stage42-AX Repaired Validation-Support Protocol Robustness Audit

- source: `cached_verified_from_stage42_aw`
- generated_at_utc: `2026-05-26T10:59:11.567521+00:00`
- git_commit: `56d7f3d`
- input_hash: `8663b3f733e25df57d59ec4d8029ab15bd72630920a8ae518ba293b5f3480caf`
- gate: `14 / 14`
- verdict: `stage42_ax_repaired_protocol_robustness_pass_with_t100_limit`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AX 是 Stage42-AW repaired validation-support protocol 的 robustness / paper-claim audit。
- Stage42-AX 不重新调 threshold，不读取 raw data，不执行 Stage5C，不启用 SMC。
- t100 仍是 raw-frame diagnostic；若 easy-safety 弱，必须写 limitation。
- dataset-local raw-frame 不能写成 metric 或 seconds-level。

## Global Bootstrap Stability

| metric | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.353076 | 0.356786 | 0.360389 | 47458 |
| `t50` | 0.285398 | 0.289509 | 0.294501 | 11538 |
| `t100_raw_frame_diagnostic` | 0.202944 | 0.210072 | 0.217191 | 7048 |
| `hard_failure` | 0.335229 | 0.339007 | 0.343176 | 35076 |
| `easy_degradation` | -0.611281 | -0.588131 | -0.566748 | 11192 |

## Domain Audit

| domain | rows | all | t50 | hard | easy | switch | positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.351423 | 0.302119 | 0.333797 | -0.361817 | 0.793739 | True |
| `UCY` | 9540 | 0.374492 | 0.245320 | 0.355073 | -0.418376 | 0.632075 | True |

## Horizon Audit

| horizon | rows | horizon metric | hard | easy | switch | weaknesses |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `10` | 15402 | 0.682892 | 0.697541 | -0.505393 | 0.860992 | `none` |
| `25` | 13470 | 0.360850 | 0.268682 | -0.502842 | 0.724722 | `none` |
| `50` | 11538 | 0.289698 | 0.289698 | -0.214067 | 0.891229 | `none` |
| `100` | 7048 | 0.210162 | 0.210162 | 0.023961 | 0.400255 | `easy_degradation_over_2pct` |

## Before / After UCY Repair

- before_after: `{'source': 'cached_verified_from_stage42_av_and_aw', 'ucy_before_blocker': 'no_validation_rows_for_domain_policy_selection_floor_only', 'ucy_before_all': 0.0, 'ucy_before_t50': 0.0, 'ucy_before_switch_rate': 0.0, 'ucy_after_all': 0.37449206739500784, 'ucy_after_t50': 0.24532046628354087, 'ucy_after_switch_rate': 0.6320754716981132, 'blocker_repaired': True}`

## Summary

- positive_domains: `['TrajNet', 'UCY']`
- weak_horizons: `['100']`
- uniform_domain_claim_allowed_under_repaired_protocol: `True`
- uniform_horizon_claim_allowed: `False`
- paper_claim: The repaired validation-support protocol has positive TrajNet and UCY source-level evidence, with global bootstrap support. Horizon 100 remains raw-frame diagnostic with an easy-safety limitation, so uniform horizon success remains disallowed.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'internal_val_from_train_only': True, 'test_sources_unchanged': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`
