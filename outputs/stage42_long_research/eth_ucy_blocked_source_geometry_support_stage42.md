# Stage42-JJ ETH_UCY Blocked-Source Geometry/Family Support

- source: `fresh_stage42_jj_eth_ucy_blocked_source_geometry_support`
- generated_at_utc: `2026-05-28T17:50:07.133937+00:00`
- git_commit: `a706b43`
- input_hash: `ac4f779835199b93ed19aad9dbed71a52e715e004dfe07656b4b5e3e17390c6d`
- gate: `11 / 11`
- verdict: `stage42_jj_eth_ucy_blocked_source_geometry_support_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JJ audits why Stage42-JI blocked ETH_UCY sources remain blocked after source-robust harm-aware guards.
- It tests causal family-baseline support and source geometry/history distribution shift without using held-out test for selection.
- future waypoints / endpoints are labels/eval only, never inference inputs.
- No central velocity, no test endpoint goals, and no test-threshold tuning are used.
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `blocked_sources_not_repaired_family_support_diagnostic`
- targeted_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repaired_sources: `[]`
- still_blocked_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- t50_family_oracle_headroom_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- next_action: If family oracle exists but static source policy fails, add source-specific history/goal geometry features; if family oracle is weak, acquire/calibrate new source support.

## Static Family Repair Attempt

| source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | oracle t50 | deployable | blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH/seq_eth/obsmat.txt` | 21598 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | 53.80% | `False` | `['t50_static_family_lift_insufficient', 'hard_failure_static_family_lift_insufficient']` |
| `UCY/students03/obsmat.txt` | 70585 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | 39.14% | `False` | `['t50_static_family_lift_insufficient', 'hard_failure_static_family_lift_insufficient']` |

## T50 Family Table

### `ETH/seq_eth/obsmat.txt`

| rank | family | t50 improvement | t50 easy degradation | rows |
| ---: | --- | ---: | ---: | ---: |
| 1 | `history_decay_baseline` | 44.58% | 97.09% | 5074 |
| 2 | `neighbor_aware_decay_baseline` | 6.11% | 150.47% | 5074 |
| 3 | `damped_velocity` | -80.96% | 398.51% | 5074 |
| 4 | `constant_position` | -142.03% | 343.19% | 5074 |
| 5 | `constant_velocity_causal_fd` | -953.83% | 2294.57% | 5074 |
| 6 | `constant_acceleration` | -1035.35% | 2766.02% | 5074 |
| 7 | `prototype_goal_directed_baseline` | -1086.18% | 2828.11% | 5074 |
| 8 | `turn_rate` | -1172.02% | 2675.79% | 5074 |

### `UCY/students03/obsmat.txt`

| rank | family | t50 improvement | t50 easy degradation | rows |
| ---: | --- | ---: | ---: | ---: |
| 1 | `neighbor_aware_decay_baseline` | 15.78% | 59.65% | 17529 |
| 2 | `constant_position` | -74.95% | 186.62% | 17529 |
| 3 | `history_decay_baseline` | -80.78% | 214.27% | 17529 |
| 4 | `damped_velocity` | -267.37% | 521.63% | 17529 |
| 5 | `constant_velocity_causal_fd` | -1376.51% | 2342.89% | 17529 |
| 6 | `turn_rate` | -1508.46% | 2553.16% | 17529 |
| 7 | `prototype_goal_directed_baseline` | -1528.21% | 3150.39% | 17529 |
| 8 | `constant_acceleration` | -1658.82% | 3410.51% | 17529 |

## Top Distribution Shifts

### `ETH/seq_eth/obsmat.txt`

| feature | train mean | test mean | z-gap |
| --- | ---: | ---: | ---: |
| `scale` | 5.1214 | 6.9848 | 1.318 |
| `history_scalar_3` | 14.4414 | 3.9982 | 1.180 |
| `history_scalar_1` | 29.3619 | 8.5156 | 1.179 |
| `history_scalar_2` | 1.0531 | 1.7978 | 0.789 |
| `history_scalar_8` | 28.3747 | 15.1162 | 0.630 |
| `track_length` | 86.2399 | 36.4515 | 0.502 |
| `history_scalar_5` | 0.2304 | 0.3350 | 0.458 |
| `history_scalar_7` | 11.1008 | 4.5473 | 0.315 |

### `UCY/students03/obsmat.txt`

| feature | train mean | test mean | z-gap |
| --- | ---: | ---: | ---: |
| `history_scalar_1` | 8.5520 | 41.8089 | 7.059 |
| `history_scalar_3` | 4.0130 | 20.6706 | 7.031 |
| `history_scalar_7` | 3.5083 | 15.2222 | 2.583 |
| `scale` | 6.2543 | 4.6900 | 0.857 |
| `history_scalar_8` | 20.7356 | 32.4160 | 0.681 |
| `history_scalar_2` | 1.5211 | 0.8539 | 0.439 |
| `history_scalar_6` | 5049.9793 | 15788.8589 | 0.322 |
| `track_length` | 63.7186 | 94.8883 | 0.319 |

## Bootstrap CI

| source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `static_all` | 0.00% | 0.00% | 0.00% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `static_t50` | 0.00% | 0.00% | 0.00% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `static_hard_failure` | 0.00% | 0.00% | 0.00% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `static_easy_degradation` | 0.00% | 0.00% | 0.00% | 1812 |
| `ETH/seq_eth/obsmat.txt` | `oracle_t50` | 52.94% | 53.79% | 54.53% | 5074 |
| `UCY/students03/obsmat.txt` | `static_all` | 0.00% | 0.00% | 0.00% | 70585 |
| `UCY/students03/obsmat.txt` | `static_t50` | 0.00% | 0.00% | 0.00% | 17529 |
| `UCY/students03/obsmat.txt` | `static_hard_failure` | 0.00% | 0.00% | 0.00% | 54600 |
| `UCY/students03/obsmat.txt` | `static_easy_degradation` | 0.00% | 0.00% | 0.00% | 21424 |
| `UCY/students03/obsmat.txt` | `oracle_t50` | 38.65% | 39.14% | 39.66% | 17529 |

## Interpretation

- This stage checks whether remaining JI blockers are caused by missing baseline-family support or source geometry/history shift.
- Static family repair is causal and validation-selected, but source identity is still a deployment precondition.
- Failed sources remain fallback-only; this remains raw-frame / dataset-local 2.5D evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_selection': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
