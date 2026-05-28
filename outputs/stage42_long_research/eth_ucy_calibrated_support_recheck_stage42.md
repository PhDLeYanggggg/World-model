# Stage42-JM ETH_UCY Calibrated Support Recheck

- source: `fresh_stage42_jm_eth_ucy_calibrated_support_recheck`
- generated_at_utc: `2026-05-28T19:57:59.617471+00:00`
- git_commit: `3fe4121`
- input_hash: `9a28d1f2b9b89ba16f3c0a3e4ebc070919d2e5b4d339e166e0d9df7754d7144c`
- gate: `11 / 11`
- verdict: `stage42_jm_eth_ucy_calibrated_support_recheck_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JM rechecks the Stage42-JL blockers using BN source-specific time/geometry evidence.
- Source-specific calibration evidence can justify a restricted diagnostic subset, not a global metric/seconds M3W claim.
- Calibrated support signatures use past-only history/geometry summaries plus source-specific calibration flags; held-out labels are evaluation-only.
- No central velocity, no test endpoint goals, no test-threshold tuning, and no Stage5C/SMC execution are used.

## Summary

- decision: `calibrated_support_recheck_blocked_no_safe_deployment`
- targeted_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- source_specific_calibrated_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repaired_sources: `[]`
- still_blocked_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- next_action: Use the user-action package to add same-family calibrated source support or source-specific easy-harm/scene context before another deployment attempt.

## Held-Out Calibrated Support Metrics

| source | local calibration | nearest | distance | threshold | all | t50 | hard/failure | easy degradation | oracle t50 | deployable |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `source_specific_annotation_step_meter_coordinate_evidence` | `ETH/seq_hotel/obsmat.txt` | 4.008 | 3.941 | 0.00% | 0.00% | 0.00% | -0.00% | 53.80% | `False` |
| `UCY/students03/obsmat.txt` | `source_specific_annotation_step_meter_coordinate_evidence` | `UCY/zara02/obsmat.txt` | 11.812 | 3.821 | 0.00% | 0.00% | 0.00% | -0.00% | 39.14% | `False` |

## Support Family Candidates

### `ETH/seq_eth/obsmat.txt`

- decision_reason: `calibrated_source_out_of_support`
- selected_family: `None`

| rank | family | safe sources | mean all | mean t50 | mean hard | max easy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `constant_position` | 0/2 | -108.89% | -169.09% | -111.02% | 199.50% |
| 2 | `history_decay_baseline` | 0/2 | -162.38% | -123.33% | -154.58% | 266.04% |
| 3 | `neighbor_aware_decay_baseline` | 0/2 | -207.77% | -242.78% | -201.51% | 359.51% |
| 4 | `damped_velocity` | 0/2 | -343.50% | -410.11% | -335.36% | 567.04% |
| 5 | `constant_velocity_causal_fd` | 0/2 | -1584.03% | -2062.91% | -1607.44% | 2238.72% |
| 6 | `turn_rate` | 0/2 | -1787.80% | -2301.19% | -1812.98% | 2542.79% |
| 7 | `constant_acceleration` | 0/2 | -1723.20% | -2149.24% | -1729.60% | 2869.56% |
| 8 | `prototype_goal_directed_baseline` | 0/2 | -1779.04% | -2167.27% | -1779.72% | 3063.30% |

### `UCY/students03/obsmat.txt`

- decision_reason: `calibrated_source_out_of_support`
- selected_family: `None`

| rank | family | safe sources | mean all | mean t50 | mean hard | max easy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `constant_position` | 0/2 | -102.35% | -155.95% | -104.31% | 150.36% |
| 2 | `history_decay_baseline` | 0/2 | -158.30% | -115.04% | -150.28% | 266.04% |
| 3 | `neighbor_aware_decay_baseline` | 0/2 | -176.21% | -197.80% | -170.13% | 338.62% |
| 4 | `damped_velocity` | 0/2 | -334.19% | -387.89% | -325.68% | 515.75% |
| 5 | `constant_velocity_causal_fd` | 0/2 | -1537.48% | -1962.70% | -1560.95% | 1882.24% |
| 6 | `turn_rate` | 0/2 | -1731.58% | -2187.08% | -1756.42% | 2121.04% |
| 7 | `constant_acceleration` | 0/2 | -1680.10% | -2050.62% | -1684.82% | 2869.56% |
| 8 | `prototype_goal_directed_baseline` | 0/2 | -1771.95% | -2108.27% | -1751.32% | 4969.38% |

## User Action Required

- `ETH/seq_eth/obsmat.txt`: `provide_or_convert_additional_same-family_calibrated_source_support`
  - The held-out source remains outside the calibrated support hull; add a source with similar history/geometry statistics before deploying switches.
  - claim boundary: restricted source-specific diagnostic only; no global metric/seconds claim
- `UCY/students03/obsmat.txt`: `provide_or_convert_additional_same-family_calibrated_source_support`
  - The held-out source remains outside the calibrated support hull; add a source with similar history/geometry statistics before deploying switches.
  - claim boundary: restricted source-specific diagnostic only; no global metric/seconds claim

## Bootstrap CI

| source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `all` | 0.00% | 0.00% | 0.00% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `t50` | 0.00% | 0.00% | 0.00% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 2614 |
| `ETH/seq_eth/obsmat.txt` | `hard_failure` | 0.00% | 0.00% | 0.00% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `easy_degradation` | 0.00% | 0.00% | 0.00% | 1812 |
| `ETH/seq_eth/obsmat.txt` | `oracle_t50` | 52.96% | 53.81% | 54.62% | 5074 |
| `UCY/students03/obsmat.txt` | `all` | 0.00% | 0.00% | 0.00% | 70585 |
| `UCY/students03/obsmat.txt` | `t50` | 0.00% | 0.00% | 0.00% | 17529 |
| `UCY/students03/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 15470 |
| `UCY/students03/obsmat.txt` | `hard_failure` | 0.00% | 0.00% | 0.00% | 54600 |
| `UCY/students03/obsmat.txt` | `easy_degradation` | 0.00% | 0.00% | 0.00% | 21424 |
| `UCY/students03/obsmat.txt` | `oracle_t50` | 38.65% | 39.15% | 39.66% | 17529 |

## Interpretation

- JM uses source-specific BN calibration evidence to recheck support, but it does not upgrade the global model to metric/seconds-level.
- If calibrated support still cannot safely switch, the blocked sources need either source-specific geometry/context labels or additional same-family calibrated support.
- This remains a protected raw-frame / dataset-local 2.5D model track; Stage5C and SMC remain off.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'heldout_labels_for_support': False, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_or_seconds_claim': False, 'source_specific_seconds_hints_only': True, 'raw_frame_dataset_local_main_claim': True, 'stage5c_executed': False, 'smc_enabled': False}`
