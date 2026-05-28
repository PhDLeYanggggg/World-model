# Stage42-JG ETH_UCY Source-Specific Easy-Guard Feasibility

- source: `fresh_stage42_jg_eth_ucy_source_specific_easy_guard`
- generated_at_utc: `2026-05-28T16:21:38.179370+00:00`
- git_commit: `9088c43`
- input_hash: `7ffa4053ca78917ea31fd3231112c1f9530efb3d8c7513a24a40522e1ecba5a8`
- gate: `11 / 11`
- verdict: `stage42_jg_eth_ucy_source_specific_easy_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JG 是 ETH_UCY source-specific easy-guard feasibility：它只在 ETH_UCY 内做 source-disjoint CV。
- 这个阶段不把 ETH_UCY 写成 cross-domain zero-shot 成功；它只测试源级支持是否能修复 easy harm。
- 每个 fold 的 held-out source 不参与 train/validation policy selection。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `eth_ucy_source_specific_policy_partial_source_support`
- deployable_heldout_sources: `['ETH/seq_hotel/obsmat.txt', 'UCY/zara01/obsmat.txt']`
- blocked_heldout_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt', 'UCY/zara02/obsmat.txt']`
- positive_but_easy_unsafe_sources: `['UCY/students03/obsmat.txt', 'UCY/zara02/obsmat.txt']`
- next_action: If only part of ETH_UCY is deployable, keep ETH_UCY fallback-only by default and require source-identity/calibration or per-source support before promotion.

## Source-CV Fold Metrics

| heldout source | rows | cap | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | 21598 | 1.00 | 0.58% | -32.47% | 7.75% | 0.63% | -11.79% | 91.61% |
| `ETH/seq_hotel/obsmat.txt` | 16611 | 0.75 | 8.64% | 15.05% | 0.00% | 8.70% | -15.89% | 25.81% |
| `UCY/students03/obsmat.txt` | 70585 | 0.75 | 8.73% | 9.39% | 5.89% | 10.24% | 19.42% | 56.75% |
| `UCY/zara01/obsmat.txt` | 16103 | 0.75 | 12.50% | 17.97% | 0.00% | 11.43% | -24.69% | 64.82% |
| `UCY/zara02/obsmat.txt` | 25901 | 0.75 | 27.54% | 36.18% | 15.26% | 28.92% | 81.62% | 82.31% |

## Bootstrap CI

| heldout source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `all` | 0.05% | 0.58% | 1.12% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `t50` | -34.13% | -32.55% | -30.90% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `t100_raw_frame_diagnostic` | 7.48% | 7.74% | 8.00% | 2614 |
| `ETH/seq_eth/obsmat.txt` | `hard_failure` | 0.14% | 0.66% | 1.13% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `easy_degradation` | -17.64% | -13.32% | -8.66% | 1812 |
| `ETH/seq_hotel/obsmat.txt` | `all` | 8.35% | 8.64% | 8.96% | 16611 |
| `ETH/seq_hotel/obsmat.txt` | `t50` | 14.63% | 15.05% | 15.48% | 3994 |
| `ETH/seq_hotel/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 2560 |
| `ETH/seq_hotel/obsmat.txt` | `hard_failure` | 8.40% | 8.71% | 9.03% | 14776 |
| `ETH/seq_hotel/obsmat.txt` | `easy_degradation` | -21.04% | -18.89% | -16.73% | 5755 |
| `UCY/students03/obsmat.txt` | `all` | 8.50% | 8.73% | 8.96% | 70585 |
| `UCY/students03/obsmat.txt` | `t50` | 9.12% | 9.39% | 9.63% | 17529 |
| `UCY/students03/obsmat.txt` | `t100_raw_frame_diagnostic` | 5.57% | 5.89% | 6.20% | 15470 |
| `UCY/students03/obsmat.txt` | `hard_failure` | 10.02% | 10.24% | 10.45% | 54600 |
| `UCY/students03/obsmat.txt` | `easy_degradation` | 15.39% | 16.29% | 17.13% | 21424 |
| `UCY/zara01/obsmat.txt` | `all` | 12.23% | 12.50% | 12.77% | 16103 |
| `UCY/zara01/obsmat.txt` | `t50` | 17.64% | 17.97% | 18.30% | 3988 |
| `UCY/zara01/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 3251 |
| `UCY/zara01/obsmat.txt` | `hard_failure` | 11.14% | 11.42% | 11.68% | 14187 |
| `UCY/zara01/obsmat.txt` | `easy_degradation` | -35.85% | -32.84% | -30.06% | 1352 |
| `UCY/zara02/obsmat.txt` | `all` | 27.01% | 27.54% | 28.10% | 25901 |
| `UCY/zara02/obsmat.txt` | `t50` | 34.94% | 36.14% | 37.44% | 6422 |
| `UCY/zara02/obsmat.txt` | `t100_raw_frame_diagnostic` | 14.65% | 15.25% | 15.88% | 5433 |
| `UCY/zara02/obsmat.txt` | `hard_failure` | 28.41% | 28.91% | 29.50% | 20861 |
| `UCY/zara02/obsmat.txt` | `easy_degradation` | 42.50% | 44.94% | 47.32% | 6214 |

## Interpretation

- This stage tests source-specific ETH_UCY support, not cross-domain zero-shot transfer.
- If some ETH_UCY sources remain blocked, the default deployment boundary remains fallback-only for ETH_UCY unless source identity/calibration support is available.
- This remains dataset-local/raw-frame 2.5D evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
