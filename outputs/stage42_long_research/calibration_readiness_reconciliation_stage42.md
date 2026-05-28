# Stage42-JD Calibration Readiness Reconciliation

- source: `fresh_stage42_jd_calibration_readiness_reconciliation`
- generated_at_utc: `2026-05-28T15:01:41.598286+00:00`
- git_commit: `0f6b172`
- input_hash: `21eb76de69bba8fc5ce85f9ee46f2b8f61b82c68f9c3ed88ee9d463dfb9820ba`
- gate: `21 / 21`
- verdict: `stage42_jd_calibration_readiness_reconciliation_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JD 是 data/time/geometry calibration readiness reconciliation，不训练、不下载、不转换、不评估。
- ETH/UCY 等 source-specific metric/time hints 只能写成 candidates；除非 terms、guarded conversion、no-leakage 和 restricted eval 都完成，否则不能写成 metric/seconds result。
- SDD 仍是 pixel/raw-frame；TrajNet/OpenTraj 当前仍是 dataset-local/raw-frame 或 source-dependent。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `calibration_hints_reconciled_metric_time_claim_still_blocked`
- required_datasets_covered: `['aerialmpt', 'eth_ucy', 'opentraj', 'sdd', 'tgsim', 'trajnet', 'ucy']`
- direct_path_groups_found: `9 / 9`
- source_specific_candidate_count: `7`
- restricted_terms_confirmed: `False`
- restricted_metric_time_ready_now: `False`
- converted/evaluated restricted datasets now: `0 / 0`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`
- next_action: Ask user to confirm official source terms/local source identity for ETH/UCY calibrated candidates before guarded conversion; keep all current claims raw-frame/dataset-local.

## Dataset Readiness

| dataset | raw | converted | calibration | metric claim | seconds claim | terms/login/app |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `sdd` | `True` | `True` | `pixel_raw_frame_only` | `False` | `False` | `False` |
| `opentraj` | `True` | `True` | `calibration_files_found_but_not_validated` | `False` | `False` | `False` |
| `eth_ucy` | `True` | `True` | `calibration_files_found_but_not_validated` | `False` | `False` | `False` |
| `trajnet` | `True` | `True` | `not_verified` | `False` | `False` | `False` |
| `ucy` | `True` | `True` | `calibration_files_found_but_not_validated` | `False` | `False` | `False` |
| `tgsim` | `True` | `True` | `traffic_metric_diagnostic_only` | `True` | `False` | `False` |
| `aerialmpt` | `False` | `True` | `not_verified` | `False` | `False` | `False` |

## Direct Local Path Check

| key | found | existing paths |
| --- | ---: | --- |
| `sdd_raw` | `True` | `['external_data/StanfordDroneDataset']` |
| `sdd_converted` | `True` | `['data/stage21_sdd_world_state', 'data/stage24_sdd_fast_cache']` |
| `opentraj_raw` | `True` | `['external_data/OpenTraj']` |
| `opentraj_converted` | `True` | `['data/stage20_world_state/opentraj', 'data/stage31_external_feature_store']` |
| `eth_ucy_converted` | `True` | `['data/stage20_world_state/eth_ucy_full', 'data/stage5b_world_state/eth_ucy']` |
| `trajnet_converted` | `True` | `['data/stage20_world_state/trajnet_full', 'data/stage5b_world_state/trajnet']` |
| `ucy_converted` | `True` | `['data/stage20_world_state/ucy_crowd', 'data/stage37_t50_history']` |
| `tgsim_converted` | `True` | `['data/stage5_world_state/tgsim', 'data/stage5b_world_state/tgsim']` |
| `aerialmpt_converted` | `True` | `['data/aerialmpt', 'data/stage11_multiagent_episodes/aerialmpt']` |

## Source-Specific Metric/Time Candidates

| source | domain | H parseable | fps | timestep_s | status | claimable now |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `ETH_seq_eth` | `ETH_UCY` | `True` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |
| `ETH_seq_hotel` | `ETH_UCY` | `True` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |
| `UCY_zara01` | `UCY` | `True` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |
| `UCY_zara02` | `UCY` | `True` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |
| `UCY_zara03` | `UCY` | `True` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |
| `UCY_students03` | `UCY` | `True` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |
| `UCY_students01` | `UCY` | `False` | 2.5 | 0.4 | `candidate_after_terms_not_converted_or_evaluated` | `False` |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'download_executed': False, 'conversion_executed': False, 'training_executed': False, 'evaluation_executed': False, 'claim_reconciliation_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_claim': False, 'global_seconds_claim': False, 'restricted_metric_time_claim_allowed_now': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Existing local converted state is enough to continue raw-frame/dataset-local external validation and full-waypoint dynamics work.
- ETH/UCY source-specific H/FPS/timestep hints are valuable, but they remain candidates until user-confirmed terms and guarded conversion/evaluation happen.
- Current paper language must keep global metric/seconds and restricted metric/time claims blocked.
