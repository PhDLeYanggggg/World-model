# Stage42-U UCY Candidate Endpoint-To-Full Bridge Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-26T03:55:25.199111+00:00`
- git_commit: `2eb3f61`
- gate: `7 / 8`
- verdict: `stage42_u_ucy_endpoint_to_full_bridge_failed_blocker`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-U 只审计 UCY endpoint candidate 能否桥接到 full-waypoint，不执行 Stage5C 或 SMC。
- future endpoints / waypoints 只作为 supervised labels 和 eval labels，不作为 inference input。
- Stage41 pure-UCY policy 是 train/val selected；Stage42-U 不用 test 调参。
- 如果 endpoint-to-full bridge 在 validation/test full-waypoint 上失败，必须标 blocker，不包装成 UCY 成功。

## Endpoint Candidate

- Stage41 source: `cached_verified_stage41_strict_pure_ucy_neural`
- best trial/mode: `pure_ucy_transformer` / `bounded_endpoint_residual`
- checkpoint exists: `True`

## Full-Waypoint Bridge Metrics

| split/slice | metrics |
| --- | --- |
| `val_bridge:matched_all` | rows=16103, matched=16103, ADE all=-0.042370, ADE t50=-0.460631, ADE hard=-0.048724, easy=0.629247, FDE t50=-0.091458, switch=0.937092 |
| `val_bridge:ETH_UCY_zara01_val` | rows=16103, matched=16103, ADE all=-0.042370, ADE t50=-0.460631, ADE hard=-0.048724, easy=0.629247, FDE t50=-0.091458, switch=0.937092 |
| `val_bridge:UCY_zara03_test` | rows=0, matched=0, ADE all=0.000000, ADE t50=0.000000, ADE hard=0.000000, easy=0.000000, FDE t50=0.000000, switch=0.000000 |
| `val_bridge:TrajNet_unmatched_control` | rows=37153, matched=0, ADE all=0.000000, ADE t50=0.000000, ADE hard=0.000000, easy=0.000000, FDE t50=0.000000, switch=0.000000 |
| `val_bridge:pure_ucy_val_source_zara01` | rows=16103, matched=16103, ADE all=-0.042370, ADE t50=-0.460631, ADE hard=-0.048724, easy=0.629247, FDE t50=-0.091458, switch=0.937092 |
| `test_bridge:matched_all` | rows=35441, matched=35441, ADE all=-0.029961, ADE t50=-0.373524, ADE hard=-0.036034, easy=0.461743, FDE t50=-0.064860, switch=0.792726 |
| `test_bridge:ETH_UCY_zara02_test` | rows=25901, matched=25901, ADE all=-0.014611, ADE t50=-0.329904, ADE hard=-0.018996, easy=0.403207, FDE t50=-0.013654, switch=0.778117 |
| `test_bridge:UCY_zara03_test` | rows=9540, matched=9540, ADE all=-0.070821, ADE t50=-0.492070, ADE hard=-0.083302, easy=0.566646, FDE t50=-0.210684, switch=0.832390 |
| `test_bridge:TrajNet_unmatched_control` | rows=20087, matched=0, ADE all=0.000000, ADE t50=0.000000, ADE hard=0.000000, easy=0.000000, FDE t50=0.000000, switch=0.000000 |
| `test_bridge:pure_ucy_test_sources_zara02_zara03` | rows=35441, matched=35441, ADE all=-0.029961, ADE t50=-0.373524, ADE hard=-0.036034, easy=0.461743, FDE t50=-0.064860, switch=0.792726 |

## Interpretation

- endpoint candidate available: `True`
- full-waypoint bridge deployable: `False`
- root cause: Stage41 pure-UCY endpoint residual is positive on endpoint FDE, but linear endpoint-to-waypoint interpolation is negative on Stage42 full-waypoint validation and UCY test. Endpoint success cannot be counted as full-waypoint world-state success.
- next action: Train/cache a UCY-aware full-waypoint candidate source with train/val source split, or learn a waypoint-shape bridge selected on validation; do not merge this endpoint bridge into Stage42-R/S deployable policy.

## No-Leakage / Claim Boundary

- no leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_policy_tuning': False, 'stage41_policy_train_val_selected': True}`
- claim boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
