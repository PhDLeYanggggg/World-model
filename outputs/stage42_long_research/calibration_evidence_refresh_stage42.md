# Stage42-AD Calibration Evidence Refresh

- source: `fresh_run`
- generated_at_utc: `2026-05-26T06:01:17.791747+00:00`
- git_commit: `a995971`
- input_hash: `34f63382c36e1b55a02af3fea0a4b6c46f151ed442a3c07be45c77c8cb57636c`
- gate: `10 / 10`
- verdict: `stage42_ad_calibration_evidence_refresh_pass`

## Current Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External domains 仍是 dataset-local / unverified weak-metric diagnostic。
- t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。
- homography / metric scale / effective seconds 未完成全局验证。
- Stage5C latent generative 未执行。
- SMC 未启用。
- Stage42 当前只做数据与标定 fresh audit，不训练、不下载 gated data、不写 large cache。

## Summary

- datasets_audited: `7`
- evidence_files_scanned: `1152`
- datasets_with_parseable_homography_like_matrices: `opentraj, eth_ucy, ucy`
- datasets_with_fps_evidence: `sdd, opentraj, eth_ucy, ucy`
- datasets_with_stride_or_dt_evidence: `sdd, opentraj, eth_ucy, ucy, tgsim, aerialmpt`
- datasets_with_scale_or_meter_evidence: `sdd, opentraj, eth_ucy, tgsim, aerialmpt`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`

## Dataset Evidence Table

| dataset | files | parseable H | fps | stride/dt | scale/meter | metric status | seconds status | allowed claim |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `sdd` | 5 | 0 | 3 | 3 | 3 | not_allowed_pixel_space | not_allowed_effective_seconds_unverified | pixel raw-frame only |
| `opentraj` | 1083 | 150 | 20 | 34 | 26 | weak_metric_candidate_requires_manual_validation | effective_seconds_candidate_requires_manual_validation | dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated |
| `eth_ucy` | 20 | 9 | 3 | 2 | 1 | weak_metric_candidate_requires_manual_validation | effective_seconds_candidate_requires_manual_validation | dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated |
| `trajnet` | 3 | 0 | 0 | 0 | 0 | weak_metric_candidate_requires_manual_validation | not_allowed_no_verified_fps_stride_pair | dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated |
| `ucy` | 11 | 7 | 1 | 1 | 0 | weak_metric_candidate_requires_manual_validation | effective_seconds_candidate_requires_manual_validation | dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated |
| `tgsim` | 2 | 0 | 0 | 2 | 2 | traffic_metric_diagnostic_only | time_values_dataset_diagnostic_only | traffic diagnostic metric only; not pedestrian/drone world-model success |
| `aerialmpt` | 28 | 0 | 0 | 28 | 28 | weak_metric_candidate_requires_manual_validation | not_allowed_no_verified_fps_stride_pair | dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated |

## Per-Dataset Notes

### Stanford Drone Dataset

- dataset_id: `sdd`
- data_role: `official_eval / supervised_training`
- known_coordinate_unit: `pixel`
- known_metric_status: `pixel_space; no verified homography/scale`
- local_paths_checked: `7 / 8`
- evidence_files_scanned: `5`
- homography_like_file_count: `4`
- parseable_homography_matrix_count: `0`
- fps_or_frame_rate_evidence_count: `3`
- stride_or_dt_evidence_count: `3`
- scale_or_meter_evidence_count: `3`
- coordinate_unit_evidence: `{'source pixel/world-2D; do not claim metric unless homography exists': 1, 'pixel': 1}`
- metric_claim_status: `not_allowed_pixel_space`
- seconds_claim_status: `not_allowed_effective_seconds_unverified`
- allowed_claim: `pixel raw-frame only`
- next_action: `keep_dataset_local_raw_frame_claim`

Notable evidence samples:

- `data/stage20_raw_index/stanford_drone/metadata.json`: H=0, fps=False, stride/dt=False, scale/meter=False, json=True
- `data/stage21_sdd_world_state/manifest.json`: H=0, fps=True, stride/dt=True, scale/meter=True, json=True
- `outputs/reports/stage23_sdd_time_geometry_audit.json`: H=0, fps=True, stride/dt=True, scale/meter=True, json=False
- `outputs/stage30_m3w_verified/time_geometry_raw_audit.json`: H=0, fps=True, stride/dt=True, scale/meter=True, json=False

### OpenTraj

- dataset_id: `opentraj`
- data_role: `external top-down source hub / loader input`
- known_coordinate_unit: `dataset-local mixed`
- known_metric_status: `dataset-local; underlying licenses/scales vary`
- local_paths_checked: `5 / 6`
- evidence_files_scanned: `1083`
- homography_like_file_count: `79`
- parseable_homography_matrix_count: `150`
- fps_or_frame_rate_evidence_count: `20`
- stride_or_dt_evidence_count: `34`
- scale_or_meter_evidence_count: `26`
- coordinate_unit_evidence: `{'source pixel/world-2D; do not claim metric unless homography exists': 1}`
- metric_claim_status: `weak_metric_candidate_requires_manual_validation`
- seconds_claim_status: `effective_seconds_candidate_requires_manual_validation`
- allowed_claim: `dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated`
- next_action: `manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim`

Notable evidence samples:

- `/Users/yangyue/Downloads/World/external_data/OpenTraj/README.md`: H=0, fps=True, stride/dt=True, scale/meter=True, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ATC/README.md`: H=0, fps=False, stride/dt=True, scale/meter=False, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/README.md`: H=0, fps=False, stride/dt=True, scale/meter=True, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_eth/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_eth/info.txt`: H=0, fps=True, stride/dt=False, scale/meter=False, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_hotel/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH/seq_hotel/info.txt`: H=0, fps=True, stride/dt=False, scale/meter=False, json=False
- `/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/Edinburgh/README.md`: H=0, fps=True, stride/dt=True, scale/meter=True, json=False

### ETH/UCY

- dataset_id: `eth_ucy`
- data_role: `external_eval / supervised_training`
- known_coordinate_unit: `dataset-local`
- known_metric_status: `unverified weak metric / dataset-local; do not claim metric`
- local_paths_checked: `7 / 8`
- evidence_files_scanned: `20`
- homography_like_file_count: `13`
- parseable_homography_matrix_count: `9`
- fps_or_frame_rate_evidence_count: `3`
- stride_or_dt_evidence_count: `2`
- scale_or_meter_evidence_count: `1`
- coordinate_unit_evidence: `{'world-2D if homography applied; otherwise pixel': 1, 'dataset_coordinate': 1}`
- metric_claim_status: `weak_metric_candidate_requires_manual_validation`
- seconds_claim_status: `effective_seconds_candidate_requires_manual_validation`
- allowed_claim: `dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated`
- next_action: `manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim`

Notable evidence samples:

- `data/stage20_raw_index/eth_ucy_full/metadata.json`: H=0, fps=False, stride/dt=False, scale/meter=False, json=True
- `data/stage5b_world_state/eth_ucy/metadata.json`: H=0, fps=False, stride/dt=False, scale/meter=False, json=True
- `external_data/OpenTraj/datasets/ETH/README.md`: H=0, fps=False, stride/dt=True, scale/meter=True, json=False
- `external_data/OpenTraj/datasets/ETH/seq_eth/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/ETH/seq_eth/info.txt`: H=0, fps=True, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/ETH/seq_hotel/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/ETH/seq_hotel/info.txt`: H=0, fps=True, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/README.md`: H=0, fps=True, stride/dt=True, scale/meter=False, json=False

### TrajNet++

- dataset_id: `trajnet`
- data_role: `external_eval / supervised_training`
- known_coordinate_unit: `dataset-local`
- known_metric_status: `dataset-local; terms/scale must be verified per source`
- local_paths_checked: `6 / 7`
- evidence_files_scanned: `3`
- homography_like_file_count: `1`
- parseable_homography_matrix_count: `0`
- fps_or_frame_rate_evidence_count: `0`
- stride_or_dt_evidence_count: `0`
- scale_or_meter_evidence_count: `0`
- coordinate_unit_evidence: `{'source pixel/world-2D; do not claim metric unless homography exists': 1, 'dataset_coordinate': 1}`
- metric_claim_status: `weak_metric_candidate_requires_manual_validation`
- seconds_claim_status: `not_allowed_no_verified_fps_stride_pair`
- allowed_claim: `dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated`
- next_action: `manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim`

Notable evidence samples:

- `data/stage20_raw_index/trajnet_full/metadata.json`: H=0, fps=False, stride/dt=False, scale/meter=False, json=True
- `data/stage5b_world_state/trajnet/metadata.json`: H=0, fps=False, stride/dt=False, scale/meter=False, json=True

### UCY Crowd

- dataset_id: `ucy`
- data_role: `external_eval / supervised_training`
- known_coordinate_unit: `dataset-local`
- known_metric_status: `dataset-local; not globally verified metric`
- local_paths_checked: `4 / 5`
- evidence_files_scanned: `11`
- homography_like_file_count: `9`
- parseable_homography_matrix_count: `7`
- fps_or_frame_rate_evidence_count: `1`
- stride_or_dt_evidence_count: `1`
- scale_or_meter_evidence_count: `0`
- coordinate_unit_evidence: `{'source pixel/world-2D; do not claim metric unless homography exists': 1}`
- metric_claim_status: `weak_metric_candidate_requires_manual_validation`
- seconds_claim_status: `effective_seconds_candidate_requires_manual_validation`
- allowed_claim: `dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated`
- next_action: `manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim`

Notable evidence samples:

- `data/stage20_raw_index/ucy_crowd/metadata.json`: H=0, fps=False, stride/dt=False, scale/meter=False, json=True
- `external_data/OpenTraj/datasets/UCY/README.md`: H=0, fps=True, stride/dt=True, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/students03/H-old.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/students03/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/zara01/H-cam.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/zara01/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/zara02/H-old.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False
- `external_data/OpenTraj/datasets/UCY/zara02/H.txt`: H=1, fps=False, stride/dt=False, scale/meter=False, json=False

### TGSIM

- dataset_id: `tgsim`
- data_role: `diagnostic_only`
- known_coordinate_unit: `traffic metric if source units verified by prior stage`
- known_metric_status: `metric diagnostic for traffic only; not pedestrian world-model success`
- local_paths_checked: `6 / 6`
- evidence_files_scanned: `2`
- homography_like_file_count: `0`
- parseable_homography_matrix_count: `0`
- fps_or_frame_rate_evidence_count: `0`
- stride_or_dt_evidence_count: `2`
- scale_or_meter_evidence_count: `2`
- coordinate_unit_evidence: `{'meter': 2}`
- metric_claim_status: `traffic_metric_diagnostic_only`
- seconds_claim_status: `time_values_dataset_diagnostic_only`
- allowed_claim: `traffic diagnostic metric only; not pedestrian/drone world-model success`
- next_action: `keep_as_traffic_diagnostic_only; do not use as pedestrian top-down world-model success`

Notable evidence samples:

- `data/stage5b_world_state/tgsim/metadata.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage5b_world_state/tgsim_i90/metadata.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True

### AerialMPT

- dataset_id: `aerialmpt`
- data_role: `external_eval candidate / diagnostic`
- known_coordinate_unit: `unknown / derived local`
- known_metric_status: `not verified in Stage42 audit`
- local_paths_checked: `2 / 4`
- evidence_files_scanned: `28`
- homography_like_file_count: `28`
- parseable_homography_matrix_count: `0`
- fps_or_frame_rate_evidence_count: `0`
- stride_or_dt_evidence_count: `28`
- scale_or_meter_evidence_count: `28`
- coordinate_unit_evidence: `{'pixel': 28}`
- metric_claim_status: `weak_metric_candidate_requires_manual_validation`
- seconds_claim_status: `not_allowed_no_verified_fps_stride_pair`
- allowed_claim: `dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated`
- next_action: `manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim`

Notable evidence samples:

- `data/stage11_scene_packs/aerialmpt/bauma1/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/bauma2/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/bauma3/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/bauma4/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/bauma5/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/bauma6/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/karlsplatz/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True
- `data/stage11_scene_packs/aerialmpt/marienplatz/scene_pack.json`: H=0, fps=False, stride/dt=True, scale/meter=True, json=True

## Conclusion

Stage42-AD separates calibration evidence existence from claim permission. ETH/UCY and UCY contain parseable homography-like files, and some metadata/text sources contain FPS, dt, or metric hints, but this is still insufficient for global pedestrian metric or seconds-level claims without source-specific validation of coordinate convention, annotation stride, homography direction, and scale. SDD remains pixel raw-frame; external pedestrian datasets remain dataset-local raw-frame; TGSIM remains traffic diagnostic only. Stage5C and SMC remain disabled.
