# Stage42-A Data And Calibration Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-25T19:27:01.408542+00:00`
- git_commit: `a2a368f`
- input_hash: `a6ca6bc9180dc01a00d7b8f41c1914464a1278258bbec3bfada9cc02c117cec7`

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
- raw_paths_found: `6`
- converted_paths_found: `7`
- external_domains_ready_from_existing_state: `opentraj, eth_ucy, trajnet, ucy`
- metric_claim_ready_datasets: `tgsim`
- seconds_claim_ready_datasets: `none`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`
- stage42_b_external_validation_ready: `True`
- stage42_c_full_waypoint_prereq_ready: `True`

## Dataset Audit Table

| dataset | raw found | converted found | role | calibration state | metric claim | seconds claim | next use |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| `sdd` | `True` | `True` | official_eval / supervised_training | pixel_raw_frame_only | `False` | `False` | Stage42-B/C ready |
| `opentraj` | `True` | `True` | external top-down source hub / loader input | calibration_files_found_but_not_validated | `False` | `False` | Stage42-B/C ready |
| `eth_ucy` | `True` | `True` | external_eval / supervised_training | calibration_files_found_but_not_validated | `False` | `False` | Stage42-B/C ready |
| `trajnet` | `True` | `True` | external_eval / supervised_training | not_verified | `False` | `False` | Stage42-B/C ready |
| `ucy` | `True` | `True` | external_eval / supervised_training | calibration_files_found_but_not_validated | `False` | `False` | Stage42-B/C ready |
| `tgsim` | `True` | `True` | diagnostic_only | traffic_metric_diagnostic_only | `True` | `False` | Stage42-B/C ready |
| `aerialmpt` | `False` | `True` | external_eval candidate / diagnostic | not_verified | `False` | `False` | Stage42-B/C ready |

## Per-Dataset Notes

### Stanford Drone Dataset

- source: `fresh_run`
- domain: real top-down drone pedestrian/mixed-agent
- official_hint: `https://cvgl.stanford.edu/projects/uav_data/`
- coordinate_unit: `pixel`
- metric_status: `pixel_space; no verified homography/scale`
- raw_path_found: `True`
- converted_path_found: `True`
- has_video: `True`
- has_image_or_scene_pack: `True`
- homography_like_files_found: `0`
- scale_like_files_found: `0`
- calibration_state: `pixel_raw_frame_only`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `Stanford SDD non-commercial / custom access terms`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `True`

Next actions:
- do_not_auto_download: requires terms/login/application or source-specific approval
- keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified

### OpenTraj

- source: `fresh_run`
- domain: toolkit plus multiple underlying pedestrian/traffic/crowd datasets
- official_hint: `https://github.com/crowdbotp/OpenTraj`
- coordinate_unit: `dataset-local mixed`
- metric_status: `dataset-local; underlying licenses/scales vary`
- raw_path_found: `True`
- converted_path_found: `True`
- has_video: `True`
- has_image_or_scene_pack: `True`
- homography_like_files_found: `16`
- scale_like_files_found: `20`
- calibration_state: `calibration_files_found_but_not_validated`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `MIT for toolkit; underlying datasets keep their own licenses`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `False`

Next actions:
- keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified
- Stage42-B: rebuild source/scene-level split from raw/feature rows before new validation

### ETH/UCY

- source: `fresh_run`
- domain: fixed-camera/top-down pedestrian trajectories
- official_hint: `ETH/BIWI + UCY original dataset pages; verify source-specific terms`
- coordinate_unit: `dataset-local`
- metric_status: `unverified weak metric / dataset-local; do not claim metric`
- raw_path_found: `True`
- converted_path_found: `True`
- has_video: `True`
- has_image_or_scene_pack: `True`
- homography_like_files_found: `6`
- scale_like_files_found: `0`
- calibration_state: `calibration_files_found_but_not_validated`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `research dataset terms; verify original source`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `False`

Next actions:
- keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified
- Stage42-B: rebuild source/scene-level split from raw/feature rows before new validation
- Stage42-C: use as priority external full-waypoint dynamics domain

### TrajNet++

- source: `fresh_run`
- domain: pedestrian trajectory forecasting benchmark
- official_hint: `https://www.epfl.ch/labs/vita/datasets/`
- coordinate_unit: `dataset-local`
- metric_status: `dataset-local; terms/scale must be verified per source`
- raw_path_found: `True`
- converted_path_found: `True`
- has_video: `False`
- has_image_or_scene_pack: `False`
- homography_like_files_found: `0`
- scale_like_files_found: `0`
- calibration_state: `not_verified`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `dataset-specific / challenge terms`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `False`

Next actions:
- keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified
- Stage42-B: rebuild source/scene-level split from raw/feature rows before new validation
- Stage42-C: use as priority external full-waypoint dynamics domain

### UCY Crowd

- source: `fresh_run`
- domain: pedestrian crowd trajectories
- official_hint: `http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- coordinate_unit: `dataset-local`
- metric_status: `dataset-local; not globally verified metric`
- raw_path_found: `True`
- converted_path_found: `True`
- has_video: `True`
- has_image_or_scene_pack: `True`
- homography_like_files_found: `4`
- scale_like_files_found: `0`
- calibration_state: `calibration_files_found_but_not_validated`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `UCY crowd research terms; verify before use`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `False`

Next actions:
- keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified
- Stage42-B: rebuild source/scene-level split from raw/feature rows before new validation
- Stage42-C: use as priority external full-waypoint dynamics domain

### TGSIM

- source: `fresh_run`
- domain: traffic vehicle trajectories
- official_hint: `https://data.transportation.gov/`
- coordinate_unit: `traffic metric if source units verified by prior stage`
- metric_status: `metric diagnostic for traffic only; not pedestrian world-model success`
- raw_path_found: `True`
- converted_path_found: `True`
- has_video: `False`
- has_image_or_scene_pack: `False`
- homography_like_files_found: `0`
- scale_like_files_found: `0`
- calibration_state: `traffic_metric_diagnostic_only`
- metric_claim_allowed: `True`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `trajectory data; official portal required`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `True`

Next actions:
- do_not_auto_download: requires terms/login/application or source-specific approval

### AerialMPT

- source: `fresh_run`
- domain: aerial pedestrian/crowd trajectories
- official_hint: `DLR AerialMPT official page required; local prior stages have derived scene packs`
- coordinate_unit: `unknown / derived local`
- metric_status: `not verified in Stage42 audit`
- raw_path_found: `False`
- converted_path_found: `True`
- has_video: `False`
- has_image_or_scene_pack: `False`
- homography_like_files_found: `0`
- scale_like_files_found: `0`
- calibration_state: `not_verified`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`
- legal_source: `cached_verified`
- license_name: `unknown`
- auto_download_allowed: `False`
- requires_terms_or_login_or_application: `True`

Next actions:
- do_not_auto_download: requires terms/login/application or source-specific approval
- keep_raw_frame_dataset_local_claim_until_fps_stride_homography_scale_verified

## Stage42-A Conclusion

Stage42 can proceed to external validation and full-waypoint dynamics from existing local converted state, but it cannot make metric or seconds-level claims. SDD remains pixel raw-frame. External pedestrian domains remain dataset-local raw-frame / unverified weak-metric diagnostics. TGSIM may carry traffic metric diagnostics only and cannot be used as pedestrian world-model success.

<!-- STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT:START -->
## Stage42-DU Raw Source Time/Geometry Hint Addendum

- source: `fresh_hint_audit_from_local_raw_sources_after_stage42_dt`
- role: H/FPS/stride hint extraction only; no conversion and no metric/seconds claim.
- gate: `14 / 14`; verdict `stage42_du_raw_source_time_geometry_hint_audit_pass`.
- targets checked: `7`; H-hint targets: `2`; time-hint targets: `3`; stride-hint targets: `4`.
- metric/time subset hint targets: `2`; legal conversion ready targets: `0`.
- H/FPS/stride hints remain hints until source/legal confirmation and no-leakage conversion are complete.
<!-- STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT:END -->

<!-- STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST:START -->
## Stage42-DV Calibration Candidate Manifest

- source: `fresh_synthesis_from_stage42_du_bn`
- role: ranks raw-source calibration candidates; no conversion, no evaluation, no metric/seconds claim.
- gate: `13 / 13`; verdict `stage42_dv_calibration_candidate_manifest_pass`.
- source-specific candidate targets: `2`; time/stride candidate targets: `1`.
- conversion-ready targets: `0`; converted/evaluated now: `0` / `0`.
- Candidate status remains blocked by user-confirmed terms/source/path/version and no-leakage conversion.
<!-- STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST:END -->

<!-- STAGE42_DW_SOURCE_SPECIFIC_CONVERSION_DRY_RUN:START -->
## Stage42-DW Source-Specific Conversion Dry-Run

- source: `fresh_source_specific_conversion_dry_run_from_stage42_dv`
- role: technical dry-run for calibrated UCY/ETH candidates; no conversion, no evaluation, no metric/seconds claim.
- gate: `15 / 15`; verdict `stage42_dw_source_specific_conversion_dry_run_pass`.
- sources checked: `6`; technical ready after terms: `5`.
- technical not-ready sources: `['UCY_zara03']`.
- estimated t50/t100 windows: `10060` / `5696`.
- source-CV domains after terms: `['UCY']`.
- Conversion remains blocked by terms/source/path/version confirmation.
<!-- STAGE42_DW_SOURCE_SPECIFIC_CONVERSION_DRY_RUN:END -->
