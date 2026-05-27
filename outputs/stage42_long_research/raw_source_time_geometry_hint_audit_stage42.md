# Stage42-DU Raw Source Time/Geometry Hint Audit

- source: `fresh_hint_audit_from_local_raw_sources_after_stage42_dt`
- generated_at_utc: `2026-05-27T00:27:19.408256+00:00`
- gate: `14 / 14`
- verdict: `stage42_du_raw_source_time_geometry_hint_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DU 是 raw source time/geometry hint audit：只抽取 H/FPS/stride hints，不转换轨迹。
- H matrix、FPS、frame stride 线索不是 metric/seconds-level claim。
- legal/source blocker 未关闭时，不能把 hints 写成 official metric conversion。
- 本步骤不下载、不解压 gated 数据、不训练、不评估、不生成 world-state rows。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets_checked: `7`
- files_scanned_total: `585`
- targets_with_h_matrix_hints: `2`
- targets_with_time_hints: `3`
- targets_with_frame_stride_hints: `4`
- metric_time_subset_hint_targets: `2`
- legal_conversion_ready_targets: `0`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- world_state_rows_generated: `0`

## Dataset Hint Table

| dataset | domain | files | H hints | time hints | stride hints | metric/time subset hint | claim allowed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ucy_crowd_original` | `UCY` | `47` | `7` | `1` | `9` | `True` | `False` |
| `eth_biwi_original` | `ETH_UCY` | `30` | `2` | `2` | `4` | `True` | `False` |
| `trajnetplusplus_official` | `TrajNet` | `146` | `0` | `1` | `97` | `False` | `False` |
| `opentraj_toolkit` | `OpenTraj` | `180` | `0` | `0` | `0` | `False` | `False` |
| `aerialmpt_or_other_topdown` | `other_topdown` | `2` | `0` | `0` | `0` | `False` | `False` |
| `stanford_drone_dataset` | `SDD` | `180` | `0` | `0` | `57` | `False` | `False` |
| `tgsim_diagnostic` | `traffic_diagnostic` | `0` | `0` | `0` | `0` | `False` | `False` |

## Boundary

- H/FPS/stride hints were extracted from local files, but no conversion was performed.
- These hints can guide a future no-leakage conversion only after terms/source confirmation.
- No global metric or seconds-level claim is allowed.
- Stage5C and SMC remain disabled.

## Gate

| gate | pass |
| --- | --- |
| `stage42_ds_input_present` | `True` |
| `stage42_dt_input_present` | `True` |
| `hints_audited` | `True` |
| `h_matrix_hints_found` | `True` |
| `time_hints_found` | `True` |
| `frame_stride_hints_found` | `True` |
| `metric_time_subset_hints_separated` | `True` |
| `legal_readiness_not_overclaimed` | `True` |
| `no_conversion_or_rows` | `True` |
| `no_evaluation_claim` | `True` |
| `data_calibration_addendum_written` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
