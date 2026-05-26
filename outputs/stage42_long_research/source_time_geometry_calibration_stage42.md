# Stage42-BN Strict Source Time/Geometry Calibration Audit

- source: `fresh_source_time_geometry_calibration_audit`
- generated_at_utc: `2026-05-26T13:56:53.923385+00:00`
- git_commit: `05ab286`
- input_hash: `b03c732f021417561011d8da11cef983af722be1c61b827cb8f27e884b661320`
- gate: `13 / 13`
- verdict: `stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BN 是 strict source-level time/geometry calibration audit，不训练模型，不下载数据。
- 本步骤区分 source-specific calibration evidence 与全局 metric/seconds claim。
- 即使某些 ETH/UCY source 有 meters / FPS / homography evidence，也不能把整个 M3W 写成 metric 或 seconds-level。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source_records_audited: `7`
- eth_source_specific_metric_time_sources: `2`
- ucy_source_specific_metric_time_sources: `4`
- source_specific_metric_time_sources: `['ETH_seq_eth', 'ETH_seq_hotel', 'UCY_zara01', 'UCY_zara02', 'UCY_zara03', 'UCY_students03']`
- sdd_scale_count: `60`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`
- m3w_official_metric_seconds_claim_allowed: `False`

## Source-Level Evidence

| source | domain | H parseable | annotation fps | timestep s | local claim | global metric/seconds |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `ETH_seq_eth` | `ETH_UCY` | True | 2.5 | 0.4 | `source_specific_annotation_step_meter_coordinate_evidence` | `False/False` |
| `ETH_seq_hotel` | `ETH_UCY` | True | 2.5 | 0.4 | `source_specific_annotation_step_meter_coordinate_evidence` | `False/False` |
| `UCY_zara01` | `UCY` | True | 2.5 | 0.4 | `source_specific_annotation_step_meter_coordinate_evidence` | `False/False` |
| `UCY_zara02` | `UCY` | True | 2.5 | 0.4 | `source_specific_annotation_step_meter_coordinate_evidence` | `False/False` |
| `UCY_zara03` | `UCY` | True | 2.5 | 0.4 | `source_specific_annotation_step_meter_coordinate_evidence` | `False/False` |
| `UCY_students03` | `UCY` | True | 2.5 | 0.4 | `source_specific_annotation_step_meter_coordinate_evidence` | `False/False` |
| `UCY_students01` | `UCY` | False | 2.5 | 0.4 | `source_specific_time_evidence_only_coordinate_not_verified` | `False/False` |

## SDD Scale Audit

- scale_count: `60`
- certainty_min: `0.0`
- certainty_max: `1.0`
- estimated_scale_warning_present: `True`
- metric_claim_allowed: `False`
- seconds_claim_allowed: `False`

## Diagnostic / Blocked Sources

| source | claim | reason |
| --- | --- | --- |
| `TrajNet_local_snippets` | `dataset_local_short_snippet_only` | Local TrajNet files are fixed short snippets and have no verified homography/FPS/scale evidence. |
| `TGSIM` | `traffic_metric_diagnostic_only` | TGSIM can be metric traffic diagnostic only, not pedestrian top-down world-model success. |
| `AerialMPT` | `dataset_local_raw_frame_only_until_source_terms_geometry_verified` | Local Stage42 calibration evidence did not verify source-specific homography/FPS/scale enough for metric/seconds claims. |

## Interpretation

- ETH `seq_eth` and `seq_hotel` have source-specific local evidence for meter coordinates and 2.5fps / 0.4s annotation steps.
- UCY `zara01`, `zara02`, `zara03`, and `students03` have source-specific local evidence for H.txt-backed world-2D candidate coordinates and 2.5fps / 0.4s annotation steps.
- These are source-specific calibration candidates, not a global M3W metric/seconds claim.
- Current M3W reports must still use raw-frame / dataset-local wording unless a downstream evaluation explicitly restricts itself to a verified source-specific calibrated subset.
- SDD remains pixel raw-frame in this project because its scales are estimated and the global Stage42 claim does not validate metric/seconds semantics.
