# Stage42-DV Calibration Candidate Manifest

- source: `fresh_synthesis_from_stage42_du_bn`
- generated_at_utc: `2026-05-27T00:33:47.940416+00:00`
- gate: `13 / 13`
- verdict: `stage42_dv_calibration_candidate_manifest_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DV 是 calibration candidate manifest：合并 raw H/FPS/stride hints 与 source-specific calibration evidence。
- 本步骤不转换数据、不训练、不评估，只给出下一步 terms/source/time/geometry closure 优先级。
- H/FPS/stride/source-specific evidence 不能写成全局 metric 或 seconds-level claim。
- source-specific calibrated subset 也必须等 legal terms、source identity、path/version 和 no-leakage conversion 完成后才能声明。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets_checked: `7`
- source_specific_candidate_targets: `2`
- time_stride_candidate_targets: `1`
- stride_only_candidate_targets: `0`
- conversion_ready_targets: `0`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- recommended_first_targets: `['ucy_crowd_original', 'eth_biwi_original']`

## Candidate Table

| dataset | domain | class | priority | H | time | stride | source candidates | conversion allowed |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `ucy_crowd_original` | `UCY` | `source_specific_metric_time_candidate_after_terms` | `95` | `7` | `1` | `9` | UCY_students03, UCY_zara01, UCY_zara02, UCY_zara03 | `False` |
| `eth_biwi_original` | `ETH_UCY` | `source_specific_metric_time_candidate_after_terms` | `95` | `2` | `2` | `4` | ETH_seq_eth, ETH_seq_hotel | `False` |
| `trajnetplusplus_official` | `TrajNet` | `time_stride_candidate_dataset_local_only` | `55` | `0` | `1` | `97` | none | `False` |
| `stanford_drone_dataset` | `SDD` | `reference_only_not_new_external` | `20` | `0` | `0` | `57` | none | `False` |
| `opentraj_toolkit` | `OpenTraj` | `not_calibration_candidate_now` | `10` | `0` | `0` | `0` | none | `False` |
| `aerialmpt_or_other_topdown` | `other_topdown` | `not_calibration_candidate_now` | `10` | `0` | `0` | `0` | none | `False` |
| `tgsim_diagnostic` | `traffic_diagnostic` | `diagnostic_only_missing_or_traffic` | `0` | `0` | `0` | `0` | none | `False` |

## Interpretation

- UCY and ETH/BIWI are the highest-priority source-specific calibration candidates, but terms/source/path/version confirmation is still required.
- TrajNet currently remains time/stride or dataset-local unless official coordinate and split semantics are confirmed.
- SDD is a reference pixel raw-frame source here, not a new external calibration source.
- No row in this manifest authorizes conversion, evaluation, metric claims, or seconds-level claims by itself.

## Gate

| gate | pass |
| --- | --- |
| `du_input_present` | `True` |
| `bn_input_present` | `True` |
| `candidate_manifest_written` | `True` |
| `source_specific_candidates_ranked` | `True` |
| `time_stride_candidates_ranked` | `True` |
| `legal_blockers_preserved` | `True` |
| `no_conversion_claim` | `True` |
| `no_evaluation_claim` | `True` |
| `global_metric_seconds_blocked` | `True` |
| `user_action_required_written` | `True` |
| `data_calibration_updated` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
