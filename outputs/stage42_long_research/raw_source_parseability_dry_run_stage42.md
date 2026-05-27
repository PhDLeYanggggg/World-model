# Stage42-DT Raw Source Parseability Dry Run

- source: `fresh_sample_only_raw_source_parseability_dry_run`
- generated_at_utc: `2026-05-27T00:20:12.458762+00:00`
- gate: `11 / 11`
- verdict: `stage42_dt_raw_source_parseability_dry_run_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DT 是 raw source parseability dry-run：只做文件形态和少量样例行解析，不生成转换数据。
- 本步骤不下载、不解压 gated 数据、不训练、不评估、不生成 world-state rows。
- sample parseability 不等于 legal conversion permission，也不等于 official benchmark readiness。
- derived cache 不算 raw official source。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets_checked: `7`
- files_sampled_total: `319`
- dry_run_parseable_targets: `4`
- targets_with_homography_or_time_hints: `2`
- legal_conversion_ready_targets: `0`
- user_action_required_targets: `7`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- world_state_rows_generated: `0`
- archives_extracted_now: `0`

## Target Parseability

| dataset | domain | files sampled | trajectory-like | calibration-like | H hint | time hint | legal ready | next step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ucy_crowd_original` | `UCY` | `47` | `9` | `7` | `True` | `False` | `False` | after user terms/source confirmation, run no-leakage conversion plus source-specific time/geometry audit |
| `eth_biwi_original` | `ETH_UCY` | `30` | `9` | `4` | `True` | `True` | `False` | after user terms/source confirmation, run no-leakage conversion plus source-specific time/geometry audit |
| `trajnetplusplus_official` | `TrajNet` | `80` | `60` | `0` | `False` | `False` | `False` | after user terms/source confirmation, run no-leakage conversion as dataset-local raw-frame source |
| `opentraj_toolkit` | `OpenTraj` | `80` | `0` | `0` | `False` | `False` | `False` | locate raw trajectory files or official extraction instructions after terms confirmation |
| `aerialmpt_or_other_topdown` | `other_topdown` | `2` | `0` | `0` | `False` | `False` | `False` | locate raw trajectory files or official extraction instructions after terms confirmation |
| `stanford_drone_dataset` | `SDD` | `80` | `40` | `0` | `False` | `False` | `False` | already converted SDD reference; use only for SDD pixel raw-frame work |
| `tgsim_diagnostic` | `traffic_diagnostic` | `0` | `0` | `0` | `False` | `False` | `False` | diagnostic traffic source only; not pedestrian topdown official |

## Boundary

- This is a sample-only parser preflight. It does not create feature stores, world-state rows, episodes, or benchmarks.
- Legal/source blockers from Stage42-DS remain active; legal conversion ready remains zero.
- Homography/time hints are only hints; they do not authorize metric or seconds-level claims.
- Archives were not extracted.
- Stage5C and SMC remain disabled.

## Gate

| gate | pass |
| --- | --- |
| `stage42_ds_input_present` | `True` |
| `sample_only_no_conversion` | `True` |
| `sample_only_no_evaluation` | `True` |
| `parseable_sources_identified` | `True` |
| `homography_or_time_hints_reported` | `True` |
| `legal_readiness_not_overclaimed` | `True` |
| `archives_not_extracted` | `True` |
| `user_action_present` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
