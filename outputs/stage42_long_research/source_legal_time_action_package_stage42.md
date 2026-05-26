# Stage42-DO Source Legal/Time Action Package

- source: `fresh_synthesis_from_stage42_cg_bn_dd_after_da1_rerun`
- generated_at_utc: `2026-05-26T23:28:27.142949+00:00`
- gate: `13 / 13`
- verdict: `stage42_do_source_legal_time_action_package_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DO 是 DA-1 source/legal/time closure action package，不训练模型、不下载数据、不转换数据。
- 本步骤复核 Stage42-CG terms validator 与 Stage42-BN time/geometry calibration，并生成明确 user_action_required。
- local path、parseability、H/FPS evidence 不等于 legal conversion permission。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；只有 source-specific calibrated subset 可以在未来单独声明。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets_checked: `5`
- conversion_ready_targets: `0`
- conversion_ready_ids: `[]`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- source_specific_metric_time_sources_count: `6`
- source_specific_metric_time_sources: `['ETH_seq_eth', 'ETH_seq_hotel', 'UCY_students03', 'UCY_zara01', 'UCY_zara02', 'UCY_zara03']`
- global_metric_seconds_claim_allowed: `False`
- global_t100_deployable_claim_allowed: `False`

## User Action Rows

| dataset | domain | conversion ready | source-specific metric/time candidates | official URL | required action |
| --- | --- | ---: | --- | --- | --- |
| `ucy_crowd_original` | `UCY` | `False` | `UCY_zara01, UCY_zara02, UCY_zara03, UCY_students03` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | accept/confirm official terms, allowed use, acceptance date, local path, and source identity |
| `eth_biwi_original` | `ETH_UCY` | `False` | `ETH_seq_eth, ETH_seq_hotel` | https://vision.ee.ethz.ch/datsets.html | accept/confirm official terms, allowed use, acceptance date, local path, and source identity |
| `trajnetplusplus_official` | `TrajNet` | `False` | `none` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | accept/confirm official terms, allowed use, acceptance date, local path, and source identity |
| `opentraj_toolkit` | `OpenTraj` | `False` | `none` | https://github.com/crowdbotp/OpenTraj | accept/confirm official terms, allowed use, acceptance date, local path, and source identity |
| `aerialmpt_or_other_topdown` | `other_topdown` | `False` | `none` | user_or_web_verified_official_url_required | accept/confirm official terms, allowed use, acceptance date, local path, and source identity |

## Claim Boundary

- Current source-specific calibration candidates do not permit global metric/seconds claims.
- No source is conversion-ready because terms/path/source-identity confirmation remains missing.
- No dataset is converted or evaluated by Stage42-DO.
- t+100 remains raw-frame diagnostic and not globally deployable.
- Stage5C remains unexecuted and SMC remains disabled.

## Gate

| gate | pass |
| --- | --- |
| `terms_validator_passed` | `True` |
| `time_geometry_passed` | `True` |
| `closure_audit_passed` | `True` |
| `zero_conversion_ready_recorded` | `True` |
| `global_metric_seconds_blocked` | `True` |
| `global_t100_blocked` | `True` |
| `user_action_rows_present` | `True` |
| `official_urls_present` | `True` |
| `source_specific_candidates_reported` | `True` |
| `no_conversion_claim` | `True` |
| `no_evaluation_claim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
