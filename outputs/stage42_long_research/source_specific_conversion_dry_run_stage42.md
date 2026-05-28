# Stage42-DW Source-Specific Calibrated Conversion Dry-Run

- source: `fresh_source_specific_conversion_dry_run_from_stage42_dv`
- generated_at_utc: `2026-05-28T04:09:12.766181+00:00`
- gate: `15 / 15`
- verdict: `stage42_dw_source_specific_conversion_dry_run_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DW 是 source-specific calibrated conversion dry-run，不转换数据、不训练、不评估。
- 本步骤只验证 UCY/ETH source-specific candidates 的技术可转换性、horizon support 与 source-CV readiness。
- terms/source/path/version 未确认前，source-specific metric/time subset 仍不能执行转换或声明 official result。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍不能写成全局 seconds-level；source-specific seconds 也必须等 legal conversion 后限定声明。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source_specific_sources_checked: `6`
- technical_conversion_ready_after_terms_sources: `5`
- technical_not_ready_sources: `['UCY_zara03']`
- t50_capable_sources: `5`
- t100_capable_sources: `4`
- estimated_t50_windows: `10060`
- estimated_t100_windows: `5696`
- domains_with_source_cv_after_terms: `['UCY']`
- conversion_allowed_now_sources: `0`
- full_world_state_rows_written: `0`
- evaluation_rows_written: `0`

## Source Dry-Run Table

| source | domain | rows | agents | t50 | t100 | step | gap ratio | ready after terms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_seq_eth` | `ETH_UCY` | `8908` | `360` | `291` | `91` | `6` | `0.0000` | `True` |
| `ETH_seq_hotel` | `ETH_UCY` | `6544` | `390` | `215` | `0` | `10` | `0.0000` | `True` |
| `UCY_students03` | `UCY` | `21846` | `428` | `6491` | `3413` | `10` | `0.0023` | `True` |
| `UCY_zara01` | `UCY` | `5024` | `148` | `240` | `97` | `10` | `0.0000` | `True` |
| `UCY_zara02` | `UCY` | `9537` | `204` | `2823` | `2095` | `10` | `0.0000` | `True` |
| `UCY_zara03` | `UCY` | `3600` | `180` | `0` | `0` | `10` | `0.0000` | `False` |

## Source-CV Plan

| domain | sources | t50 windows | t100 windows | source-CV feasible after terms |
| --- | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `2` | `506` | `91` | `False` |
| `UCY` | `3` | `9554` | `5605` | `True` |

## Boundary

- This dry-run did not write world-state rows, feature stores, checkpoints, or evaluation metrics.
- UCY appears technically source-CV feasible after terms confirmation; ETH/BIWI has two calibrated sources and remains weaker for source-CV by itself.
- UCY_zara03 is retained as source-specific evidence but marked technically not-ready for t50/t100 because this dry-run found no t50 windows.
- Conversion remains blocked until user confirms official terms/source/path/version and no-leakage conversion is executed.
- No global metric or seconds-level M3W claim is allowed.

## Gate

| gate | pass |
| --- | --- |
| `dv_input_passed` | `True` |
| `bn_input_passed` | `True` |
| `source_specific_candidates_loaded` | `True` |
| `technical_ready_after_terms_present` | `True` |
| `short_source_blocker_reported` | `True` |
| `t50_support_present` | `True` |
| `t100_support_present` | `True` |
| `source_cv_feasible_domain_present` | `True` |
| `legal_blocker_preserved` | `True` |
| `no_world_state_rows_written` | `True` |
| `no_evaluation_rows_written` | `True` |
| `no_leakage_preflight` | `True` |
| `global_metric_seconds_blocked` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
