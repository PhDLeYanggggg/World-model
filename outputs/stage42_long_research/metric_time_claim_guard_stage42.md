# Stage42-CH Metric/Time Claim Guard

- source: `fresh_stage42_ch_metric_time_claim_guard`
- generated_at_utc: `2026-05-26T17:22:34.180339+00:00`
- git_commit: `85c237d`
- input_hash: `b4ab1735f23fa865a4eb689aaba450a36a6d9132e8ed2a30b3b2cfd01d63e2d6`
- gate: `11 / 11`
- verdict: `stage42_ch_metric_time_claim_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CH 是 metric/time claim guard，不下载、不转换、不训练、不评估。
- source-specific calibration evidence 不等于 global M3W metric/seconds claim。
- legal/source terms readiness 和 metric/time calibration 是两个独立前置条件。
- t+50 / t+100 默认仍是 raw-frame horizon，不能写成 seconds-level。
- SDD 当前仍是 pixel raw-frame；estimated scale 只能诊断，不能作为 official metric claim。
- TGSIM 是 traffic diagnostic，不能包装成 pedestrian top-down world-model success。
- Stage5C 未执行，SMC 未启用。

## Summary

- datasets_audited: `7`
- source_records_audited: `7`
- source_specific_metric_time_candidates: `6`
- conversion_ready_targets: `0`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`
- restricted_subset_metric_seconds_claim_allowed_now: `False`

## Source Claim Guard

| source | domain | source metric/time evidence | h50 seconds if restricted | h100 seconds if restricted | paper claim allowed now | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ETH_seq_eth` | `ETH_UCY` | True | 20.0 | 40.0 | False | calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets |
| `ETH_seq_hotel` | `ETH_UCY` | True | 20.0 | 40.0 | False | calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets |
| `UCY_zara01` | `UCY` | True | 20.0 | 40.0 | False | calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets |
| `UCY_zara02` | `UCY` | True | 20.0 | 40.0 | False | calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets |
| `UCY_zara03` | `UCY` | True | 20.0 | 40.0 | False | calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets |
| `UCY_students03` | `UCY` | True | 20.0 | 40.0 | False | calibration candidate exists, but source terms/readiness validator has zero conversion-ready targets |
| `UCY_students01` | `UCY` | False | 20.0 | 40.0 | False | source-specific metric/time evidence incomplete |

## Interpretation

- Source-specific ETH/UCY calibration candidates exist, but they are not yet paper-allowed metric/seconds evaluation claims.
- The current legal/readiness validator has zero conversion-ready targets, so restricted subset metric/time claims stay blocked.
- Global M3W, SDD, TrajNet-snippet, and TGSIM-pedestrian-world-model metric/seconds claims remain forbidden.
