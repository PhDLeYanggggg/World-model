# Stage42-HJ Restricted Metric/Time Source-CV Preflight

- source: `fresh_stage42_hj_restricted_metric_time_source_cv_preflight`
- generated_at_utc: `2026-05-27T17:40:03.030933+00:00`
- git_commit: `fa87e65`
- input_hash: `183cfe1046c959c0eb2b54d6e4ff6f097af385e817811e0ccfc7f0b203985352`
- gate: `15 / 15`
- verdict: `stage42_hj_restricted_metric_time_source_cv_preflight_pass_with_eth_ucy_source_cv_limit`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HJ 是 restricted metric/time source-CV preflight，不训练、不转换、不下载、不调 threshold。
- 本阶段只解析本地 ETH/UCY technical candidate rows 来估计 source-CV / history / horizon 可行性。
- local parseability 和 source-CV feasibility 不等于 legal conversion readiness。
- future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- restricted seconds/metric wording 仍需 user terms confirmation、guarded conversion、no-leakage、source-CV、final test。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- candidate_sources: `6`
- usable_after_terms_sources: `4`
- restricted_metric_time_ready_now_sources: `0`
- domains_source_cv_feasible_after_terms: `['UCY']`
- domains_robust_source_cv_feasible_after_terms: `['UCY']`
- domains_source_cv_blocked_after_terms: `['ETH_UCY']`
- total_t50_windows_after_terms: `9845`
- total_t100_windows_after_terms: `5696`

## Source Rows

| source | domain | rows | agents | max track | t50 | t100 | k64+h100 | usable after terms | ready now |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_seq_eth` | `ETH_UCY` | 8908 | 360 | 190 | 291 | 91 | 27 | True | False |
| `ETH_seq_hotel` | `ETH_UCY` | 6544 | 390 | 100 | 215 | 0 | 0 | False | False |
| `UCY_zara01` | `UCY` | 5024 | 148 | 197 | 240 | 97 | 34 | True | False |
| `UCY_zara02` | `UCY` | 9537 | 204 | 583 | 2823 | 2095 | 1637 | True | False |
| `UCY_zara03` | `UCY` | 3600 | 180 | 20 | 0 | 0 | 0 | False | False |
| `UCY_students03` | `UCY` | 21846 | 428 | 539 | 6491 | 3413 | 1660 | True | False |

## Source-CV Plan

| domain | sources | usable after terms | blocked | robust after terms | folds | t50 | t100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 2 | 1 | 1 | False | 0 | 291 | 91 |
| `UCY` | 4 | 3 | 1 | True | 3 | 9554 | 5605 |

## Interpretation

- This preflight shows that UCY can support robust restricted source-CV after user-confirmed terms and guarded conversion.
- UCY has at least three usable sources after terms, so it is robust enough for leave-one-source style source-CV planning.
- ETH_UCY is parseable and has technical metric/time signals, but current local ETH_seq_hotel lacks t100 windows, so ETH_UCY source-CV is not feasible yet.
- No conversion, no model training, no evaluation, no Stage5C, and no SMC occurred.
- Current paper wording must remain dataset-local/raw-frame until the guarded conversion and source-CV final test are actually run.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `hi_input_passed` | True |
| `bn_input_passed` | True |
| `cg_input_passed` | True |
| `sources_parsed` | True |
| `usable_after_terms_sources_present` | True |
| `eth_ucy_preflight_complete_or_blocker_recorded` | True |
| `ucy_source_cv_feasible_after_terms` | True |
| `ucy_robust_source_cv_feasible_after_terms` | True |
| `h50_h100_windows_present` | True |
| `ready_now_zero` | True |
| `no_conversion_or_evaluation_claim` | True |
| `global_metric_seconds_blocked` | True |
| `stage5c_false` | True |
| `smc_false` | True |
