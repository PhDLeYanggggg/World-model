# Stage42-HK ETH_UCY Restricted Metric/Time Source-Support Preflight

- source: `fresh_stage42_hk_restricted_metric_time_eth_ucy_source_support_preflight`
- generated_at_utc: `2026-05-27T17:50:07.726188+00:00`
- git_commit: `3b6bc53`
- input_hash: `e59b95ba9a29a7089186a5c238fcadd8daa37c3ae163f3f3cf6ea8c38afdeba1`
- gate: `16 / 16`
- verdict: `stage42_hk_eth_ucy_source_support_preflight_pass_terms_blocked`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HK 是 ETH_UCY restricted metric/time source-support augmentation preflight，不训练、不转换、不下载、不调 threshold。
- 本阶段合并 Stage42-HJ 的 ETH_UCY blocker 与 Stage42-BL 的 ETH-Person XML technical dry-run。
- ETH-Person XML local files 仍是 terms-unverified；technical source support 不等于 official converted/evaluated data。
- future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- restricted seconds/metric wording 仍需 user terms confirmation、guarded conversion、no-leakage、source-CV、final test。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- HJ ETH_UCY blocker detected: `True`
- ETH-Person XML candidate sources: `4`
- augmented ETH_UCY independent sources after terms: `5`
- augmented ETH_UCY source-CV feasible after terms: `True`
- augmented ETH_UCY robust source-CV feasible after terms: `True`
- augmented t50/t100 windows after terms: `4397` / `1433`
- cached BL technical t100 safe-positive: `True`
- cached BL technical t100 mean improvement vs fallback: `0.6835494566410742`
- conversion ready targets now: `0`

## Augmented ETH_UCY Sources

| source | path | rows | agents | max track | t50 | t100 | ready now |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH-Person_bahnhof_assc_gt` | `ETH-Person/data/bahnhof_assc_gt.xml` | 7653 | 223 | 217 | 1897 | 348 | False |
| `ETH-Person_jelmoli_assc_gt` | `ETH-Person/data/jelmoli_assc_gt.xml` | 2582 | 74 | 161 | 561 | 126 | False |
| `ETH-Person_seq0_assc_gt` | `ETH-Person/data/seq0_assc_gt-interp.xml` | 2573 | 46 | 353 | 1040 | 465 | False |
| `ETH-Person_sunnyday_assc_gt` | `ETH-Person/data/sunnyday_assc_gt.xml` | 1898 | 36 | 305 | 626 | 406 | False |
| `ETH_seq_eth` | `ETH/seq_eth/obsmat.txt` | 8908 | 360 | 190 | 273 | 88 | False |

## Interpretation

- HJ correctly records ETH_UCY as blocked under the original HI source list because ETH_seq_hotel has no t100 windows.
- HK shows the local ETH-Person XML technical dry-run can supply enough ETH_UCY independent t100-capable sources after terms.
- This narrows the ETH_UCY blocker from raw source support to user-confirmed terms plus guarded conversion/evaluation.
- No conversion, no official evaluation, no model training, no metric/seconds claim, no Stage5C, and no SMC occurred.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `hj_input_passed` | True |
| `bl_input_passed` | True |
| `cg_input_passed` | True |
| `hj_eth_ucy_blocker_detected` | True |
| `eth_person_xml_candidates_present` | True |
| `augmented_sources_enough` | True |
| `augmented_t50_t100_windows_present` | True |
| `augmented_eth_ucy_source_cv_feasible_after_terms` | True |
| `augmented_eth_ucy_robust_source_cv_feasible_after_terms` | True |
| `cached_bl_technical_support_positive_recorded` | True |
| `terms_still_block_conversion_now` | True |
| `no_conversion_or_evaluation_claim` | True |
| `global_metric_seconds_blocked` | True |
| `stage5c_false` | True |
| `smc_false` | True |
