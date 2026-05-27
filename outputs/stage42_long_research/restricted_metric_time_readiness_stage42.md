# Stage42-HI Restricted Metric/Time Readiness

- source: `fresh_stage42_hi_restricted_metric_time_readiness`
- generated_at_utc: `2026-05-27T17:22:48.879726+00:00`
- git_commit: `d42b4eb`
- input_hash: `2751531223effb6aa8709c4a7faf448ea72e61a2014fb113273f4272d3e6a5fc`
- gate: `14 / 14`
- verdict: `stage42_hi_restricted_metric_time_readiness_pass_blocked_by_terms`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HI 是 restricted metric/time readiness recheck，不训练、不转换、不下载、不调 threshold。
- ETH/UCY 源级 H/FPS/stride 线索存在，但 legal/source terms readiness 仍是独立前置条件。
- restricted metric/time subset 只有在 user terms/path/source identity 确认、conversion、no-leakage、source-CV、final test 后才能写。
- SDD 仍是 pixel raw-frame；TrajNet snippets 仍无可用 homography/FPS/scale；TGSIM 只可 traffic diagnostic。
- future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- restricted_metric_time_candidate_count: `6`
- candidate_domains: `['ETH_UCY', 'UCY']`
- technical_ready_after_terms_count: `6`
- restricted_metric_time_ready_now_count: `0`
- paper_claim_allowed_now: `False`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`

## Candidate Rows

| source | domain | terms target | H | fps | h50 seconds if restricted | h100 seconds if restricted | after terms | ready now | blockers |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_seq_eth` | `ETH_UCY` | `eth_biwi_original` | True | 2.5 | 20.0 | 40.0 | True | False | source_terms_or_source_cv_not_conversion_ready |
| `ETH_seq_hotel` | `ETH_UCY` | `eth_biwi_original` | True | 2.5 | 20.0 | 40.0 | True | False | source_terms_or_source_cv_not_conversion_ready |
| `UCY_zara01` | `UCY` | `ucy_crowd_original` | True | 2.5 | 20.0 | 40.0 | True | False | source_terms_or_source_cv_not_conversion_ready |
| `UCY_zara02` | `UCY` | `ucy_crowd_original` | True | 2.5 | 20.0 | 40.0 | True | False | source_terms_or_source_cv_not_conversion_ready |
| `UCY_zara03` | `UCY` | `ucy_crowd_original` | True | 2.5 | 20.0 | 40.0 | True | False | source_terms_or_source_cv_not_conversion_ready |
| `UCY_students03` | `UCY` | `ucy_crowd_original` | True | 2.5 | 20.0 | 40.0 | True | False | source_terms_or_source_cv_not_conversion_ready |

## Interpretation

- ETH/UCY has restricted source-level metric/time candidates: H is parseable and annotation timing is available for selected sources.
- The current claim is still blocked because source terms/path/source identity confirmation has zero conversion-ready targets.
- No conversion, training, evaluation, Stage5C, or SMC execution occurred in this stage.
- Current paper wording remains dataset-local/raw-frame 2.5D. A future restricted metric/time claim must run conversion, no-leakage, source-CV, and final test after user-confirmed source terms.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `bn_input_passed` | True |
| `ch_input_passed` | True |
| `cg_input_passed` | True |
| `restricted_candidates_identified` | True |
| `eth_and_ucy_candidates_present` | True |
| `technical_after_terms_candidates_present` | True |
| `ready_now_zero` | True |
| `paper_claim_blocked_now` | True |
| `global_metric_seconds_blocked` | True |
| `user_action_written` | True |
| `no_training_or_conversion` | True |
| `stage5c_false` | True |
| `smc_false` | True |
