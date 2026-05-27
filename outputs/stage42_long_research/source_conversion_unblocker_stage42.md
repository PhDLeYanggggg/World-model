# Stage42-ED Source Conversion Unblocker Package

- source: `fresh_synthesis_from_stage42_cg_dw_do_ds`
- generated_at_utc: `2026-05-27T01:37:52.651358+00:00`
- git_commit: `50714ae`
- input_hash: `60cd8fe27aee84e3a92410446840d74c168a557bcbf8706fa593062b90d22de1`
- gate: `15 / 15`
- verdict: `stage42_ed_source_conversion_unblocker_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-ED 是 source-conversion unblocker package，不下载、不转换、不训练、不评估。
- 本阶段把 CG/DW/DO/DS 的 legal/source/time blockers 汇总成可执行用户动作。
- local path、parseability、technical dry-run 都不等于 legal conversion readiness。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；metric/seconds-level claim 仍被阻塞。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_synthesis_from_stage42_cg_dw_do_ds`
- targets: `5`
- conversion_ready_now: `0`
- conversion_allowed_now: `0`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- raw_path_found_targets: `6`
- technical_ready_after_terms_targets: `2`
- estimated_t50_windows_after_terms: `10060`
- estimated_t100_windows_after_terms: `5696`
- domains_with_source_cv_after_terms: `['UCY']`
- source_specific_metric_time_sources: `['ETH_seq_eth', 'ETH_seq_hotel', 'UCY_students03', 'UCY_zara01', 'UCY_zara02', 'UCY_zara03']`
- terms_accepted_targets: `0`
- user_action_required_targets: `5`

## Unblocker Table

| dataset | domain | raw path | technical sources after terms | t50 after terms | t100 after terms | blocker class | official URL |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `ucy_crowd_original` | `UCY` | True | 3 | 9554 | 5605 | `local_path_and_terms_required` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data |
| `eth_biwi_original` | `ETH_UCY` | True | 2 | 506 | 91 | `local_path_and_terms_required` | https://vision.ee.ethz.ch/datsets.html |
| `aerialmpt_or_other_topdown` | `other_topdown` | True | 0 | 0 | 0 | `new_official_source_required` | user_or_web_verified_official_url_required |
| `opentraj_toolkit` | `OpenTraj` | True | 0 | 0 | 0 | `toolkit_not_independent_source` | https://github.com/crowdbotp/OpenTraj |
| `trajnetplusplus_official` | `TrajNet` | True | 0 | 0 | 0 | `local_path_and_terms_required` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ |

## Required Next Commands

1. Fill `outputs/stage42_long_research/source_terms_confirmation_template_stage42.json` with explicit user-confirmed official terms, allowed use, local path, and source identity.
2. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.
3. Only if conversion-ready targets become nonzero, run a future guarded conversion/no-leakage/source-CV stage. Stage42-ED does not convert.

## Interpretation

- UCY and ETH/BIWI are the first legal unblock targets because the dry-run found source-specific metric/time candidates after terms.
- OpenTraj remains useful as toolkit/reference evidence, but toolkit presence is not an independent source-rights claim.
- No metric/seconds-level or converted-data claim is allowed until terms/path/source identity are confirmed and a future conversion/audit runs.

## Gate

| gate | pass |
| --- | ---: |
| `cg_input_passed` | True |
| `dw_input_passed` | True |
| `do_input_passed` | True |
| `ds_input_passed` | True |
| `action_rows_written` | True |
| `ucy_priority_present` | True |
| `eth_priority_present` | True |
| `technical_ready_after_terms_recorded` | True |
| `t50_t100_after_terms_recorded` | True |
| `legal_blocker_preserved` | True |
| `no_conversion_or_eval_claim` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
