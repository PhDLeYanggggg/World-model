# Stage42-GP Source Terms Paper Claim Guard

- source: `fresh_stage42_gp_source_terms_paper_claim_guard`
- generated_at_utc: `2026-05-27T13:34:47.346933+00:00`
- git_commit: `c9e7b5d`
- input_hash: `a93a0aae6133b19053f7ebf23007389578fc43bf88274578939c768657095aec`
- gate: `12 / 12`
- verdict: `stage42_gp_source_terms_paper_claim_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GP 将 GO 官方 source/terms live audit 写入 paper package claim guard；不下载、不转换、不训练、不评估。
- OpenTraj toolkit MIT 许可不能写成 ETH/UCY/TrajNet/AerialMPT 底层数据许可。
- 用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。
- dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_gp_source_terms_paper_claim_guard`
- go_source: `fresh_stage42_go_official_source_terms_live_verifier`
- go_verdict: `stage42_go_official_source_terms_live_verifier_pass`
- datasets_guarded: `5`
- underlying_data_license_confirmed: `0`
- auto_download_allowed_now: `0`
- contract_ready_now: `0`
- total_t50_after_terms: `10060`
- total_t100_after_terms: `5696`
- paper_files_refreshed: `['outputs/stage42_long_research/data_card_stage42.md', 'outputs/stage42_long_research/a_journal_gap_stage42.md', 'outputs/stage42_long_research/method_draft_stage42.md']`
- download_executed: `False`
- conversion_executed: `False`
- training_executed: `False`
- evaluation_executed: `False`
- next_required_action: `paper claims must keep source/legal blocker wording until user confirmation and guarded conversion pass`

## Guard Rows

| dataset | paper claim status | allowed wording | disallowed wording |
| --- | --- | --- | --- |
| `ucy_crowd_original` | `blocked_until_user_terms_path_source_confirmation` | ucy_crowd_original is a post-confirmation source candidate with official/source terms still requiring user confirmation; it is not counted as converted or evaluated data. | Do not write that ucy_crowd_original has been legally converted, evaluated, auto-downloaded, or metric/seconds-calibrated. |
| `eth_biwi_original` | `blocked_until_user_terms_path_source_confirmation` | eth_biwi_original is a post-confirmation source candidate with official/source terms still requiring user confirmation; it is not counted as converted or evaluated data. | Do not write that eth_biwi_original has been legally converted, evaluated, auto-downloaded, or metric/seconds-calibrated. |
| `opentraj_toolkit` | `blocked_until_user_terms_path_source_confirmation` | OpenTraj is used only as toolkit/metadata/source-discovery evidence; underlying dataset terms are separate. | Do not write that OpenTraj MIT license grants permission for ETH/UCY/TrajNet underlying data. |
| `trajnetplusplus_official` | `blocked_until_user_terms_path_source_confirmation` | trajnetplusplus_official is a post-confirmation source candidate with official/source terms still requiring user confirmation; it is not counted as converted or evaluated data. | Do not write that trajnetplusplus_official has been legally converted, evaluated, auto-downloaded, or metric/seconds-calibrated. |
| `aerialmpt_or_other_topdown` | `blocked_until_user_terms_path_source_confirmation` | aerialmpt_or_other_topdown is a post-confirmation source candidate with official/source terms still requiring user confirmation; it is not counted as converted or evaluated data. | Do not write that aerialmpt_or_other_topdown has been legally converted, evaluated, auto-downloaded, or metric/seconds-calibrated. |

## Claim Scan

- unsafe source claim violations: `0`

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `go_loaded` | True |
| `go_gate_passed` | True |
| `paper_rows_built` | True |
| `paper_files_refreshed` | True |
| `no_unsafe_source_claims` | True |
| `no_license_or_auto_download_claim` | True |
| `no_download_conversion_training_eval` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
