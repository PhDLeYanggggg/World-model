# Stage42-GO Official Source / Terms Live Verifier

- source: `fresh_stage42_go_official_source_terms_live_verifier`
- generated_at_utc: `2026-05-27T13:29:25.186550+00:00`
- git_commit: `9cc1ddc`
- input_hash: `91b8e7f074a6362ccd3425d3ec393143a71232bc7f6374468977976fac4fef0a`
- gate: `14 / 14`
- verdict: `stage42_go_official_source_terms_live_verifier_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GO 只记录官方 source / terms live audit；不下载、不转换、不训练、不评估。
- OpenTraj toolkit license 不能替代 ETH/UCY/TrajNet/AerialMPT 底层数据授权。
- 用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。
- dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_go_official_source_terms_live_verifier`
- datasets_audited: `5`
- official_sources_reachable: `3`
- underlying_data_license_confirmed: `0`
- auto_download_allowed_now: `0`
- contract_ready_now: `0`
- top_priority_dataset: `ucy_crowd_original`
- top_priority_terms_status: `not_verified_by_agent`
- total_t50_after_terms: `10060`
- total_t100_after_terms: `5696`
- download_executed: `False`
- conversion_executed: `False`
- training_executed: `False`
- evaluation_executed: `False`
- next_required_action: `user confirms official terms/path/source identity; no source can be converted automatically yet`

## Official Source / Terms Audit

| priority | dataset | live source status | terms status | auto download | t50/t100 after terms | notes |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | `ucy_crowd_original` | `official_url_known_but_page_unavailable_in_live_audit` | `not_verified_by_agent` | False | 9554 / 5605 | UCY crowd data official URL is the known prior-audit URL, but live page retrieval failed; do not auto-download or treat local files as legally confirmed. |
| 2 | `eth_biwi_original` | `official_page_reachable_with_dataset_download_links` | `not_verified_by_agent` | False | 506 / 91 | ETH Vision dataset page is reachable and lists BIWI Walking Pedestrians material/download links, but live audit did not establish full redistribution/derived-data terms. |
| 3 | `opentraj_toolkit` | `official_github_reachable_toolkit_license_only` | `toolkit_mit_not_underlying_dataset_terms` | False | 0 / 0 | OpenTraj GitHub is official for toolkit/code and metadata, but its MIT license should not be treated as permission for every redistributed underlying trajectory dataset. |
| 4 | `trajnetplusplus_official` | `official_epfl_page_reachable_platform_access_via_aicrowd` | `not_verified_by_agent` | False | 0 / 0 | EPFL TrajNet++ page is reachable and points to benchmark/platform access; data/platform terms must be accepted by the user and current local snippets remain h100-limited. |
| 5 | `aerialmpt_or_other_topdown` | `not_verified` | `not_verified_by_agent` | False | 0 / 0 | AerialMPT or other top-down sources need an official URL and terms path before any guarded conversion. |

## Interpretation

- UCY and ETH/BIWI remain the highest-value confirmation targets, but neither can be converted until the user confirms terms/path/source identity.
- TrajNet++ official/platform access still requires manual terms confirmation and current local snippets do not solve h100.
- OpenTraj is useful as toolkit/metadata evidence, but its MIT license is not treated as underlying dataset permission.
- No source is auto-downloadable or conversion-ready in this audit.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `priority_board_loaded` | True |
| `contract_loaded` | True |
| `top_sources_audited` | True |
| `official_sources_recorded` | True |
| `terms_not_agent_accepted` | True |
| `no_auto_download_allowed` | True |
| `user_actions_recorded` | True |
| `opportunity_preserved` | True |
| `no_download_conversion_training_eval` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
