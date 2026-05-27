# Stage42-EF Source Terms Gap Audit

- source: `fresh_rerun_cg_plus_ed_source_terms_gap_audit`
- generated_at_utc: `2026-05-27T02:00:35.053179+00:00`
- git_commit: `cad81ce`
- input_hash: `d5067cf46f72dd51089824ce381160718544740c275763ed45ed6099345bd4d6`
- gate: `13 / 13`
- verdict: `stage42_ef_source_terms_gap_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EF reruns the source terms validator and merges it with Stage42-ED technical-after-terms potential.
- 本阶段不下载、不转换、不训练、不评估，只生成 legal/source/time blocker closure checklist。
- local path、parseability、technical dry-run 都不等于 legal conversion readiness。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets: `5`
- cg_terms_accepted_targets: `0`
- cg_conversion_ready_targets: `0`
- conversion_ready_now: `0`
- converted/evaluated now: `0` / `0`
- estimated_t50/t100_windows_after_terms: `10060` / `5696`
- top_unblock_targets: `['ucy_crowd_original', 'eth_biwi_original', 'aerialmpt_or_other_topdown']`

## Gap Table

| rank | dataset | domain | t50 after terms | t100 after terms | source-CV after terms | missing fields | blocker class |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `ucy_crowd_original` | `UCY` | 9554 | 5605 | True | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity | `local_path_and_terms_required` |
| 2 | `eth_biwi_original` | `ETH_UCY` | 506 | 91 | True | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity | `local_path_and_terms_required` |
| 3 | `aerialmpt_or_other_topdown` | `other_topdown` | 0 | 0 | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity | `new_official_source_required` |
| 4 | `opentraj_toolkit` | `OpenTraj` | 0 | 0 | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity | `toolkit_not_independent_source` |
| 5 | `trajnetplusplus_official` | `TrajNet` | 0 | 0 | False | terms_accepted_by_user, terms_acceptance_date, allowed_use, local_path, source_identity | `local_path_and_terms_required` |

## Interpretation

- UCY remains the first legal unblock target because it has the largest t50/t100 after-terms potential and source-CV support.
- ETH/BIWI is second because it has source-specific metric/time candidates but far fewer t50/t100 rows.
- TrajNet/OpenTraj/AerialMPT still need source identity, legal terms, or independent source repair before conversion claims.
- No conversion, no evaluation, no metric/seconds claim, no Stage5C, and no SMC are made by this stage.

## Gate

| gate | pass |
| --- | ---: |
| `cg_fresh_rerun_passed` | True |
| `ed_input_passed` | True |
| `all_targets_gap_scored` | True |
| `empty_template_still_blocks_conversion` | True |
| `ucy_priority_preserved` | True |
| `eth_priority_present` | True |
| `technical_potential_recorded` | True |
| `missing_fields_are_concrete` | True |
| `user_action_written` | True |
| `no_conversion_or_eval_claim` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
