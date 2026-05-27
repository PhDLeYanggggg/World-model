# Stage42-GQ Source Terms Package Claim Linter

- source: `fresh_stage42_gq_source_terms_package_claim_linter`
- generated_at_utc: `2026-05-27T13:40:06.814461+00:00`
- git_commit: `42b145e`
- input_hash: `1a55d0a3304d62c4784a7c8ea31818d40bb98c1adcd20a5880c43744672890f0`
- gate: `13 / 13`
- verdict: `stage42_gq_source_terms_package_claim_linter_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GQ 是 package-wide source/legal claim linter；不下载、不转换、不训练、不评估。
- OpenTraj toolkit MIT 许可不能写成 ETH/UCY/TrajNet/AerialMPT 底层数据许可。
- 用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。
- dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_gq_source_terms_package_claim_linter`
- go_source: `fresh_stage42_go_official_source_terms_live_verifier`
- go_verdict: `stage42_go_official_source_terms_live_verifier_pass`
- gp_source: `fresh_stage42_gp_source_terms_paper_claim_guard`
- gp_verdict: `stage42_gp_source_terms_paper_claim_guard_pass`
- files_scanned: `14`
- files_with_violations: `0`
- violation_count: `0`
- underlying_data_license_confirmed: `0`
- auto_download_allowed_now: `0`
- contract_ready_now: `0`
- gp_paper_files_refreshed: `['outputs/stage42_long_research/data_card_stage42.md', 'outputs/stage42_long_research/a_journal_gap_stage42.md', 'outputs/stage42_long_research/method_draft_stage42.md']`
- download_executed: `False`
- conversion_executed: `False`
- training_executed: `False`
- evaluation_executed: `False`
- next_required_action: `fix any future source/legal overclaim before treating paper package as claim-safe`

## Package Scan

| file | size | violations |
| --- | ---: | ---: |
| `README_RESULTS.md` | 330203 | 0 |
| `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md` | 180672 | 0 |
| `README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md` | 47787 | 0 |
| `outputs/stage42_long_research/paper_outline_stage42.md` | 47729 | 0 |
| `outputs/stage42_long_research/method_draft_stage42.md` | 53770 | 0 |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | 61829 | 0 |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | 69290 | 0 |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | 48823 | 0 |
| `outputs/stage42_long_research/model_card_stage42.md` | 59497 | 0 |
| `outputs/stage42_long_research/data_card_stage42.md` | 48056 | 0 |
| `outputs/stage42_long_research/reproducibility_stage42.md` | 53843 | 0 |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | 80774 | 0 |
| `outputs/stage42_long_research/paper_claim_evidence_audit_stage42.md` | 6845 | 0 |
| `outputs/stage42_long_research/paper_ready_evidence_matrix_stage42.md` | 4856 | 0 |

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `go_loaded` | True |
| `gp_loaded` | True |
| `go_gate_passed` | True |
| `gp_gate_passed` | True |
| `package_files_scanned` | True |
| `no_source_terms_claim_violations` | True |
| `no_license_or_auto_download_claim` | True |
| `no_download_conversion_training_eval` | True |
| `user_action_written` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
