# Stage42-CY Worktree Caveat Classifier

- source: `fresh_worktree_caveat_classification`
- generated_at_utc: `2026-05-26T20:14:51.627397+00:00`
- git_commit: `1d2da72`
- gate: `11 / 11`
- verdict: `stage42_cy_worktree_caveat_classifier_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CY 是 worktree caveat classifier，不重新训练，不调 threshold。
- 本阶段只分类 tracked dirty files，不提交 raw data/cache/checkpoint/video/第三方数据。
- metadata-only diff 不等于新模型结果；paper-size-only diff 不等于新实验结果。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- tracked_dirty_files: `21`
- stage42_dirty_files: `9`
- stage42_substantive_dirty_files: `0`
- classification_counts: `{'substantive_markdown_change': 3, 'substantive_json_change': 9, 'metadata_only': 6, 'metadata_and_paper_size_only': 2, 'append_only_run_ledger': 1}`

## Dirty File Classification

| path | scope | status | classification | allowed |
| --- | --- | --- | --- | ---: |
| `README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md` | `outside_stage42_scope` | ` M` | `substantive_markdown_change` | `True` |
| `README_RESULTS.md` | `outside_stage42_scope` | ` M` | `substantive_markdown_change` | `True` |
| `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md` | `outside_stage42_scope` | ` M` | `substantive_markdown_change` | `True` |
| `outputs/reports/stage17_baseline_oracle_metrics.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage17_baseline_oracle_report.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage17_baseline_selector_report.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage18_jepa_training_report.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage18_multimodal_data_report.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage19_jepa_probe_report.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage19_jepa_training_report.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/reports/stage19_wam_data_registry.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |
| `outputs/stage42_long_research/context_contribution_forensics_stage42.json` | `stage42` | ` M` | `metadata_only` | `True` |
| `outputs/stage42_long_research/context_contribution_forensics_stage42.md` | `stage42` | ` M` | `metadata_only` | `True` |
| `outputs/stage42_long_research/paper_claim_evidence_audit_stage42.json` | `stage42` | ` M` | `metadata_and_paper_size_only` | `True` |
| `outputs/stage42_long_research/paper_claim_evidence_audit_stage42.md` | `stage42` | ` M` | `metadata_and_paper_size_only` | `True` |
| `outputs/stage42_long_research/retrained_ablation_matrix_stage42.json` | `stage42` | ` M` | `metadata_only` | `True` |
| `outputs/stage42_long_research/retrained_ablation_matrix_stage42.md` | `stage42` | ` M` | `metadata_only` | `True` |
| `outputs/stage42_long_research/run_ledger.jsonl` | `stage42` | ` M` | `append_only_run_ledger` | `True` |
| `outputs/stage42_long_research/source_time_geometry_calibration_stage42.json` | `stage42` | ` M` | `metadata_only` | `True` |
| `outputs/stage42_long_research/source_time_geometry_calibration_stage42.md` | `stage42` | ` M` | `metadata_only` | `True` |
| `research_state.json` | `outside_stage42_scope` | ` M` | `substantive_json_change` | `True` |

## Interpretation

- Stage42-CY does not resolve old dirty files by pretending they are clean.
- It proves the current Stage42 dirty tracked diffs are metadata/hash/paper-size/append-only ledger caveats rather than substantive metric changes.
- Historical non-Stage42 report drift remains outside the Stage42 paper-freeze scope and should not be cited as new Stage42 evidence.
