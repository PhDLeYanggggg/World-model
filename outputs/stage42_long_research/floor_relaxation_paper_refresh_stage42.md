# Stage42-GU Floor Relaxation Paper Refresh

- source: `fresh_stage42_gu_floor_relaxation_paper_refresh`
- generated_at_utc: `2026-05-27T14:37:46.917324+00:00`
- git_commit: `5784563`
- input_hash: `6d3204e2de82e34d8dd16afac428e9a9a5a09f79aaeb1426afc3a62b0326be46`
- gate: `13 / 13`
- verdict: `stage42_gu_floor_relaxation_paper_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GU 是 paper package refresh 与 floor-relaxation claim linter；不训练、不下载、不转换、不调 threshold。
- Stage42-GT 只支持 validation-backed t50 partial floor relaxation 的 all-agent safety stress evidence。
- Global floor removal、floor-free neural deployment、teacher/floor context removal 均不被支持。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 坐标不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- source: `fresh_stage42_gu_floor_relaxation_paper_refresh`
- gt_verdict: `stage42_gt_floor_relaxation_safety_stress_pass`
- by_gate_passed: `True`
- bz_gate_passed: `True`
- en_gate_passed: `True`
- target_union_rows: `11538`
- target_union_t50_improvement: `0.28969780582672955`
- target_union_hard_failure_improvement: `0.28969780582672955`
- target_union_easy_degradation: `-0.21406723960013185`
- target_union_near_collision_005_delta: `-0.007379425459017833`
- target_union_jagged_rate_delta: `0.0`
- target_union_safety_pass: `True`
- deployment_decision: `partial_t50_floor_relaxation_safety_supported`
- floor_free_neural_deployable: `False`
- global_floor_removal_allowed: `False`
- paper_files_refreshed: `['outputs/stage42_long_research/method_draft_stage42.md', 'outputs/stage42_long_research/experiment_tables_stage42.md', 'outputs/stage42_long_research/ablation_tables_stage42.md', 'outputs/stage42_long_research/failure_taxonomy_stage42.md', 'outputs/stage42_long_research/model_card_stage42.md', 'outputs/stage42_long_research/data_card_stage42.md', 'outputs/stage42_long_research/reproducibility_stage42.md', 'outputs/stage42_long_research/a_journal_gap_stage42.md', 'outputs/stage42_long_research/paper_ready_evidence_matrix_stage42.md']`
- scan_files: `12`
- floor_claim_violation_count: `0`
- training_executed: `False`
- download_executed: `False`
- conversion_executed: `False`
- threshold_tuned_on_test: `False`

## Evidence Rows

| item | source | status | evidence |
| --- | --- | --- | --- |
| Stage42-BY protected t50 floor-relaxability repair | `fresh_stage42_by_t50_floor_relaxability_repair` | `protected_positive_not_floor_free` | repaired_slices=['TrajNet|50', 'UCY|50']; global_t50=28.97%; selected_variant=family_baseline_rel_only; floor_free=False |
| Stage42-BZ bootstrap evidence | `fresh_stage42_bz_t50_repair_statistical_evidence` | `statistically_positive_protected_t50` | bootstrap_n=3000; target_union_t50_ci_low=28.52%; target_union_easy_ci_high=-25.16%; ci_positive_easy_safe=True |
| Stage42-GT all-agent safety stress | `fresh_stage42_gt_floor_relaxation_safety_stress` | `all_agent_safety_supported_for_narrow_t50_relaxation` | rows=11538; t50=28.97%; hard=28.97%; easy=-21.41%; near@0.05_delta=-0.74%; jagged_delta=0.00% |
| Stage42-GT per-slice stress | `fresh_stage42_gt_floor_relaxation_safety_stress` | `slice_limited_not_global` | TrajNet|50 rows=9198, t50=30.21%, near@0.05_delta=-0.95%; UCY|50 rows=2340, t50=24.53%, near@0.05_delta=0.13% |
| Stage42-EN floor removability decision map | `fresh_stage42_floor_removability_decision_map` | `global_floor_removal_blocked` | safe_partial_floor_relaxation_available=True; global_floor_removal_allowed=False; floor_free_neural_deployable=False; teacher_floor_context_removal_allowed=False |

## Paper Files Refreshed

| file | refreshed | GT rows | floor boundary | floor-free boundary | metric boundary |
| --- | ---: | ---: | ---: | ---: | ---: |
| `outputs/stage42_long_research/method_draft_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/model_card_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/data_card_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/reproducibility_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/paper_ready_evidence_matrix_stage42.md` | True | True | True | True | True |

## Claim Linter

| file | violations |
| --- | ---: |
| `README_RESULTS.md` | 0 |
| `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md` | 0 |
| `README_M3W_CURRENT_GOAL_CONSOLIDATED_SUMMARY_ZH.md` | 0 |
| `outputs/stage42_long_research/method_draft_stage42.md` | 0 |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | 0 |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | 0 |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | 0 |
| `outputs/stage42_long_research/model_card_stage42.md` | 0 |
| `outputs/stage42_long_research/data_card_stage42.md` | 0 |
| `outputs/stage42_long_research/reproducibility_stage42.md` | 0 |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | 0 |
| `outputs/stage42_long_research/paper_ready_evidence_matrix_stage42.md` | 0 |

## Gate

| gate | pass |
| --- | ---: |
| `gt_loaded_and_passed` | True |
| `by_bz_en_loaded_and_passed` | True |
| `partial_t50_relaxation_supported` | True |
| `paper_files_refreshed` | True |
| `paper_files_contain_claim_boundaries` | True |
| `floor_claim_linter_clean` | True |
| `global_floor_removal_false` | True |
| `floor_free_neural_false` | True |
| `teacher_floor_context_removal_false` | True |
| `no_metric_seconds_overclaim` | True |
| `no_training_download_conversion_or_test_tuning` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- Stage42-GT strengthens the safety evidence for narrow t50 partial floor relaxation because the alpha-blended BY/BZ policy survives all-agent group stress checks.
- This does not permit global floor removal, floor-free neural deployment, teacher/floor context removal, or metric/seconds-level claims.
- Deployment remains protected and validation-backed; Stage5C remains unexecuted and SMC remains disabled.
- Verification after implementation: focused pytest passed; full suite passed with `929 passed`.
