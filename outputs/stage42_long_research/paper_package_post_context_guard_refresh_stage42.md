# Stage42-CL Post-CJ/CK Context Guard Paper Refresh

- source: `fresh_synthesis_from_stage42_cj_ck_artifacts`
- generated_at_utc: `2026-05-26T18:18:34.636269+00:00`
- git_commit: `55621fd`
- input_hash: `1509149e9c311795afb1f817bf8c0efb3ed80002e2760286fa463d07fbc882fb`
- gate: `11 / 11`
- verdict: `stage42_cl_context_guard_paper_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CL 是 paper package refresh，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。
- Stage42-CJ/CK 是 validation-only gated expert audits；test 只最终评估，不用于选择。
- goal/scene 与 neighbor/interaction 仍是 diagnostic / auxiliary evidence，不是独立主贡献。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-CJ goal/scene gated expert | `diagnostic_negative` | claim boundary / limitation | gate=10/10; selected=baseline_family_control; goal_scene_rescue_success=False; control all/t50/hard=28.78%/31.54%/27.58%; goal all/t50/hard=26.25%/22.76%/24.86% |
| Stage42-CJ motion+goal context | `diagnostic_negative` | ablation boundary | motion_goal all/t50/hard=24.58%/22.02%/23.75%; delta_t50_vs_control=-9.53% |
| Stage42-CK neighbor/interaction gated expert | `diagnostic_negative` | claim boundary / limitation | gate=11/11; selected=baseline_family_control; neighbor_interaction_rescue_success=False; graph_rows=337991; rows_with_neighbors=334525; control all/t50/hard=28.78%/31.54%/27.58% |
| Stage42-CK graph/scalar candidates | `diagnostic_negative` | ablation boundary | scalar all/t50/hard=26.37%/22.96%/24.88%; knn_graph all/t50/hard=24.38%/22.38%/23.78%; graph_goal all/t50/hard=20.67%/22.21%/18.81% |

## Paper File Status

| file | refreshed | blocks goal/scene | blocks neighbor/interaction | metric boundary |
| --- | ---: | ---: | ---: | ---: |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | True | True | True | True |

## Interpretation

- CJ/CK are fresh diagnostic negative evidence, not failures hidden under a broader success claim.
- The paper package can claim baseline-family rollout context, causal history, guarded domain expert, and protected safe-switch evidence.
- The paper package must not claim goal/scene or neighbor/interaction as independent uniformly positive main contributions under the current protocol.
- Stage5C and SMC remain disabled.
