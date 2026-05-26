# Stage42-AC Paper Package Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- generated_at_utc: `2026-05-26T05:45:43.283329+00:00`
- git_commit: `40e84fb`
- input_hash: `900c2a7cbb37318f1bfa412a2451d80f5266f346312991168227d7c9422a31ce`
- gate: `12 / 12`
- verdict: `stage42_ac_paper_package_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AC 刷新 paper package，不重新训练模型，不读取 raw data/cache。
- Stage42-AB auxiliary-head ablation 是 fresh retrained evidence，但结果是 mixed/partial，不是统一正贡献。
- future endpoints / waypoints 只可作为 label 或 evaluation，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Refreshed Evidence Rows

| item | source | status | paper use | evidence |
| --- | --- | --- | --- | --- |
| Stage42-X unified row-level full-waypoint cache | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `stage42_x_unified_row_level_full_waypoint_cache_pass` | main protected 2.5D full-waypoint evidence | ADE all=0.090014, t50=0.061094, hard=0.093746, easy=0.001102 |
| Stage42-Y unified ablation evidence | `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports` | `stage42_y_unified_ablation_evidence_pass` | paper-table ablation synthesis | gate=13/13; history/domain positive, goal/neighbor mixed |
| Stage42-Z claim evidence audit | `fresh_audit_from_stage42_wxy_and_paper_package_artifacts` | `stage42_z_paper_claim_evidence_audit_pass` | claim boundary audit | supports protected 2.5D raw-frame paper scope; rejects metric/seconds/foundation/ungated claims |
| Stage42-AA retrained ablation matrix | `fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z` | `stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary` | required ablation coverage matrix | gate=15/15; no-JEPA cached negative; no-Transformer proxy boundary |
| Stage42-AB auxiliary-head retrained ablation | `fresh_run` | `stage42_ab_full_waypoint_auxiliary_ablation_pass` | mixed/partial auxiliary evidence, not main uniform-positive claim | no_aux all=-0.002339, t50=-0.037443; full-minus-no-aux t50=0.005361, all=-0.008219; uniform_positive=False |

## Verdict

- Stage42-AB is included in the paper package as mixed/partial auxiliary-head evidence.
- The current paper-ready claim remains protected dataset-local raw-frame 2.5D world-state dynamics.
- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim is enabled.
