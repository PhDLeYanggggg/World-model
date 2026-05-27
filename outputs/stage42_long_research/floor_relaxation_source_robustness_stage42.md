# Stage42-GV Floor-Relaxation Source Robustness Audit

- source: `fresh_stage42_gv_floor_relaxation_source_robustness`
- generated_at_utc: `2026-05-27T14:45:06.521419+00:00`
- git_commit: `4e9d39b`
- input_hash: `02d9a2c2427c5588fb82b66fc8e4cb3b7e60ff9b4c9c5f5caa3b52adf4f991a2`
- gate: `14 / 14`
- verdict: `stage42_gv_floor_relaxation_source_robustness_pass_with_source_concentration_caveat`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GV 是 Stage42-GT partial t50 floor-relaxation 的 source-level all-agent robustness audit。
- 本阶段不训练新模型，不下载数据，不转换新数据，不执行 Stage5C，不启用 SMC。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 坐标不能写成 global metric。
- 如果 source 过于集中，必须保留 source-concentration caveat，不能写成 broad source-level generalization。

## Summary

- source: `fresh_stage42_gv_floor_relaxation_source_robustness`
- gt_verdict: `stage42_gt_floor_relaxation_safety_stress_pass`
- gu_verdict: `stage42_gu_floor_relaxation_paper_refresh_pass`
- target_slices: `['TrajNet|50', 'UCY|50']`
- source_safety_positive_slices: `['TrajNet|50', 'UCY|50']`
- source_concentration_limited_slices: `['TrajNet|50', 'UCY|50']`
- broad_source_generalization_claim_allowed: `False`
- source_level_claim: `major-source all-agent safety support with source-concentration caveat; not broad source-level generalization`
- target_union_rows: `11538`
- target_union_t50_improvement: `0.28969780582672955`
- target_union_near_collision_005_delta: `-0.007379425459017833`
- training_executed: `False`
- download_executed: `False`
- conversion_executed: `False`
- threshold_tuned_on_test: `False`
- global_floor_removal_allowed: `False`
- floor_free_neural_deployable: `False`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`
- stage5c_executed: `False`
- smc_enabled: `False`

## Source-Level Audit

| slice | source count | total rows | largest source fraction | safety-positive sources | broad source claim |
| --- | ---: | ---: | ---: | --- | ---: |
| `TrajNet|50` | 2 | 9198 | 99.08% | `['crowds/students002.txt', 'crowds/students003.txt']` | False |
| `UCY|50` | 1 | 2340 | 100.00% | `['crowds/crowds_zara03.txt']` | False |

## Per-Source Metrics

| slice | source | rows | groups | t50 | hard | easy | switch | near@0.05 delta | safety pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet|50` | `crowds/students002.txt` | 85 | 66 | 71.35% | 71.35% | -76.59% | 100.00% | 0.00% | True |
| `TrajNet|50` | `crowds/students003.txt` | 9113 | 531 | 29.20% | 29.20% | -22.90% | 95.22% | -0.95% | True |
| `UCY|50` | `crowds/crowds_zara03.txt` | 2340 | 663 | 24.53% | 24.53% | -12.64% | 65.00% | 0.13% | True |

## Gate

| gate | pass |
| --- | ---: |
| `gt_input_passed` | True |
| `gu_input_passed` | True |
| `target_slices_audited` | True |
| `source_rows_present` | True |
| `major_source_safety_positive` | True |
| `source_concentration_caveat_recorded` | True |
| `broad_source_generalization_not_overclaimed` | True |
| `no_training_download_conversion_or_test_tuning` | True |
| `no_leakage_pass` | True |
| `global_floor_removal_false` | True |
| `floor_free_neural_false` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- Stage42-GT partial t50 floor relaxation has source-level all-agent safety support on the available major sources.
- The evidence remains source-concentrated, so broad source-level generalization is still disallowed.
- This strengthens the safety claim for audited t50 slices while preserving the source-diversity limitation.
- Stage5C remains unexecuted and SMC remains disabled.
