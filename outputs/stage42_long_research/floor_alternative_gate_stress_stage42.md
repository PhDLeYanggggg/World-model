# Stage42-HC Floor-Alternative Gate Stress Matrix

- source: `fresh_stage42_hc_floor_alternative_gate_stress`
- generated_at_utc: `2026-05-27T16:12:56.903808+00:00`
- git_commit: `5814bb9`
- input_hash: `da6334c640d2de5ec03b9310f0f52aedca03dd245665383da30639305e642099`
- gate: `14 / 14`
- verdict: `stage42_hc_floor_alternative_gate_stress_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HC 是 floor-alternative gate stress matrix，不重新训练，不下载，不转换，不调 test threshold。
- 本审计使用 Stage42-E fresh validation-selected gate families，并把结果按 floor-free / teacher-dependent / bounded residual 分组。
- future endpoint / future waypoint 只允许作为监督或评估标签，不允许作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C latent generative 未执行，SMC 未启用。

## Direct Decision

- deployment_decision: `keep_stage37_teacher_floor_globally_allow_only_validation_backed_partial_t50_relaxation`
- floor_free_deployable_count: `0`
- teacher_dependent_deployable_count: `6`
- partial_t50_target_slices: `['TrajNet|50', 'UCY|50']`
- global_floor_removal_allowed: `False`
- floor_free_neural_deployable: `False`

## Best Candidates

- best_floor_free_candidate: `harm_predictor_gate`; strict_deployable=`False`; failure_reasons=`['near_collision_delta_over_1pp']`; metrics=`{'rows': 55528, 'all_improvement': 0.35954392093793963, 't10_improvement': 0.6065347468876006, 't25_improvement': 0.15477738209432224, 't50_improvement': 0.2519769905692425, 't100_improvement': 0.3666779052571525, 'hard_failure_improvement': 0.35864193491919594, 'easy_degradation': 0.0, 'harm_over_fallback': -0.17208845069860598, 'switch_rate': 0.5055467511885895, 'collision_delta_vs_floor_005': 0.019718258189955595}`
- best_teacher_dependent_candidate: `teacher_raw_policy`; strict_deployable=`False`; failure_reasons=`['near_collision_delta_over_1pp']`; metrics=`{'rows': 55528, 'all_improvement': 0.35147372419646705, 't10_improvement': 0.6053046394540129, 't25_improvement': 0.1510837734096614, 't50_improvement': 0.23666446707361, 't100_improvement': 0.3579659237738704, 'hard_failure_improvement': 0.350941376256631, 'easy_degradation': 0.0, 'harm_over_fallback': -0.16822581369322984, 'switch_rate': 0.46194712577438407, 'collision_delta_vs_floor_005': 0.018672731941744014}`
- best_deployable_teacher_dependent_candidate: `current_composite_tail_policy`; strict_deployable=`True`; metrics=`{'rows': 55528, 'all_improvement': 0.2102513255185352, 't10_improvement': 0.4810525302753197, 't25_improvement': 0.10635705713395971, 't50_improvement': 0.13652231450154184, 't100_improvement': 0.14694086716388166, 'hard_failure_improvement': 0.20384916307933942, 'easy_degradation': 0.0, 'harm_over_fallback': -0.1006325590804755, 'switch_rate': 0.3410171445036738, 'alpha_mean': 0.29906641694280367, 'alpha_positive_rate': 0.3410171445036738, 'collision_delta_vs_floor_005': -0.0038702813749587617, 'smoothness_jagged_delta': 0.0}`

## Candidate Matrix

| family | type | strict deployable | Stage42-E deployable | all | t50 | t100 raw | hard | easy | collision d005 | failure reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `teacher_repaired_floor` | `teacher_dependent_switch_gate` | True | True | 20.36% | 13.12% | 13.37% | 19.66% | 0.00% | -0.40% | `[]` |
| `teacher_prob_gate` | `teacher_dependent_switch_gate` | True | True | 12.12% | 9.31% | 16.55% | 12.70% | 0.00% | 0.44% | `[]` |
| `harm_predictor_gate` | `floor_free_switch_gate` | False | False | 35.95% | 25.20% | 36.67% | 35.86% | 0.00% | 1.97% | `['near_collision_delta_over_1pp']` |
| `uncertainty_gate` | `floor_free_switch_gate` | False | False | 35.92% | 25.10% | 36.65% | 35.82% | 0.00% | 1.96% | `['near_collision_delta_over_1pp']` |
| `internal_self_gate` | `floor_free_switch_gate` | False | False | 35.87% | 25.00% | 36.58% | 35.77% | 0.00% | 1.95% | `['near_collision_delta_over_1pp']` |
| `conformal_risk_gate` | `floor_free_switch_gate` | False | False | 35.13% | 23.52% | 36.23% | 35.18% | 0.00% | 1.86% | `['near_collision_delta_over_1pp']` |
| `teacher_raw_policy` | `teacher_dependent_switch_gate` | False | False | 35.15% | 23.67% | 35.80% | 35.09% | 0.00% | 1.87% | `['near_collision_delta_over_1pp']` |
| `floor_only` | `other_switch_gate` | False | False | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | `['all_not_positive', 't50_and_hard_not_positive', 'no_intervention']` |
| `current_composite_tail_policy` | `teacher_dependent_bounded_or_current` | True | True | 21.03% | 13.65% | 14.69% | 20.38% | 0.00% | -0.39% | `[]` |
| `bounded_teacher_switch_alpha` | `teacher_dependent_bounded_or_current` | True | True | 20.36% | 13.12% | 13.37% | 19.66% | 0.00% | -0.40% | `[]` |
| `bounded_teacher_prob70_alpha` | `teacher_dependent_bounded_or_current` | True | True | 8.38% | 8.58% | 10.91% | 8.90% | 0.00% | 0.41% | `[]` |
| `bounded_horizon_teacher_switch_alpha` | `teacher_dependent_bounded_or_current` | True | True | 3.84% | 4.81% | 4.39% | 3.95% | 0.00% | -0.06% | `[]` |
| `bounded_horizon_alpha` | `floor_free_bounded_residual` | False | False | 9.92% | 12.31% | 13.64% | 10.55% | 5.09% | 0.27% | `['easy_degradation_over_2pct']` |
| `bounded_all_rows_alpha` | `floor_free_bounded_residual` | False | False | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | 3.77% | `['easy_degradation_over_2pct', 'near_collision_delta_over_1pp']` |

## Interpretation

- Floor-free switch gates can produce high raw improvements, but the best such candidates fail strict deployment because near-collision delta exceeds the safety limit.
- Teacher-dependent gates and the current composite remain deployable, which supports HB's conclusion that Stage37/teacher floor is still a core mechanism.
- Partial t50 floor relaxation remains allowed only for validation-backed slices and does not license global floor removal.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is made.
