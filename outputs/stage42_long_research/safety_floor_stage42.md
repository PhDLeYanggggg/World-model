# Stage42-E Safety Floor Research

- source: `fresh_run`
- generated_at_utc: `2026-05-25T20:26:06.685652+00:00`
- git_commit: `8265c5d`
- input_hash: `d27165c007daed939036476a33d58182055215660422453f513b6c0b36e4437e`

## Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-E safety-floor study 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。
- future endpoints / future waypoints 只作为 loss/eval label，不作为 inference input。
- 所有 threshold/policy 选择只用 validation；test 只最终评估一次。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Best Deployable Policy

- family: `current_composite_tail_policy`
- source: `cached_verified_policy_fresh_eval`
- deployable: `True`
- metrics: `{'rows': 55528, 'all_improvement': 0.2102513255185352, 't10_improvement': 0.4810525302753197, 't25_improvement': 0.10635705713395971, 't50_improvement': 0.13652231450154184, 't100_improvement': 0.14694086716388166, 'hard_failure_improvement': 0.20384916307933942, 'easy_degradation': 0.0, 'harm_over_fallback': -0.1006325590804755, 'switch_rate': 0.3410171445036738, 'alpha_mean': 0.29906641694280367, 'alpha_positive_rate': 0.3410171445036738, 'collision_delta_vs_floor_005': -0.0038702813749587617, 'smoothness_jagged_delta': 0.0}`

## Switch Gate Families

| family | source | deployable | all | t50 | t100 diag | hard/failure | easy degr | switch | collision d005 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher_repaired_floor` | `fresh_run` | `True` | 0.2036 | 0.1312 | 0.1337 | 0.1966 | 0.0000 | 0.2954 | -0.0040 |
| `teacher_prob_gate` | `fresh_run` | `True` | 0.1212 | 0.0931 | 0.1655 | 0.1270 | 0.0000 | 0.1250 | 0.0044 |
| `harm_predictor_gate` | `fresh_run` | `False` | 0.3595 | 0.2520 | 0.3667 | 0.3586 | 0.0000 | 0.5055 | 0.0197 |
| `uncertainty_gate` | `fresh_run` | `False` | 0.3592 | 0.2510 | 0.3665 | 0.3582 | 0.0000 | 0.5074 | 0.0196 |
| `internal_self_gate` | `fresh_run` | `False` | 0.3587 | 0.2500 | 0.3658 | 0.3577 | 0.0000 | 0.5012 | 0.0195 |
| `conformal_risk_gate` | `fresh_run` | `False` | 0.3513 | 0.2352 | 0.3623 | 0.3518 | 0.0000 | 0.4509 | 0.0186 |
| `teacher_raw_policy` | `fresh_run` | `False` | 0.3515 | 0.2367 | 0.3580 | 0.3509 | 0.0000 | 0.4619 | 0.0187 |
| `floor_only` | `fresh_run` | `False` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Bounded Residual / Blend Families

| family | source | deployable | all | t50 | t100 diag | hard/failure | easy degr | alpha/switch | collision d005 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_composite_tail_policy` | `cached_verified_policy_fresh_eval` | `True` | 0.2103 | 0.1365 | 0.1469 | 0.2038 | 0.0000 | 0.3410 | -0.0039 |
| `bounded_teacher_switch_alpha` | `fresh_run` | `True` | 0.2036 | 0.1312 | 0.1337 | 0.1966 | 0.0000 | 0.2954 | -0.0040 |
| `bounded_teacher_prob70_alpha` | `fresh_run` | `True` | 0.0838 | 0.0858 | 0.1091 | 0.0890 | 0.0000 | 0.0868 | 0.0041 |
| `bounded_horizon_teacher_switch_alpha` | `fresh_run` | `True` | 0.0384 | 0.0481 | 0.0439 | 0.0395 | 0.0000 | 0.2954 | -0.0006 |
| `bounded_horizon_alpha` | `fresh_run` | `False` | 0.0992 | 0.1231 | 0.1364 | 0.1055 | 0.0509 | 1.0000 | 0.0027 |
| `bounded_all_rows_alpha` | `fresh_run` | `False` | 0.2966 | 0.2152 | 0.3592 | 0.3294 | 1.2459 | 1.0000 | 0.0377 |

## Floor Necessity

- conclusion: `teacher_floor_required_for_current_deployment`
- floor-only metrics: `{'rows': 55528, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'collision_delta_vs_floor_005': 0.0}`
- ungated endpoint metrics from Stage42-B: `{'rows': 55528, 'all_improvement': 0.296621240422128, 't10_improvement': 0.49983686172432895, 't25_improvement': -0.044336116247622126, 't50_improvement': 0.21520334842637612, 't100_improvement': 0.359213230578716, 'hard_failure_improvement': 0.32937413785262626, 'easy_degradation': 1.2458611044726973, 'harm_over_fallback': -0.14197177795519889, 'switch_rate': 1.0}`
- ungated full-waypoint metrics from Stage42-C: `{'rows': 55528, 'all_improvement': 0.296621240422128, 't10_improvement': 0.49983686172432895, 't25_improvement': -0.044336116247622126, 't50_improvement': 0.21520334842637612, 't100_improvement': 0.359213230578716, 'hard_failure_improvement': 0.32937413785262626, 'easy_degradation': 1.2458611044726973, 'harm_over_fallback': -0.14197177795519889, 'switch_rate': 1.0}`
- no-teacher internal deployable families: `[]`
- bounded no-switch deployable families: `[]`
- why: Ungated neural improves raw errors but violates easy safety. Validation-selected internal/harm/uncertainty gates are reported separately; only deployable families may be considered for floor reduction, never Stage5C/SMC execution.

## Cached-Verified Context

- Stage42-B verdict: `stage42_b_external_validation_pass_protected_neural_not_ungated`
- Stage42-C verdict: `stage42_c_full_waypoint_dynamics_pass`
- Stage42-D verdict: `stage42_d_causal_ablation_evidence_pass_with_retrain_boundary`
- Stage41 composite-tail evidence pass: `True`

## Verdict

`stage42_e_safety_floor_research_pass` (12 / 12)
