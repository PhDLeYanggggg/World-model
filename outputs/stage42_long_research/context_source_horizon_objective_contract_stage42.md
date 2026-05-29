# Stage42-KA Context Source/Horizon Objective Contract

- source: `fresh_stage42_ka_context_source_horizon_objective_contract`
- generated_at_utc: `2026-05-29T08:14:49.911210+00:00`
- git_commit: `4316f32`
- input_hash: `55c277ce40cc5cd75cba6fc0c83f01106cbd2493f8f1670d7dd7eed249b6dc62`
- gate: `15 / 15`
- verdict: `stage42_ka_context_source_horizon_objective_contract_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-KA 不是新训练；它把 AO/JY/JS/IO 的 fresh/cached-verified context 证据转成 source+horizon objective contract。
- future endpoints / future waypoints 只能作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Baseline-Family Control

- all/t50/t100raw/hard: `28.78%` / `31.54%` / `14.28%` / `27.58%`
- easy_degradation: `-32.42%`; switch_rate: `66.06%`

## Contract Summary

- global_material_context_variants: `[]`
- narrow_auxiliary_context_slices: `[{'variant': 'history_only', 'horizon': 10}, {'variant': 'motion_goal_context', 'horizon': 10}]`
- diagnostic_router_conflicts: `[{'horizon': 25, 'candidate': 'baseline_plus_history_goal_neighbor', 'decision': 'diagnostic_router_only_not_baseline_family_positive'}]`
- t50_oracle_headroom: `3.52%`
- t100_oracle_headroom: `1.12%`
- claim_decision: `keep_scene_goal_neighbor_interaction_blocked_as_independent_main_claims`
- deployment_decision: `keep_baseline_family_stage37_teacher_floor_as_deployable_context_mechanism`

## Horizon Objective Matrix

| horizon | candidate | current supported | decision | reason | delta all | delta t50 | delta t100raw | delta hard | easy delta |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | `motion_goal_context` | `True` | `auxiliary_retrain_candidate_only` | narrow horizon-positive router exists, but it is not a global context contribution. | 2.75% | 0.00% | 0.00% | 2.69% | -1.71% |
| 25 | `baseline_plus_history_goal_neighbor` | `True` | `diagnostic_router_only_not_baseline_family_positive` | horizon router was positive in the sequence/graph protocol, but it is not positive versus baseline-family control in this contract. | -8.29% | 0.00% | 0.00% | -6.45% | 11.69% |
| 50 | `baseline_plus_history_goal_neighbor` | `False` | `blocked_until_new_row_level_objective` | router_under_switches_despite_headroom | -9.53% | -9.53% | 0.00% | -9.53% | 5.88% |
| 100 | `history_only` | `False` | `diagnostic_blocked_until_new_source_slice_objective` | weak_predictive_signal_or_baseline_family_dominance | -14.56% | 0.00% | -14.56% | -14.56% | 3.06% |

## Next Training Contract

- Do not promote any current context variant globally: none beats baseline-family control on all+t50+hard with easy<=2%.
- Use h10 context routers only as auxiliary/narrow evidence unless retrained source-level validation proves material global lift.
- Treat h25 as diagnostic-only when it is positive in the router protocol but negative versus baseline-family control.
- For t50, build a new row-level source/horizon objective because current gain/harm context routers under-switch despite oracle headroom.
- For t100 raw-frame diagnostic, build a separate source-slice objective; current evidence is micro-positive but not material.
- Preserve Stage37/teacher floor and baseline-family rollout context for any deployable policy.
- Keep future endpoints/waypoints as labels only; no central velocity, no test endpoint goals, no test threshold tuning.

## Gate

| gate | pass |
| --- | ---: |
| `ao_incremental_ablation_loaded` | `True` |
| `io_horizon_router_loaded` | `True` |
| `js_context_closure_loaded` | `True` |
| `jy_materiality_loaded` | `True` |
| `baseline_family_control_positive` | `True` |
| `no_global_context_promotion` | `True` |
| `narrow_h10_auxiliary_recorded` | `True` |
| `diagnostic_h25_conflict_recorded` | `True` |
| `t50_context_blocker_recorded` | `True` |
| `t100_context_blocker_recorded` | `True` |
| `next_training_contract_emitted` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_3d_foundation` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- KA turns the current context evidence into an explicit claim/deployment contract.
- Context modules remain useful as auxiliary diagnostics and narrow h10 objectives, but not as an independent global paper contribution.
- t50/t100 context work must change objective and row-level supervision rather than repeating the closed threshold/router family.
