# Stage42-JY Context Materiality By Source Slice

- source: `fresh_stage42_jy_context_materiality_by_source_slice`
- generated_at_utc: `2026-05-29T07:48:26.241591+00:00`
- git_commit: `e01d0b8`
- input_hash: `7c24bb0f4b21f870f736474a3ec65119be7059754c6b09aaffff993d74dbc883`
- gate: `14 / 14`
- verdict: `stage42_jy_context_materiality_by_source_slice_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JY 审计 context 模块在 source/domain/horizon 切片上的 materiality，不训练、不调 threshold。
- JY 使用 AO fresh source-level incremental ablation、JV source-slice matrix、JT claim refresh 和 JS context closure。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Baseline-Family Control

- all/t50/t100raw/hard: `28.78%` / `31.54%` / `14.28%` / `27.58%`
- easy_degradation: `-32.42%`; switch_rate: `66.06%`

## Context Materiality Summary

- positive_standalone_context_variants: `['history_only', 'motion_goal_context']`
- AO positive_incremental_context_variants: `[]`
- material_global_incremental_variants: `[]`
- best_narrow_slice_signal: `{'variant': 'motion_goal_context', 'slice': 'horizon=10', 'metric': 'all_improvement', 'delta': 0.02748739379455012}`
- context_claim_decision: `block_independent_context_main_claim_keep_as_auxiliary_or_new_objective`

## Variant Delta vs Baseline-Family Control

| variant | all delta | t50 delta | t100raw delta | hard delta | easy delta | material global | narrow horizons |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `full` | -4.20% | -9.53% | 0.08% | -3.83% | 6.76% | `False` | `['100']` |
| `horizon_domain_only` | -28.78% | -31.54% | -14.28% | -27.58% | 32.42% | `False` | `[]` |
| `history_only` | -11.24% | -11.57% | -14.56% | -10.77% | 18.26% | `False` | `['10']` |
| `goal_only` | -28.78% | -31.54% | -14.28% | -27.58% | 32.42% | `False` | `[]` |
| `neighbor_only` | -28.78% | -31.54% | -14.28% | -27.58% | 32.42% | `False` | `[]` |
| `motion_goal_context` | -10.90% | -11.31% | -14.28% | -10.40% | 17.26% | `False` | `['10']` |
| `baseline_plus_history` | -2.94% | -8.89% | 0.10% | -3.56% | 1.12% | `False` | `['100']` |
| `baseline_plus_goal` | -2.53% | -8.78% | -0.01% | -2.72% | 2.25% | `False` | `[]` |
| `baseline_plus_neighbor` | -2.40% | -8.58% | 0.04% | -2.70% | 1.10% | `False` | `['100']` |
| `baseline_plus_history_goal_neighbor` | -4.20% | -9.53% | 0.08% | -3.83% | 6.76% | `False` | `['100']` |

## Next Training Spec

- Do not repeat the closed residual/sequence/graph context family unchanged.
- If context is retried, optimize source/horizon-slice objectives against baseline-family control, not only global raw ADE.
- Use validation-only source/horizon routing and preserve Stage37/teacher floor for deployment.
- Treat history/motion-goal as auxiliary candidates; require material all/t50/hard improvement and easy <=2% before paper main claim.
- For t100 raw-frame diagnostic, test a dedicated source-slice objective because only micro-slice deltas appear in current evidence.

## Gate

| gate | pass |
| --- | ---: |
| `ao_incremental_ablation_loaded` | `True` |
| `jt_claim_refresh_passed` | `True` |
| `jv_source_slice_matrix_passed` | `True` |
| `js_context_closure_passed` | `True` |
| `baseline_family_control_positive` | `True` |
| `standalone_context_signal_recorded` | `True` |
| `no_material_global_incremental_context` | `True` |
| `narrow_slice_signals_recorded` | `True` |
| `blocked_context_claim_preserved` | `True` |
| `next_training_spec_emitted` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_3d_foundation` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- Current context evidence is not globally material after baseline-family rollout context.
- There are useful standalone/context slice signals, but the paper main claim must stay with protected row-cache/full-waypoint + baseline-family/safe-switch/floor until a new source-slice objective proves material gain.
