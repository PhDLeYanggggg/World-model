# Stage42-G Retrained Ablation Phase1

- source: `fresh_run`
- generated_at_utc: `2026-05-26T05:01:07.723931+00:00`
- git_commit: `b1280f8`
- input_hash: `0532940fdbf60ffe62ffc5afd739a6c8d417b65f3ee4ed23558374e3d8a7136d`
- gate: `11 / 11`
- verdict: `stage42_g_retrained_ablation_phase1_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- External 数据仍是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- future endpoints / family_fde 只作为 supervised label/eval，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## What This Freshly Retrains

- expected-FDE baseline-family selector refits on external train rows.
- validation-only safety threshold selection.
- test evaluated once per variant/seed.
- variants: full, no_history, no_neighbor, no_goal, no_scene_goal, no_interaction, no_domain_expert, no_transformer_proxy_history_sequence, no_safe_switch, no_teacher_floor_proxy.

## Metrics

| variant | all mean | t50 mean | t100 diag mean | hard mean | easy mean | switch mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_retrained_external` | 0.812217 | 0.846142 | 0.952709 | 0.845888 | -0.841261 | 0.349999 |
| `no_domain_expert` | 0.812219 | 0.846139 | 0.952708 | 0.845889 | -0.841253 | 0.349999 |
| `no_goal` | 0.802580 | 0.846064 | 0.952692 | 0.835035 | -0.841516 | 0.349999 |
| `no_history` | 0.821845 | 0.854253 | 0.953245 | 0.863743 | -0.789838 | 0.349999 |
| `no_interaction` | 0.808752 | 0.846036 | 0.953056 | 0.842339 | -0.839688 | 0.349999 |
| `no_neighbor` | 0.808752 | 0.846036 | 0.953056 | 0.842339 | -0.839688 | 0.349999 |
| `no_safe_switch` | 0.790820 | 0.905293 | 0.955978 | 0.801011 | -0.910998 | 0.961882 |
| `no_scene_goal` | 0.802580 | 0.846064 | 0.952692 | 0.835035 | -0.841516 | 0.349999 |
| `no_teacher_floor_proxy` | 0.790820 | 0.905293 | 0.955978 | 0.801011 | -0.910998 | 0.961882 |
| `no_transformer_proxy_history_sequence` | 0.817834 | 0.852847 | 0.950078 | 0.859902 | -0.804650 | 0.349999 |

## Contribution Deltas

`full_minus_ablation > 0` means the removed component helped the full model on that slice.

| ablation | all delta | t50 delta | hard delta | easy delta ablation-minus-full |
| --- | ---: | ---: | ---: | ---: |
| `no_domain_expert` | -0.000002 | 0.000003 | -0.000001 | 0.000008 |
| `no_goal` | 0.009638 | 0.000078 | 0.010854 | -0.000255 |
| `no_history` | -0.009628 | -0.008111 | -0.017855 | 0.051423 |
| `no_interaction` | 0.003465 | 0.000106 | 0.003550 | 0.001573 |
| `no_neighbor` | 0.003465 | 0.000106 | 0.003550 | 0.001573 |
| `no_safe_switch` | 0.021397 | -0.059151 | 0.044877 | -0.069737 |
| `no_scene_goal` | 0.009638 | 0.000078 | 0.010854 | -0.000255 |
| `no_teacher_floor_proxy` | 0.021397 | -0.059151 | 0.044877 | -0.069737 |
| `no_transformer_proxy_history_sequence` | -0.005617 | -0.006705 | -0.014014 | 0.036610 |

## Not-Run Boundaries
- `no_jepa`: not_run_in_phase1; current deployable path does not use JEPA features, and JEPA remains diagnostic-only from Stage18/19/41 evidence.
- `full_transformer_retrain`: not_run_in_phase1; this phase retrains ridge expected-FDE selectors on causal external features, not torch Transformer checkpoints.
- `no_endpoint_bridge`: not_run_in_phase1; Stage42-C has fresh full-waypoint vs endpoint-linear comparisons, but this phase does not retrain the waypoint model.
- `no_full_waypoint_shape`: not_run_in_phase1; Stage42-C covers protected full-waypoint sequence dynamics, but all-component waypoint-shape retraining remains open.
