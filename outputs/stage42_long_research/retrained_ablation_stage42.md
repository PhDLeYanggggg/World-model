# Stage42-G Retrained Ablation Phase1

- source: `fresh_run`
- generated_at_utc: `2026-05-25T20:47:06.503899+00:00`
- git_commit: `fe260d1`
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
| `full_retrained_external` | 0.812216 | 0.846151 | 0.952708 | 0.845884 | -0.841275 | 0.349999 |
| `no_domain_expert` | 0.812215 | 0.846141 | 0.952708 | 0.845883 | -0.841275 | 0.349999 |
| `no_goal` | 0.802529 | 0.846021 | 0.952680 | 0.834977 | -0.841562 | 0.349999 |
| `no_history` | 0.821830 | 0.854175 | 0.953179 | 0.863741 | -0.789750 | 0.349999 |
| `no_interaction` | 0.808850 | 0.845984 | 0.953045 | 0.842431 | -0.839808 | 0.349999 |
| `no_neighbor` | 0.808850 | 0.845984 | 0.953045 | 0.842431 | -0.839808 | 0.349999 |
| `no_safe_switch` | 0.790828 | 0.905292 | 0.955978 | 0.801008 | -0.910996 | 0.961751 |
| `no_scene_goal` | 0.802529 | 0.846021 | 0.952680 | 0.834977 | -0.841562 | 0.349999 |
| `no_teacher_floor_proxy` | 0.790828 | 0.905292 | 0.955978 | 0.801008 | -0.910996 | 0.961751 |
| `no_transformer_proxy_history_sequence` | 0.817870 | 0.852805 | 0.950085 | 0.859882 | -0.804738 | 0.349999 |

## Contribution Deltas

`full_minus_ablation > 0` means the removed component helped the full model on that slice.

| ablation | all delta | t50 delta | hard delta | easy delta ablation-minus-full |
| --- | ---: | ---: | ---: | ---: |
| `no_domain_expert` | 0.000001 | 0.000009 | 0.000001 | 0.000000 |
| `no_goal` | 0.009687 | 0.000130 | 0.010908 | -0.000287 |
| `no_history` | -0.009614 | -0.008024 | -0.017856 | 0.051525 |
| `no_interaction` | 0.003366 | 0.000167 | 0.003453 | 0.001467 |
| `no_neighbor` | 0.003366 | 0.000167 | 0.003453 | 0.001467 |
| `no_safe_switch` | 0.021389 | -0.059141 | 0.044876 | -0.069721 |
| `no_scene_goal` | 0.009687 | 0.000130 | 0.010908 | -0.000287 |
| `no_teacher_floor_proxy` | 0.021389 | -0.059141 | 0.044876 | -0.069721 |
| `no_transformer_proxy_history_sequence` | -0.005654 | -0.006655 | -0.013998 | 0.036537 |

## Not-Run Boundaries
- `no_jepa`: not_run_in_phase1; current deployable path does not use JEPA features, and JEPA remains diagnostic-only from Stage18/19/41 evidence.
- `full_transformer_retrain`: not_run_in_phase1; this phase retrains ridge expected-FDE selectors on causal external features, not torch Transformer checkpoints.
- `no_endpoint_bridge`: not_run_in_phase1; Stage42-C has fresh full-waypoint vs endpoint-linear comparisons, but this phase does not retrain the waypoint model.
- `no_full_waypoint_shape`: not_run_in_phase1; Stage42-C covers protected full-waypoint sequence dynamics, but all-component waypoint-shape retraining remains open.
