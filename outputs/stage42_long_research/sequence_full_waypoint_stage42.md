# Stage42-I Sequence-To-Full-Waypoint Dynamics

- source: `fresh_run`
- generated_at_utc: `2026-05-25T22:29:43.456046+00:00`
- git_commit: `5a4987e`
- input_hash: `30c0c23271a3317a818220e4dfe60f97df0dece6125eb12095013e79406447b3`
- gate: `10 / 11`
- verdict: `stage42_i_sequence_full_waypoint_partial`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-I full-waypoint evaluation 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Metrics

| variant | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | ungated easy degr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequence_waypoint_full` | -0.010558 | -0.032082 | -0.003398 | -0.011591 | 0.000000 | 0.020169 | 0.020136 | 0.081971 | 9.686026 |
| `sequence_waypoint_no_history` | -0.016313 | -0.036105 | -0.018173 | -0.017882 | 0.000000 | 0.010187 | 0.010753 | 0.030711 | 11.951696 |
| `sequence_waypoint_no_neighbor` | -0.009093 | -0.032599 | 0.002989 | -0.009968 | 0.000000 | 0.018782 | 0.019630 | 0.098047 | 9.572726 |
| `sequence_waypoint_no_static_context` | 0.011491 | 0.019854 | 0.014125 | 0.012881 | 0.000000 | 0.025397 | 0.061141 | 0.077901 | 5.504676 |

## Contribution Deltas

`full_minus_ablation > 0` means the removed component helped the full sequence-to-waypoint model.

| ablation | ADE all delta | ADE t50 delta | ADE hard delta | FDE t50 delta |
| --- | ---: | ---: | ---: | ---: |
| `sequence_waypoint_no_history` | 0.005755 | 0.004024 | 0.006291 | 0.009384 |
| `sequence_waypoint_no_neighbor` | -0.001465 | 0.000517 | -0.001623 | 0.000506 |
| `sequence_waypoint_no_static_context` | -0.022049 | -0.051936 | -0.024472 | -0.041005 |

## Interpretation

- Stage42-I connects the Stage42-H causal sequence-history signal to actual reconstructed full-waypoint ADE/FDE labels.
- All thresholds are selected on validation; test is evaluated once per variant/seed.
- Future waypoints are supervised labels/eval only, not inference inputs.
- Results remain dataset-local raw-frame 2.5D and do not enable Stage5C or SMC.

## Failure Analysis

- The main `sequence_waypoint_full` model is not deployable: protected ADE all/t50/hard are negative (`-0.0106`, `-0.0321`, `-0.0116`) although FDE all/t50 is positive (`0.0202`, `0.0201`) and easy degradation is controlled.
- Removing history makes t50 and hard/failure slightly worse, so the Stage42-H history signal does transfer weakly into full-waypoint labels: `history_ade_t50_delta = 0.0040`, `history_fde_t50_delta = 0.0094`.
- Removing neighbor tokens is nearly neutral, with tiny positive t50 delta but negative all/hard delta. Neighbor interaction is not yet a robust full-waypoint contribution under this small sequence head.
- The strongest variant is `sequence_waypoint_no_static_context`, with ADE all `0.0115`, ADE t50 `0.0199`, ADE hard `0.0129`, FDE t50 `0.0611`, and easy degradation `0.0`. This points to static/context feature overfit or cross-domain scale mismatch in the full model.
- Next repair should not blindly enlarge the model. It should train a static-gated or static-dropout sequence-to-waypoint head, keep the causal history encoder, and allow static/context features only when validation predicts positive gain without easy/proximity harm.
