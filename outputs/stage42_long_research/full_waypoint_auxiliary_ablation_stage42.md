# Stage42-AB Full-Waypoint Auxiliary-Head Ablation

- source: `fresh_run`
- generated_at_utc: `2026-05-26T05:37:57.957457+00:00`
- git_commit: `adc08ac`
- input_hash: `1a34619d3bc7d3d320ad0e17b6238ab60ff7b09a310e73547415e78a90afda54`
- gate: `11 / 11`
- verdict: `stage42_ab_full_waypoint_auxiliary_ablation_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AB 是 retrained auxiliary-head loss ablation，不是 inference masking。
- future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## No-Aux Metrics

| metric | mean | ci low | ci high |
| --- | ---: | ---: | ---: |
| `ade_all` | -0.002339 | -0.015347 | 0.010670 |
| `ade_t50` | -0.037443 | -0.047483 | -0.027403 |
| `ade_t100_raw_frame_diagnostic` | -0.014136 | -0.042862 | 0.014590 |
| `ade_hard_failure` | -0.002564 | -0.016823 | 0.011696 |
| `ade_easy_degradation` | 0.000000 | 0.000000 | 0.000000 |
| `fde_all` | 0.013054 | 0.001114 | 0.024995 |
| `fde_t50` | 0.015052 | -0.006814 | 0.036918 |
| `switch_rate` | 0.092380 | 0.045211 | 0.139549 |
| `ungated_easy_degradation` | 8.094260 | 7.055019 | 9.133501 |

## Full Minus No-Aux

`full_minus_no_aux > 0` means the supervised auxiliary heads helped the Stage42-I full-waypoint model on that metric.

| delta | value |
| --- | ---: |
| `ade_all_delta_full_minus_no_aux` | -0.008219 |
| `ade_t50_delta_full_minus_no_aux` | 0.005361 |
| `ade_hard_delta_full_minus_no_aux` | -0.009027 |
| `fde_t50_delta_full_minus_no_aux` | 0.005084 |

## Interpretation

- uniform_aux_positive_claim_allowed: `False`
- conclusion: Auxiliary interaction/occupancy/physical heads are mixed or negative in this retrained ablation; keep them as limitation/partial evidence rather than a main positive claim.
- This is a fresh retrained ablation over full-waypoint sequence dynamics, but it remains dataset-local raw-frame 2.5D.
- Stage5C and SMC remain disabled.
