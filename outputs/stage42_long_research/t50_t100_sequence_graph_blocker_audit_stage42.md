# Stage42-IP t50/t100 Sequence+Graph Blocker Audit

- source: `fresh_stage42_t50_t100_sequence_graph_blocker_audit`
- generated_at_utc: `2026-05-29T04:45:50.164802+00:00`
- git_commit: `0ca07d9`
- input_hash: `9229e20f0f7c442c4d7a483ac410f0dc9b29450fe6ed7a26293665022baf319f`
- gate: `12 / 12`
- verdict: `stage42_ip_t50_t100_sequence_graph_blocker_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IP 是 Stage42-IO 后续 blocker audit：只解释 t50/t100 sequence+graph context 为什么没有形成 deployable lift。
- sequence summary 与 graph summary 只使用当前帧和过去 history。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- t50_diagnosis: `router_under_switches_despite_headroom`
- t100_diagnosis: `weak_predictive_signal_or_baseline_family_dominance`
- blocker_counts: `{'unsafe_or_uncalibrated_switching': 2, 'weak_predictive_signal_or_baseline_family_dominance': 2, 'router_under_switches_despite_headroom': 1, 'low_margin_candidate_ambiguity': 1}`

Stage42-IP does not add a new deployable model. It converts the Stage42-IO t50/t100 negative result into a blocker map: whether the issue is missing candidate oracle headroom, low-margin ambiguity, train/test shift, under-switching, unsafe switching, or baseline-family dominance.

## Best By Horizon

| horizon | best oracle | oracle headroom | best router | router improvement | best capture | capture rate | dominant blocker |
| ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| 50 | `h50_history_only` | 0.035246 | `h50_baseline_plus_history_goal_neighbor` | 0.000001 | `h50_history_only` | 0.162816 | `router_under_switches_despite_headroom` |
| 100 | `h100_history_only` | 0.011163 | `h100_history_only` | 0.001448 | `h100_baseline_plus_history_goal_neighbor` | 0.347605 | `weak_predictive_signal_or_baseline_family_dominance` |

## Candidate-Horizon Audit

| key | rows test | oracle | router | capture | pos gain | switch | switched good | missed good | low margin 0.01 | easy deg | blocker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `h100_baseline_plus_history_goal_neighbor` | 7048 | 0.005947 | 0.001279 | 0.347605 | 0.107123 | 0.187287 | 0.219697 | 0.081180 | 0.829030 | -0.000398 | `low_margin_candidate_ambiguity` |
| `h100_history_only` | 7048 | 0.011163 | 0.001448 | 0.269808 | 0.089671 | 0.128405 | 0.142541 | 0.081882 | 0.474177 | 0.011562 | `weak_predictive_signal_or_baseline_family_dominance` |
| `h100_motion_goal_context` | 7048 | 0.010689 | 0.001432 | 0.253370 | 0.086833 | 0.101873 | 0.160167 | 0.078515 | 0.492480 | 0.009937 | `weak_predictive_signal_or_baseline_family_dominance` |
| `h50_baseline_plus_history_goal_neighbor` | 11538 | 0.012826 | 0.000001 | 0.000059 | 0.128618 | 0.015947 | 0.005435 | 0.130615 | 0.315653 | -0.000000 | `router_under_switches_despite_headroom` |
| `h50_history_only` | 11538 | 0.035246 | -0.002457 | 0.162816 | 0.223436 | 0.164847 | 0.260252 | 0.216169 | 0.316259 | 0.013345 | `unsafe_or_uncalibrated_switching` |
| `h50_motion_goal_context` | 11538 | 0.033450 | -0.001980 | 0.086985 | 0.221876 | 0.098371 | 0.232599 | 0.220706 | 0.319033 | 0.008696 | `unsafe_or_uncalibrated_switching` |

## Gate

| gate | pass |
| --- | ---: |
| `source_level_split_loaded` | True |
| `target_horizons_audited` | True |
| `all_candidates_audited` | True |
| `oracle_headroom_measured` | True |
| `router_capture_measured` | True |
| `blocker_diagnosis_present` | True |
| `source_breakdown_present` | True |
| `negative_result_not_overclaimed` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- Stage42-IP is a fresh diagnostic follow-up to Stage42-IO.
- It does not promote sequence/graph context to a t50/t100 main contribution.
- If oracle headroom is small, the candidate context itself is weak under this protocol.
- If oracle headroom exists but capture is low, the next repair should target switchability/calibration rather than more raw context features.
- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.
