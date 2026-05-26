# Stage42-AH Post-Repair Stress / Paper-Claim Refresh

- source: `fresh_synthesis_from_stage42ag_post_repair_stress`
- generated_at_utc: `2026-05-26T06:34:25.992662+00:00`
- git_commit: `adc5d74`
- gate: `11 / 11`
- verdict: `stage42_ah_post_repair_claim_refresh_pass`

## Current Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AH 是 post-repair stress / paper-claim refresh，不重新训练大模型。
- Stage42-AH 读取 Stage42-AG fresh report 并刷新可写 claim 与剩余 limitation。
- Future waypoints/endpoints 只作为 labels/eval，不作为 inference input。
- t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。
- External coordinates remain dataset-local / unverified weak metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Global Post-Repair Summary

- ADE all CI low: `0.08525846671997603`
- ADE t50 CI low: `0.05851255877278698`
- ADE hard/failure CI low: `0.08976713874728331`
- easy degradation CI high: `0.00334819268959823`
- FDE@50 CI low: `0.14823015795452749`

## Claim Matrix

| claim | status | evidence |
| --- | --- | --- |
| Global protected row-level full-waypoint ADE remains positive after AF/AG repairs. | `supported` | ADE all=0.091656, CI low=0.085258 |
| Global protected row-level t50 remains positive after AF/AG repairs. | `supported` | ADE t50=0.064957, CI low=0.058513; FDE@50=0.177856 |
| Horizon=25 negative slice from Stage42-AE is repaired to non-harm floor. | `supported_as_non_harm_not_positive_dynamics` | before=-0.004781149088858072, after=0.0 |
| ETH_UCY t50/FDE@50 lower-bound weakness from Stage42-AF is repaired. | `supported` | ADE@50 CI low -0.013218100958604987 -> 0.002820688160982139; FDE@50 CI low -0.04199023614248535 -> 0.021040393452369632 |
| t100 can be written as a uniformly deployable long-horizon result. | `rejected` | t100 remains raw-frame diagnostic; some t100 slices retain easy-degradation safety limits. |
| Metric or seconds-level pedestrian claims are allowed. | `rejected` | Stage42-AD global_metric_claim_allowed=False; global_seconds_claim_allowed=False |
| True 3D / foundation / Stage5C / SMC claims are allowed. | `rejected` | Stage5C not executed; SMC disabled; current model remains protected dataset-local raw-frame 2.5D. |
| Stage42-AE weak-slice limitations have been fully erased. | `partially_repaired_not_fully_erased` | AE weak findings were ['ETH_UCY FDE@50 seed-CI lower bound is not positive; keep FDE@50 as stress evidence, not universal guarantee.', 'ETH_UCY t50 seed-CI lower bound is not positive; write t50 domain claim with caution.', 'horizon=25 has non-positive ADE lower bound or mean; Stage42-X is not uniformly positive across every horizon slice.']; AF/AG repair horizon=25 and ETH_UCY|50, but t100 diagnostic/safety limits remain. |

## Slice Status

| slice | status | ADE all low | ADE t50 low | FDE@50 low | hard low | easy high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY|10` | `positive_supported` | 0.096360 | 0.000000 | 0.000000 | 0.109796 | 0.000000 |
| `ETH_UCY|100` | `positive_supported` | 0.009570 | 0.000000 | 0.000000 | 0.009570 | 0.000000 |
| `ETH_UCY|25` | `floor_non_harm` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `ETH_UCY|50` | `positive_supported` | 0.002821 | 0.002821 | 0.021040 | 0.002821 | 0.000000 |
| `TrajNet|10` | `positive_supported` | 0.090334 | 0.000000 | 0.000000 | 0.100413 | 0.000000 |
| `TrajNet|100` | `safety_limited` | 0.018233 | 0.000000 | 0.000000 | 0.018233 | 0.084984 |
| `TrajNet|25` | `floor_non_harm` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `TrajNet|50` | `positive_supported` | 0.077573 | 0.077573 | 0.207428 | 0.077573 | 0.011380 |
| `UCY|10` | `positive_supported` | 0.225801 | 0.000000 | 0.000000 | 0.231635 | 0.000000 |
| `UCY|100` | `positive_supported` | 0.157699 | 0.000000 | 0.000000 | 0.157699 | 0.000000 |
| `UCY|25` | `floor_non_harm` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `UCY|50` | `positive_supported` | 0.048050 | 0.048050 | 0.218371 | 0.048050 | 0.000000 |

## Remaining Limitations

- TrajNet|100 remains safety_limited: all_low=0.018233, hard_low=0.018233, fde50_low=0.000000, easy_high=0.084984.
- ETH_UCY|25, TrajNet|25, UCY|25 are floor/non-harm slices, not positive dynamics contributions.
- Metric claim remains blocked until source-specific homography direction, coordinate convention, scale, FPS, and stride are verified.
- Seconds-level horizon claim remains blocked; t50/t100 stay raw-frame horizons.

## Conclusion

Stage42-AH updates the paper-claim boundary after AF/AG repairs. The former horizon=25 negative slice is now floor/non-harm, and the ETH_UCY t50/FDE@50 lower-bound weakness is repaired. The correct claim is stronger than Stage42-AE, but still bounded: t100 remains raw-frame diagnostic, some t100 safety slices remain limited, and metric/seconds/true-3D/foundation claims remain rejected.
