# Stage42-AE Unified Row-Level Cache Stress Audit

- source: `fresh_synthesis_from_stage42x_row_level_cache`
- generated_at_utc: `2026-05-26T06:09:21.685544+00:00`
- git_commit: `e66df52`
- input_hash: `68faea82d2d7b7525425f5a1391afb11a5d44e572cf97860614f139dbf05c47d`
- gate: `12 / 12`
- verdict: `stage42_ae_unified_row_cache_stress_pass_with_limitations`

## Current Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AE 是 Stage42-X row-level cache stress audit，不重新训练模型，不读取/提交 raw data。
- Stage42-X 的 future waypoints/endpoints 只作为 labels/eval，不作为 inference input。
- t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。
- External coordinates remain dataset-local / unverified weak metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Stage42-X Reference

- cache_hash: `ffa31b2525fa1a10db356ac5b1ef78602e44bc6f065c63cfc05ac29083e08937`
- Stage42-X gate: `16 / 16`
- ADE all seed mean: `0.0900136608879362`
- ADE t50 seed mean: `0.06109367671246102`
- ADE t50 seed CI low: `0.05367075264893123`
- ADE hard/failure seed mean: `0.09374591375146946`
- easy degradation CI high: `0.0032618559800165533`

## Per-Domain Stress

| domain | rows | ADE all | ADE all low | ADE t50 | ADE t50 low | hard | hard low | easy high | FDE t50 | FDE t50 low |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.042817 | 0.021917 | 0.017093 | -0.013218 | 0.044320 | 0.022912 | 0.004778 | 0.059435 | -0.041990 |
| `TrajNet` | 20087 | 0.102635 | 0.056281 | 0.097465 | 0.073102 | 0.108920 | 0.062801 | 0.010288 | 0.235413 | 0.201139 |
| `UCY` | 9540 | 0.196091 | 0.107935 | 0.122892 | 0.031230 | 0.207360 | 0.114014 | 0.000000 | 0.292236 | 0.201771 |

## Per-Horizon Stress

| horizon | rows | ADE all | ADE all low | hard | hard low | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 16726 | 0.201453 | 0.157234 | 0.212007 | 0.165025 | 0.372813 |
| `25` | 15208 | -0.004781 | -0.009699 | -0.009893 | -0.020857 | 0.013239 |
| `50` | 13689 | 0.061094 | 0.053671 | 0.061094 | 0.053671 | 0.273309 |
| `100` | 9905 | 0.081533 | 0.052781 | 0.081533 | 0.052781 | 0.276325 |

## Leave-One-Domain Stress

This is a row-count weighted diagnostic over Stage42-X per-domain means, not a new raw-row bootstrap.

| held out | kept domains | ADE all | ADE t50 | hard | easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `TrajNet, UCY` | 0.132728 | 0.105653 | 0.140618 | 0.003087 | 0.253710 |
| `TrajNet` | `ETH_UCY, UCY` | 0.084075 | 0.045572 | 0.088207 | 0.001180 | 0.122100 |
| `UCY` | `ETH_UCY, TrajNet` | 0.068945 | 0.052198 | 0.072537 | 0.002898 | 0.136300 |

## Findings

- strong_domains: `ETH_UCY, TrajNet, UCY`
- weak_domains: `ETH_UCY`
- strong_horizons: `10, 50, 100`
- weak_horizons: `25`

## Limitations To Write In Paper

- ETH_UCY FDE@50 seed-CI lower bound is not positive; keep FDE@50 as stress evidence, not universal guarantee.
- ETH_UCY t50 seed-CI lower bound is not positive; write t50 domain claim with caution.
- horizon=25 has non-positive ADE lower bound or mean; Stage42-X is not uniformly positive across every horizon slice.

## Conclusion

Stage42-AE strengthens the Stage42-X paper evidence by explicitly stress-testing where the unified row-level full-waypoint cache is stable and where it is not. The global Stage42-X t50 seed and bootstrap lower bounds remain positive, and at least two domains have strong all/hard/easy stress evidence. However, the claim must not be written as uniformly positive across every domain/horizon/FDE slice: ETH_UCY t50/FDE@50 has weak lower bounds and horizon=25 remains a limitation slice. Claims remain protected dataset-local raw-frame 2.5D; Stage5C and SMC remain disabled.
