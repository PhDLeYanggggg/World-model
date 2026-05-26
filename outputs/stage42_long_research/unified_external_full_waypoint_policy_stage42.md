# Stage42-W Unified External Full-Waypoint Policy Package

- source: `fresh_unified_from_cached_verified_stage42s_and_stage42v`
- generated_at_utc: `2026-05-26T04:31:08.023419+00:00`
- git_commit: `5f86b10`
- input_hash: `32df67bd086ca4748f9c9fcbf3708725ff37125964e9c47b3ea1b32325e24d4c`
- policy_hash: `a2439e23c0c2e3f7aa99efa8a84e42868ea52258394ce41339c96ee0a2ec910e`
- gate: `16 / 16`
- verdict: `stage42_w_unified_external_full_waypoint_policy_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-W 只合并已验证 external full-waypoint policy sources；不是 metric 或 seconds-level 结果。
- ETH_UCY / TrajNet 来自 Stage42-S row-cache combo policy；UCY 来自 Stage42-V strict pure-UCY full-waypoint candidate 的 UCY-domain slice。
- Stage42-W 不把 Stage42-V 的 ETH_UCY slice 重复计入，避免 double counting。
- future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。
- policy source selection 来自 validation-only / source-heldout protocol；不使用 test 调阈值。
- merged single row-cache artifact 尚未建立；本阶段输出 unified policy package 和 per-domain stress。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Source Package

- `ETH_UCY`: Stage42-S row-cache combo policy.
- `TrajNet`: Stage42-S row-cache combo policy.
- `UCY`: Stage42-V strict pure-UCY full-waypoint candidate, UCY-domain slice only.
- Stage42-V `ETH_UCY` slice is excluded to avoid double counting with Stage42-S.
- A single merged row-cache artifact is not yet built; this report is a unified policy package with per-domain stress.

## Weighted Package Summary

- rows: `55528`
- CI note: Global row-level bootstrap was not rerun because Stage42-S and Stage42-V are separate validated sources; per-domain CIs are reported and global means are row-weighted.

| metric | weighted mean | domain min CI low | domain max CI high |
| --- | ---: | ---: | ---: |
| ADE all | 0.099339 | 0.021917 | 0.330147 |
| ADE t50 | 0.093998 | -0.013218 | 0.378387 |
| ADE t100 raw-frame diagnostic | 0.084776 | 0.000135 | 0.282687 |
| ADE hard/failure | 0.104867 | 0.022912 | 0.349043 |
| ADE easy degradation | 0.002400 | -0.001550 | 0.010288 |
| FDE t50 | 0.168792 | -0.041990 | 0.410201 |
| switch rate | 0.232591 | 0.114942 | 0.525752 |

## Per-Domain Metrics

| domain | source | rows | ADE all | ADE t50 | ADE hard | easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `Stage42-S` | 25901 | 0.042817 | 0.017093 | 0.044320 | 0.001614 | 0.059435 |
| `TrajNet` | `Stage42-S` | 20087 | 0.102635 | 0.097465 | 0.108920 | 0.004552 | 0.235413 |
| `UCY` | `Stage42-V` | 9540 | 0.245852 | 0.295497 | 0.260718 | 0.000000 | 0.325422 |

## Interpretation

- Stage42-W closes the main Stage42-S UCY fallback-only gap at policy-package level by importing the Stage42-V UCY full-waypoint candidate source.
- The result is stronger external full-waypoint branch evidence across ETH_UCY, TrajNet, and UCY.
- It is not yet a single merged row-cache artifact; future work should build row-level UCY candidate cache and rerun one unified bootstrap.
- All claims remain dataset-local raw-frame 2.5D. No metric, seconds-level, Stage5C, or SMC claim is made.
