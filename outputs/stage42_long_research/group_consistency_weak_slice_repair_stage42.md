# Stage42-HQ UCY Weak-Slice Group-Consistency Repair

- source: `fresh_stage42_hq_ucy_weak_slice_repair`
- generated_at_utc: `2026-05-27T19:07:55.328937+00:00`
- git_commit: `cc62d52`
- input_hash: `f49649f011e6b1729c69b43dd85b08d2eb4170e9e2f3cdc9ecde0f3a4939357d`
- gate: `23 / 23`
- verdict: `stage42_hq_group_consistency_weak_slice_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HQ 针对 Stage42-HP 暴露的 UCY 0-gain weak slice 做 fresh UCY-internal-validation-supported group-consistency repair。
- HQ 不用 test metrics 调阈值；UCY support 来自 original train sources carved internal validation。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## HP Weak Slice Before

- HP UCY rows: `9540`
- HP UCY all: `0.00%`
- HP UCY t50: `0.00%`
- HP UCY reason: `non_positive_all; non_positive_t50`

## Fresh Repair Result

| metric | global | UCY | TrajNet |
| --- | ---: | ---: | ---: |
| all | 32.89% | 35.58% | 32.07% |
| t50 | 26.99% | 22.72% | 28.18% |
| t100 raw diag | 21.12% | 27.56% | 19.01% |
| hard/failure | 31.89% | 33.78% | 31.29% |
| easy degradation | -32.09% | -40.60% | -30.55% |
| switch | 71.90% | 63.21% | 74.08% |

## Safety And Remaining Risk

- near@0.05 base/final/floor: `2.08%` / `1.31%` / `2.24%`
- t100 easy rows: `975`
- t100 easy degradation after repair: `2.56%`
- t100 remains raw-frame diagnostic; if this slice is used as a main claim, it needs a separate validation-only easy guard.

## Interpretation

- Stage42-HQ directly addresses the Stage42-HP UCY zero-gain weak slice.
- UCY repair is supported by train-only internal validation, not test-threshold tuning.
- This is still protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true-3D, Stage5C, or SMC evidence.
