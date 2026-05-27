# Stage42-HT Runtime T100 Easy Guard Policy

- source: `fresh_runtime_api_from_frozen_stage42_hs_t100_easy_guard_policy`
- generated_at_utc: `2026-05-27T19:59:35.881981+00:00`
- git_commit: `c38c2b0`
- input_hash: `d4c1cd805eda97f17d4d094f87c5e8e07939532a11ec269a86aaed6e66d0a43f`
- gate: `19 / 19`
- verdict: `stage42_ht_t100_easy_guard_runtime_policy_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HT 把 Stage42-HS frozen t100 easy guard policy 变成可调用 runtime API。
- runtime policy 只使用 domain、horizon、候选 rollout 和 train-horizon causal floor rollout。
- 未知 domain 的 t100 默认回退 floor，因为没有 validation support。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Runtime Rule

- `TrajNet|100`: fallback to train-horizon causal floor.
- `UCY|100`: keep candidate rollout.
- unknown `domain|100`: fallback to floor because validation support is absent.
- non-t100 rows: unchanged candidate rollout.

## Smoke Replay

- passes: `True`
- actual_switch: `[False, True, True, False]`
- actual_reasons: `['validation_easy_harm_t100_fallback_floor', 'validation_supported_t100_keep_candidate', 'non_t100_not_guarded', 'unknown_t100_domain_no_validation_support_fallback_floor']`

## Inherited Metrics From Frozen HS Policy

| metric | value |
| --- | ---: |
| all | 27.72% |
| t50 | 26.99% |
| t100 raw diagnostic | 6.79% |
| hard/failure | 25.93% |
| easy degradation | -32.33% |
| t100 easy degradation | -0.31% |

## Interpretation

- HT makes the HS t100 guard callable at deployment/replay time.
- It does not retrain, retune thresholds, execute Stage5C, enable SMC, or make metric/seconds-level claims.
