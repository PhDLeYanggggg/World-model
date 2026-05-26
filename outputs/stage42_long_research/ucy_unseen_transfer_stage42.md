# Stage42-T UCY Unseen-Domain Transfer Attempt

- source: `fresh_run`
- generated_at_utc: `2026-05-26T03:37:21.433327+00:00`
- git_commit: `b1a931b`
- gate: `8 / 11`
- verdict: `stage42_t_ucy_transfer_blocked_no_candidate_predictions`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-T 只修复/诊断 UCY fallback-only 问题，不执行 Stage5C 或 SMC。
- unseen-domain transfer rule 只从 validation domains 推导；UCY test 只最终评估一次。
- future waypoints / endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。
- 当前 row cache 是本地 derived cache，不提交 GitHub。
- 如果 UCY 无可用非 floor prediction source，必须标 blocker，不包装成成功。

## Domain Coverage

- validation domains: `{'ETH_UCY': 16103, 'TrajNet': 37153}`
- test domains: `{'ETH_UCY': 25901, 'TrajNet': 20087, 'UCY': 9540}`
- unseen test domains: `['UCY']`

## UCY Test-Once Evaluation

| metric | mean | ci_low | ci_high |
| --- | ---: | ---: | ---: |
| ADE all | 0.000000 | 0.000000 | 0.000000 |
| ADE t50 | 0.000000 | 0.000000 | 0.000000 |
| ADE hard/failure | 0.000000 | 0.000000 | 0.000000 |
| ADE easy degradation | 0.000000 | 0.000000 | 0.000000 |
| FDE t50 | 0.000000 | 0.000000 | 0.000000 |
| switch rate | 0.000000 | 0.000000 | 0.000000 |

## Available Source Oracle Diagnostic

- any_available_nonfloor_prediction: `False`
- This diagnostic is test-only for blocker analysis, not policy selection.

## Interpretation

- UCY remains fallback-only because the current Stage42-R row cache contains no non-floor Stage42-J/P predictions for UCY.
- A validation-only unseen-domain transfer rule cannot create positive UCY transfer without a candidate source that actually switches or changes UCY rows.
- The next aligned action is to train/cache a UCY-aware or source-agnostic prediction source using train/validation only, or rebuild splits so UCY has legal calibration support.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
