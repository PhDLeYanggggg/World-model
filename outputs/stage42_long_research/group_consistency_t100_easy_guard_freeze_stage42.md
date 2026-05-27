# Stage42-HS T100 Easy Guard Freeze / Replay

- source: `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact`
- generated_at_utc: `2026-05-27T19:37:47.773405+00:00`
- git_commit: `3c934ac`
- input_hash: `25c0aec9647c35090ab84c8ca613a2277431f1ec96b2d50b3ef398932ce5e090`
- gate: `27 / 27`
- verdict: `stage42_hs_t100_easy_guard_freeze_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HS 冻结 Stage42-HR validation-only domain|t100 easy guard policy。
- HS 不重新调阈值，不使用 test metrics 做 policy decision；只把 HR fresh artifact 固化为轻量 policy/replay 证据。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Frozen Policy

- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_t100_easy_guard_policy_stage42.json`
- policy hash: `8dcc60f145df211084868a57b57246b69364adf51add1578c88cd012a6121e6e`
- guarded_slices: `{'TrajNet|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'TrajNet', 'val_rows': 1160, 'test_rows': 5608, 'val_all_improvement': 0.23260462520508085, 'val_easy_degradation': 0.017118176622190173, 'threshold': 0.0, 'keep': False, 'reason': 'validation_easy_degradation_above_threshold_or_nonpositive_gain'}}`
- kept_slices: `{'UCY|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'UCY', 'val_rows': 1440, 'test_rows': 1440, 'val_all_improvement': 0.27564518723015075, 'val_easy_degradation': -0.021788147627511134, 'threshold': 0.0, 'keep': True}}`

## Replay

- decision_table_exact_replay: `True`
- metric_summary_exact_replay: `True`
- max_metric_abs_diff: `0.0`

## Guarded Metrics

| metric | value |
| --- | ---: |
| all | 27.72% |
| t50 | 26.99% |
| t100 raw diagnostic | 6.79% |
| hard/failure | 25.93% |
| easy degradation | -32.33% |
| t100 easy degradation | -0.31% |
| switch | 68.16% |

## Interpretation

- HS freezes the HR t100 easy guard as a lightweight deployment/paper artifact.
- It does not rerun training, retune thresholds, execute Stage5C, enable SMC, or make metric/seconds-level claims.
- The t100 result remains raw-frame diagnostic; the primary value is safety: t100 easy harm is guarded while all/t50/hard remain positive.
