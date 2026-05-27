# Stage42-HR Group-Consistency T100 Easy Guard

- source: `fresh_stage42_hr_validation_only_t100_easy_guard`
- generated_at_utc: `2026-05-27T19:14:45.492730+00:00`
- git_commit: `fef6bca`
- input_hash: `ed29defa3112a32ed254e95ad65bdb83b48374a0ac54daf0f14b897e40009424`
- gate: `23 / 23`
- verdict: `stage42_hr_t100_easy_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HR 针对 Stage42-HQ 暴露的 t100 easy degradation 做 validation-only domain|t100 guard。
- HR 不用 test metrics 调阈值；domain|t100 是否保留只由 validation all gain 和 validation easy degradation 决定。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Before / After

| metric | HQ before | HR after |
| --- | ---: | ---: |
| all | 32.89% | 27.72% |
| t50 | 26.99% | 26.99% |
| t100 raw diagnostic | 21.12% | 6.79% |
| hard/failure | 31.89% | 25.93% |
| easy degradation | -32.09% | -32.33% |
| t100 easy degradation | 2.56% | -0.31% |
| switch | 71.90% | 68.16% |

## Validation-Only Decisions

- threshold: `0.0`
- guarded_slices: `{'TrajNet|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'TrajNet', 'val_rows': 1160, 'test_rows': 5608, 'val_all_improvement': 0.23260462520508085, 'val_easy_degradation': 0.017118176622190173, 'threshold': 0.0, 'keep': False, 'reason': 'validation_easy_degradation_above_threshold_or_nonpositive_gain'}}`
- kept_slices: `{'UCY|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'UCY', 'val_rows': 1440, 'test_rows': 1440, 'val_all_improvement': 0.27564518723015075, 'val_easy_degradation': -0.021788147627511134, 'threshold': 0.0, 'keep': True}}`

## By Domain After Guard

| domain | rows | all | t50 | t100 raw | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 25.33% | 28.18% | 0.00% | 23.45% | -30.83% | 69.41% |
| `UCY` | 9540 | 35.58% | 22.72% | 27.56% | 33.78% | -40.60% | 63.21% |

## Interpretation

- Stage42-HR repairs the HQ t100 easy-safety weak slice using validation-only domain|t100 decisions.
- The t100 result remains raw-frame diagnostic and must not be described as seconds-level long horizon.
- This step does not execute Stage5C, does not enable SMC, and does not make metric/foundation/true-3D claims.
