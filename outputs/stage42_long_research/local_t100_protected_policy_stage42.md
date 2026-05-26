# Stage42-BG Local T100 Protected Policy Source-CV

- source: `fresh_source_cv_protected_policy`
- generated_at_utc: `2026-05-26T12:44:04.036036+00:00`
- git_commit: `ffe6c51`
- input_hash: `7cf20945b6a10fe3e70dc474cc1861e9c2987b61f75723c53f6916d40b3bd1d0`
- gate: `13 / 13`
- verdict: `stage42_bg_local_t100_protected_policy_pass_with_global_t100_blocker`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BG 使用 Stage42-BF 的本地 t100 in-memory conversion 结果继续做 protected policy source-CV。
- 本步骤只训练/选择 baseline-family policy，不训练神经模型，不执行 Stage5C，不启用 SMC。
- policy threshold / baseline choice 只从 train/validation source 选择，holdout source 只评估一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic；即使 UCY source-CV positive，也不能写成全局 t100 已修复。

## Summary

- candidate_sources: `4`
- t50_policy_windows: `15058`
- t100_policy_windows: `6071`
- source_cv_domains_evaluated: `UCY`
- source_cv_domains_blocked: `ETH_UCY`
- ucy_t100_source_cv_supported: `True`
- ucy_t100_mean_improvement_vs_fallback: `0.44093796402512603`
- ucy_t100_min_improvement_vs_fallback: `0.43857862817152227`
- ucy_t100_max_easy_degradation: `0.011339719285930428`
- global_t100_positive_claim_allowed: `False`

## Domain Summary

| domain | horizon | folds | safe folds | mean improvement | min improvement | max easy degradation | all safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UCY` | 50 | 3 | 0 | 0.120466 | 0.000000 | 0.185291 | False |
| `UCY` | 100 | 3 | 3 | 0.440938 | 0.438579 | 0.011340 | True |

## Fold Details

| domain | holdout | horizon | selected policy | holdout rows | improvement | easy degradation | switch rate |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `UCY` | `UCY/students03/obsmat_px.txt` | 50 | `global_damped_velocity_0p50` | 6252 | 0.361399 | 0.185291 | 1.000000 |
| `UCY` | `UCY/students03/obsmat_px.txt` | 100 | `global_damped_velocity_0p50` | 3342 | 0.441619 | 0.011340 | 1.000000 |
| `UCY` | `UCY/students01/students001.txt` | 50 | `global_constant_velocity_causal_fd` | 6134 | 0.000000 | 0.000000 | 0.000000 |
| `UCY` | `UCY/students01/students001.txt` | 100 | `global_damped_velocity_0p50` | 1866 | 0.442617 | -0.076291 | 1.000000 |
| `UCY` | `UCY/students03/students003.txt` | 50 | `global_constant_velocity_causal_fd` | 2596 | 0.000000 | 0.000000 | 0.000000 |
| `UCY` | `UCY/students03/students003.txt` | 100 | `global_damped_velocity_0p50` | 851 | 0.438579 | -0.041436 | 1.000000 |

## Blockers

- `ETH_UCY`: `fewer_than_three_t100_capable_sources_or_no_source_cv_folds`; t100_capable_sources=1; estimated_t100_windows=14

## Interpretation

- Stage42-BG is stronger than BF: it selects a protected baseline-family policy on validation sources and evaluates once on held-out sources.
- UCY local t100 source-CV is positive and easy-safe under this limited protocol.
- This is still not a global t100 deployment claim because ETH_UCY is under-supported and TrajNet is not represented in these new local candidates.
- Stage5C remains unexecuted and SMC remains disabled.
