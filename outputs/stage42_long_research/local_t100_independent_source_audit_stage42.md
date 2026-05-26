# Stage42-BH Local T100 Independent Source Audit

- source: `fresh_local_independent_source_audit`
- generated_at_utc: `2026-05-26T12:56:51.629223+00:00`
- git_commit: `2c99af8`
- input_hash: `a0d83559b860e433050446597a0191a9f16def3f4952df7bff36564fd65afb04`
- gate: `13 / 14`
- verdict: `stage42_bh_independent_t100_source_audit_partial`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BH 审计本地 t100-capable 文件是否真的是独立 source，而不是同一 scene 的重复格式。
- 本步骤只做 local source independence audit 和 validation-selected protected baseline-family source-CV。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic；局部 UCY 支持不等于全局 t100 修复。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- raw_t100_capable_files: `8`
- independent_t100_sources: `5`
- duplicate_or_alternate_format_group_count: `2`
- ucy_independent_sources: `4`
- eth_ucy_independent_sources: `1`
- trajnet_independent_sources: `0`
- ucy_t100_source_cv_supported: `False`
- ucy_t100_mean_improvement_vs_fallback: `0.4834140197696444`
- ucy_t100_min_improvement_vs_fallback: `0.3405590303668396`
- ucy_t100_max_easy_degradation: `0.06332289296349253`
- blocked_domains: `ETH_UCY, TrajNet`
- global_t100_positive_claim_allowed: `False`

## Independent Source Support Matrix

| domain | independent sources | estimated t100 windows | source-CV feasible | blocker |
| --- | ---: | ---: | ---: | --- |
| `ETH_UCY` | 1 | 91 | False | `needs_2_additional_independent_t100_sources` |
| `UCY` | 4 | 7556 | True | `None` |
| `TrajNet` | 0 | 0 | False | `needs_3_additional_independent_t100_sources` |

## Duplicate / Alternate Format Groups

| independent key | chosen source | candidates | deduplicated |
| --- | --- | --- | ---: |
| `ETH_UCY::ETH/seq_eth` | `ETH/seq_eth/obsmat.txt` | `ETH/seq_eth/biwi_eth_10fps.txt, ETH/seq_eth/obsmat.txt` | True |
| `UCY::UCY/students01` | `UCY/students01/students001.txt` | `UCY/students01/students001.txt` | False |
| `UCY::UCY/students03` | `UCY/students03/obsmat_px.txt` | `UCY/students03/obsmat.txt, UCY/students03/obsmat_px.txt, UCY/students03/students003.txt` | True |
| `UCY::UCY/zara01` | `UCY/zara01/obsmat.txt` | `UCY/zara01/obsmat.txt` | False |
| `UCY::UCY/zara02` | `UCY/zara02/obsmat.txt` | `UCY/zara02/obsmat.txt` | False |

## Domain Source-CV Summary

| domain | horizon | folds | safe folds | mean improvement | min improvement | max easy degradation | all safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UCY` | 50 | 4 | 0 | 0.110345 | 0.000000 | 0.383753 | False |
| `UCY` | 100 | 4 | 3 | 0.483414 | 0.340559 | 0.063323 | False |

## Interpretation

- Counting files is not enough: alternate formats from the same scene/source are deduplicated before source-CV.
- UCY has enough independent local t100 sources and positive mean t100 gain, but it is not easy-safe under strict independent source-CV.
- The safe-switch repair reduced one large easy-harm fold, but `students03` still exceeds the 2% easy-degradation gate.
- ETH_UCY and TrajNet remain hard blockers for global t100 support.
- No metric/seconds-level, true-3D, Stage5C, or SMC claim is allowed.
