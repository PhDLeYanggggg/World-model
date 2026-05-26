# Stage42-BF Local T100 Schema Conversion And Source-CV Baseline Audit

- source: `fresh_in_memory_schema_conversion`
- generated_at_utc: `2026-05-26T12:33:12.689637+00:00`
- git_commit: `f6e0265`
- input_hash: `22e46b6a2d4745dd439f05bb24396a10fd83bb885a7f79dc9ca236d222ee05e9`
- gate: `12 / 12`
- verdict: `stage42_bf_local_t100_schema_conversion_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BF 做 local t100 candidates 的 in-memory schema conversion 和 causal baseline/source-CV audit。
- 本步骤不提交 full feature store，不训练神经模型，不改变部署模型。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic / blocker，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- candidate_sources: `4`
- converted_sources: `4`
- t50_eval_windows: `15058`
- t100_eval_windows: `6071`
- source_cv_domains_evaluated: `ETH_UCY, UCY`
- source_cv_domains_positive_vs_constant_velocity: `UCY`
- materialized_feature_store_written: `False`
- t100_positive_claim_allowed: `False`

## Strongest Baseline By Source And Horizon

| source | domain | h10 | h25 | h50 | h100 |
| --- | --- | --- | --- | --- | --- |
| `ETH/seq_eth/biwi_eth_10fps.txt` | ETH_UCY | constant_velocity_causal_fd (0.000) | damped_velocity_0p50 (0.335) | constant_position (0.645) | constant_position (0.923) |
| `UCY/students01/students001.txt` | UCY | constant_velocity_causal_fd (0.000) | damped_velocity_0p75 (0.148) | damped_velocity_0p25 (0.361) | damped_velocity_0p25 (0.568) |
| `UCY/students03/obsmat_px.txt` | UCY | damped_velocity_0p75 (0.077) | damped_velocity_0p50 (0.139) | damped_velocity_0p25 (0.441) | constant_position (0.705) |
| `UCY/students03/students003.txt` | UCY | constant_velocity_causal_fd (0.000) | damped_velocity_0p75 (0.114) | damped_velocity_0p25 (0.335) | constant_position (0.692) |

## Source-CV Baseline Audit

| domain | folds | mean holdout improvement vs CV | min holdout improvement vs CV | all folds positive |
| --- | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 0 | NA | NA | False |
| `UCY` | 3 | 0.607043 | 0.491545 | True |

## Interpretation

- Stage42-BF performs an actual in-memory schema conversion and causal baseline audit, but does not write a large feature store.
- Future labels are used only to compute baseline errors; they are not inference inputs.
- Source-CV baseline selection is useful readiness evidence, not protected M3W policy training.
- t100 remains a raw-frame diagnostic / blocker until Stage42-BG trains/evaluates a protected policy on these converted sources.
