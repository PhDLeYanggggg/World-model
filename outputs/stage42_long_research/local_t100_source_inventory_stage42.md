# Stage42-BD Local T100 Source Inventory

- source: `fresh_local_path_inventory`
- generated_at_utc: `2026-05-26T12:11:24.098182+00:00`
- git_commit: `499f885`
- input_hash: `1492f03e4a350624a4903452390f949203a8dcef83277764a60545fb68f815bf`
- gate: `10 / 10`
- verdict: `stage42_bd_local_t100_source_inventory_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BD 是本机 local t100 source inventory，不训练模型、不下载数据。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 本步骤只识别可转换候选；是否纳入 official evaluation 还需要后续 conversion/no-leakage/source-CV。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- files_scanned: `93`
- parseable_files: `74`
- t100_capable_files: `8`
- already_used_t100_files: `4`
- novel_t100_candidate_files: `4`
- estimated_novel_t100_windows: `6257`
- stage42_be_conversion_recommended: `True`

## Domain Summary

| domain | files | t100 capable | already used | novel candidates | est novel t100 windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 11 | 2 | 1 | 1 | 14 |
| `TrajNet` | 60 | 0 | 0 | 0 | 0 |
| `UCY_or_ETH_UCY` | 22 | 6 | 3 | 3 | 6243 |

## Top Novel T100 Candidates

| path | domain | rows | agents | max track | est t100 windows | next step |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ETH/seq_eth/biwi_eth_10fps.txt` | ETH_UCY | 5492 | 360 | 114 | 14 | candidate_for_stage42_be_conversion_and_train_only_source_cv |
| `UCY/students01/students001.txt` | UCY_or_ETH_UCY | 21813 | 415 | 352 | 1949 | candidate_for_stage42_be_conversion_and_train_only_source_cv |
| `UCY/students03/obsmat_px.txt` | UCY_or_ETH_UCY | 21859 | 428 | 540 | 3415 | candidate_for_stage42_be_conversion_and_train_only_source_cv |
| `UCY/students03/students003.txt` | UCY_or_ETH_UCY | 17953 | 434 | 289 | 879 | candidate_for_stage42_be_conversion_and_train_only_source_cv |

## Interpretation

- Stage42-BD found local candidate files; this is not yet conversion, training, or official evaluation.
- Any novel t100 candidates must go through Stage42-BE conversion, no-leakage split construction, and train-only source-CV before t100 claims can change.
- Synthetic/diagnostic files are excluded from real external t100 repair claims unless a separate diagnostic-only protocol is written.
