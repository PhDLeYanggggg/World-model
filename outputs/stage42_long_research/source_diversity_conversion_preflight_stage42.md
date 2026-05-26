# Stage42-CE Source Diversity Conversion Preflight

- source: `fresh_stage42_ce_source_diversity_conversion_preflight`
- generated_at_utc: `2026-05-26T17:02:36.785402+00:00`
- git_commit: `c7ec174`
- input_hash: `44de14580b13e621c06b186c74d89cec522f0b6ffe10a1336a010040a7fe8a6a`
- gate: `12 / 12`
- verdict: `stage42_ce_source_diversity_conversion_preflight_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CE 是 local path conversion preflight，不转换数据，不训练模型，不调 threshold。
- 本轮只检查本地路径结构和 track-like 文件；不绕过 license，不下载数据。
- local path found 不等于 legal / converted / evaluated。
- alternate representation 不等于 independent held-out source。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C 未执行，SMC 未启用。

## Summary

- targets_checked: `5`
- targets_with_local_path: `4`
- targets_with_schema_possible: `4`
- targets_with_t50_files: `3`
- targets_with_t100_files: `3`
- targets_with_independent_t50_candidates: `0`
- targets_source_cv_ready_now: `0`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- source_diversity_repair_ready_now: `False`

## Target Preflight Table

| target | local path | schema possible | t50 files | t100 files | independent t50 | legal blocked | source-CV ready | next action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ucy_crowd_original` | True | True | 6 | 6 | 0 | True | False | verify/accept official dataset terms before conversion; local parseability is not legal permission |
| `eth_biwi_original` | True | True | 3 | 2 | 0 | True | False | verify/accept official dataset terms before conversion; local parseability is not legal permission |
| `trajnetplusplus_official` | True | True | 0 | 0 | 0 | True | False | verify/accept official dataset terms before conversion; local parseability is not legal permission |
| `opentraj_toolkit` | True | True | 270 | 270 | 0 | False | False | rebuild source split or provide independent source; current files are already-used, alternate, short, or diagnostic |
| `aerialmpt_or_other_topdown` | False | False | 0 | 0 | 0 | True | False | provide a legal local path containing parseable trajectory rows |

## Interpretation

- Stage42-CE is a local conversion preflight, not conversion.
- Local parseability and local path existence are not legal permission.
- No target is counted as source-diversity repair in this stage.
- Any future conversion must rebuild source-level split and rerun no-leakage/source-CV/final test.
