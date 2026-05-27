# Stage42-HV T100 Runtime Row-Cache Batch Replay

- source: `cached_verified_row_cache_runtime_batch_replay_from_stage42_hr_ht`
- generated_at_utc: `2026-05-27T20:24:37.082680+00:00`
- git_commit: `130ef59`
- input_hash: `9eb7611b8ea8881282c3105f05a463befc40afc33a41765abd541a9d6bb7ad13`
- cache_hash: `166fdede23d8f14bbf6eb4c0398b32b9c90d489a03d3e6e9acdbc608db5ed127`
- gate: `28 / 28`
- verdict: `stage42_hv_t100_runtime_row_cache_replay_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HV 修复 Stage42-HU 暴露的 row-level replay blocker：从 HR 的可重建路径生成本地 row-level replay cache。
- cache 是 derived local artifact，写在 data/ 下，不提交 GitHub。
- runtime replay 只使用 domain、horizon、candidate rollout、floor rollout 和 optional candidate switch。
- future waypoints / endpoints 只作为 evaluation labels 存储，不作为 runtime input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Row Cache

- cache_path: `data/stage42_t100_runtime_replay_cache/stage42hv_t100_runtime_replay_test_cache.npz`
- cache_status: `cached_verified`
- cache_size_bytes: `7576461`
- required fields present: `True`
- rows: `47458`
- domains: `{'TrajNet': 37918, 'UCY': 9540}`
- t100 rows: `7048`
- t100 easy rows: `975`

## Runtime Batch Replay Metrics

| metric | value |
| --- | ---: |
| all | 27.72% |
| t50 | 26.99% |
| t100 raw diagnostic | 6.79% |
| hard/failure | 25.93% |
| easy degradation | -32.33% |
| t100 easy degradation | -0.31% |
| switch rate | 68.16% |

## Exact Replay Checks

- selected_xy_max_abs_diff_vs_stored_hr: `0.0`
- selected_ade_max_abs_diff_vs_stored: `0.0`
- floor_ade_max_abs_diff_vs_stored: `0.0`
- switch_mismatch_vs_stored: `0`
- metric_diff_vs_hr: `{'all_improvement': 0.0, 't50_improvement': 0.0, 't100_raw_frame_diagnostic_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 't100_easy_degradation': 0.0, 'switch_rate': 0.0}`

## Interpretation

- Stage42-HV closes the Stage42-HU blocker locally by reconstructing a row-level replay cache from the HR rebuild path.
- The cache is intentionally not committed because it is derived row-level rollout data.
- The report and hashes are committed so reviewers can see what was replayed and why the claim remains bounded.
- This is real row-level runtime batch replay for the frozen t100 easy guard, not Stage5C, not SMC, not metric, and not seconds-level.
