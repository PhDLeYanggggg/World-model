# Stage42-CC Independent T50 Source Inventory

- source: `fresh_stage42_cc_independent_t50_source_inventory`
- generated_at_utc: `2026-05-26T16:43:38.704373+00:00`
- git_commit: `a38cdaf`
- input_hash: `0a595f7579adefe3f173615f8a9f2cc175f0f55d377cae02de624f23b97eec7c`
- gate: `10 / 10`
- verdict: `stage42_cc_independent_t50_source_inventory_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CC 是 independent t50 source inventory / user-action audit，不训练模型，不调 threshold。
- 本审计只扫描本地可见文件；不绕过 license，不自动下载，不把 registry-only 当 converted。
- 如果 source 已被当前 split 使用，只能作为 split rebuild/source-CV 候选，不能直接算新 held-out evidence。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C 未执行，SMC 未启用。

## Summary

- scanned_files: `93`
- t50_capable_files: `10`
- unused_candidate_t50_sources: `0`
- alternate_current_source_candidates: `4`
- diagnostic_t50_candidates: `1`
- source_diversity_repair_ready: `False`
- candidate_names: `[]`

## By Family

| family | files | t50 capable | unused candidates | alternates | diagnostic | already used | homography hints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 13 | 3 | 0 | 1 | 0 | 4 | 11 |
| `TrajNet` | 50 | 1 | 0 | 0 | 1 | 1 | 0 |
| `UCY_or_crowds` | 30 | 6 | 0 | 3 | 0 | 13 | 20 |

## Top Candidate / Blocker Rows

| source | family | rows | max track | t50 windows | status | next step |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `students03/obsmat_px.txt` | `UCY_or_crowds` | 21859 | 540 | 6493 | `alternate_representation_of_current_source` | same parent/source directory as current data; useful for split rebuild or format repair, not counted as independent source |
| `students03/obsmat.txt` | `UCY_or_crowds` | 21846 | 539 | 6491 | `already_in_current_combined_split` | can only be reused through a new train/val/test split rebuild, not counted as new held-out source |
| `students01/students001.txt` | `UCY_or_crowds` | 21813 | 352 | 6445 | `alternate_representation_of_current_source` | same parent/source directory as current data; useful for split rebuild or format repair, not counted as independent source |
| `zara02/obsmat.txt` | `UCY_or_crowds` | 9537 | 583 | 2823 | `already_in_current_combined_split` | can only be reused through a new train/val/test split rebuild, not counted as new held-out source |
| `students03/students003.txt` | `UCY_or_crowds` | 17953 | 289 | 2793 | `alternate_representation_of_current_source` | same parent/source directory as current data; useful for split rebuild or format repair, not counted as independent source |
| `synth_data/orca_circle_crossing_5ped.ndjson` | `TrajNet` | 240894 | 69 | 2269 | `diagnostic_or_simulation_only` | do_not_count_simulation_or_synthetic_as_real_external_source |
| `seq_eth/obsmat.txt` | `ETH_UCY` | 8908 | 190 | 291 | `already_in_current_combined_split` | can only be reused through a new train/val/test split rebuild, not counted as new held-out source |
| `zara01/obsmat.txt` | `UCY_or_crowds` | 5024 | 197 | 240 | `already_in_current_combined_split` | can only be reused through a new train/val/test split rebuild, not counted as new held-out source |
| `seq_hotel/obsmat.txt` | `ETH_UCY` | 6544 | 100 | 215 | `already_in_current_combined_split` | can only be reused through a new train/val/test split rebuild, not counted as new held-out source |
| `seq_eth/biwi_eth_10fps.txt` | `ETH_UCY` | 5492 | 114 | 82 | `alternate_representation_of_current_source` | same parent/source directory as current data; useful for split rebuild or format repair, not counted as independent source |
| `seq_eth/H.txt` | `ETH_UCY` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `seq_eth/destinations.txt` | `ETH_UCY` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `seq_eth/groups.txt` | `ETH_UCY` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `seq_eth/info.txt` | `ETH_UCY` | 0 | 0 | 0 | `not_track_source` | ignore_for_t50_source_diversity |
| `seq_hotel/H.txt` | `ETH_UCY` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `seq_hotel/destinations.txt` | `ETH_UCY` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `seq_hotel/groups.txt` | `ETH_UCY` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `seq_hotel/info.txt` | `ETH_UCY` | 0 | 0 | 0 | `not_track_source` | ignore_for_t50_source_diversity |
| `students01/students001-trajnet.txt` | `UCY_or_crowds` | 17820 | 20 | 0 | `already_in_current_combined_split` | can only be reused through a new train/val/test split rebuild, not counted as new held-out source |
| `students03/H-old.txt` | `UCY_or_crowds` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `students03/H.txt` | `UCY_or_crowds` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `students03/groups.txt` | `UCY_or_crowds` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `zara01/H-cam.txt` | `UCY_or_crowds` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `zara01/H.txt` | `UCY_or_crowds` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |
| `zara01/destinations.txt` | `UCY_or_crowds` | 0 | 0 | 0 | `auxiliary_metadata_not_track_source` | ignore_for_t50_source_diversity |

## Interpretation

- Stage42-CC is an inventory, not conversion or benchmark success.
- Candidate files require legal/terms verification, split rebuild, conversion, no-leakage audit, validation-only policy selection, and final test before any claim.
- Source diversity remains the next blocker exposed by Stage42-CB.
