# Stage42-BC T100 Source Acquisition Plan

- source: `fresh_synthesis_from_stage42_bb_plus_official_web_pages`
- generated_at_utc: `2026-05-26T12:04:15.083148+00:00`
- git_commit: `d64da9c`
- input_hash: `797ef87e226d38d7265b3fed76fa585e0823431b55f47db6026b657a4d201614`
- gate: `11 / 11`
- verdict: `stage42_bc_t100_source_acquisition_plan_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BC 是 Stage42-BB 后的 t100 source acquisition planner，不训练模型、不执行 Stage5C、不启用 SMC。
- t100 positive gain 仍缺独立 train-only source-CV 支持。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 所有 pedestrian / top-down claims 继续保持 raw-frame / dataset-local，除非未来官方 FPS/stride/homography/scale 验证完成。

## Official Web Sources Used

| source | url | retrieval_date | summary |
| --- | --- | --- | --- |
| AerialMPT DLR official page | `https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/aerialmpt-a-dataset-for-pedestrian-tracking-in-aerial-imagery` | `2026-05-26` | DLR reports 14 aerial pedestrian tracking sequences, 307 frames, co-registered/georeferenced crops, 2 fps, 2,528 pedestrians, and 44,740 point annotations. Public download is about 75 MB. |
| VITA EPFL datasets page | `https://www.epfl.ch/labs/vita/datasets/` | `2026-05-26` | Lists TrajNet++ as an interaction-centric human trajectory forecasting benchmark. |
| TrajNet++ EPFL open-research page | `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/` | `2026-05-26` | EPFL describes TrajNet++ as an open trajectory forecasting challenge with accompanying data and evaluation code, curated categories, and reproducible sampling; challenge/data access is linked through AIcrowd. |

## Candidate Source Ranking

| id | dataset | priority | score | local path | t100 value | auto download | reason |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `ucy_crowd_original` | UCY Crowd | A | 87 | `True` | high | `False` | adds the missing independent UCY t100-capable original-train source if legally available |
| `trajnetpp_epfl_aicrowd` | TrajNet++ | A | 77 | `True` | high | `False` | adds/validates independent trajectory sources for TrajNet and possibly ETH/UCY-style splits |
| `opentraj_toolkit` | OpenTraj toolkit | B | 70 | `True` | medium | `False` | source discovery/loader hub; may expose additional legal source files already local |
| `eth_ucy_original_sources` | ETH/UCY original pedestrian sources | B | 67 | `True` | high | `False` | highest-priority source-level repair target for ETH_UCY and UCY t100 support |
| `aerialmpt_dlr` | AerialMPT | C | 42 | `True` | low | `False` | useful aerial calibration/scene diagnostic; likely too short for robust t100 raw-frame support because only 307 frames across 14 sequences |
| `sdd_stanford` | Stanford Drone Dataset | C | 35 | `True` | low_for_external_high_for_sdd | `False` | SDD-specific pixel raw-frame benchmark; not external source-CV repair unless a separate cross-domain protocol is designed |

## Summary

- bb_unsupported_t100_domains: `['ETH_UCY', 'TrajNet', 'UCY']`
- candidate_sources: `6`
- official_sources_found: `5`
- local_path_found_sources: `6`
- high_priority_sources: `['ucy_crowd_original', 'trajnetpp_epfl_aicrowd', 'opentraj_toolkit', 'eth_ucy_original_sources']`
- auto_download_allowed_sources: `[]`
- auto_download_executed: `False`
- user_action_required_count: `6`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`

## Interpretation

- Stage42-BC does not download restricted/gated data and does not bypass license terms.
- Highest-priority t100 repair path is legal independent TrajNet++ / ETH-UCY / UCY source support plus rerunning train-only source-CV.
- AerialMPT is official and locally useful as aerial calibration/diagnostic context, but its 307 frames across 14 sequences make it unlikely to solve t100 support alone.
- TGSIM/traffic-style metric diagnostics must not be converted into pedestrian/top-down world-model success claims.
