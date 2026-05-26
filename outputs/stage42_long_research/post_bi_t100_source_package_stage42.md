# Stage42-BJ Post-BI T100 Source Package

- source: `fresh_post_bi_t100_source_package`
- generated_at_utc: `2026-05-26T13:22:34.860624+00:00`
- git_commit: `260a4ad`
- input_hash: `f5ddf095e6062ff59383bfbc2abacce8705805971c459a5c8b66b07b6b3c3c5a`
- gate: `14 / 14`
- verdict: `stage42_bj_post_bi_t100_source_package_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BI 已修复 UCY independent-source t100 easy guard，但这不是 global t100 success。
- ETH_UCY 和 TrajNet 仍缺足够 independent t100 sources，不能写全局 t100 positive claim。
- Stage42-BJ 不训练模型、不下载 gated/restricted raw data、不执行 Stage5C、不启用 SMC。
- future waypoints / endpoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic，不能写成 seconds-level。

## Domain Support Under Strict Post-BI Protocol

| domain | independent sources | needed | t100 supported | mean gain | min gain | max easy | blocker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | 1 | 2 | `False` | None | None | None | needs_2_additional_independent_t100_sources_under_strict_post_bi_protocol |
| `UCY` | 4 | 0 | `True` | 0.44591415479775254 | 0.42531297412698216 | 0.011339719285930428 | None |
| `TrajNet` | 0 | 3 | `False` | None | None | None | needs_3_additional_independent_t100_sources_under_strict_post_bi_protocol |

## Local Inventory Exhaustion

- raw_t100_capable_file_count: `8`
- t100_file_count_by_domain: `{'ETH_UCY': 2, 'UCY': 6, 'TrajNet': 0}`
- independent_t100_group_count_by_domain: `{'ETH_UCY': 1, 'UCY': 4, 'TrajNet': 0}`
- independent_t100_groups_by_domain: `{'ETH_UCY': ['ETH_UCY::ETH/seq_eth'], 'UCY': ['UCY::UCY/students01', 'UCY::UCY/students03', 'UCY::UCY/zara01', 'UCY::UCY/zara02'], 'TrajNet': []}`
- local_inventory_exhausted_for_domains: `['ETH_UCY', 'TrajNet']`

## Action Queue

### ETH_UCY

- priority: `critical`
- additional_independent_t100_sources_needed: `2`
- candidate_source_ids: `['ucy_crowd_original', 'trajnetpp_epfl_aicrowd', 'opentraj_toolkit', 'eth_ucy_original_sources']`
- next_action: `provide_or_approve_legal_independent_t100_sources_then_rerun_conversion_and_source_cv`

| candidate | official_url | local path found | auto download | blocked reasons |
| --- | --- | ---: | ---: | --- |
| `ucy_crowd_original` | `http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data` | `True` | `False` | ['requires user confirmation of official terms/login/challenge access', 'auto_download_allowed is false by policy'] |
| `trajnetpp_epfl_aicrowd` | `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/` | `True` | `False` | ['requires user confirmation of official terms/login/challenge access', 'auto_download_allowed is false by policy'] |
| `opentraj_toolkit` | `https://github.com/crowdbotp/OpenTraj` | `True` | `False` | ['auto_download_allowed is false by policy'] |
| `eth_ucy_original_sources` | `ETH/BIWI and UCY original dataset pages; source-specific terms must be manually verified` | `True` | `False` | ['requires user confirmation of official terms/login/challenge access', 'official source not fully resolved in Stage42-BC', 'auto_download_allowed is false by policy'] |

### TrajNet

- priority: `critical`
- additional_independent_t100_sources_needed: `3`
- candidate_source_ids: `['trajnetpp_epfl_aicrowd', 'opentraj_toolkit']`
- next_action: `provide_or_approve_legal_independent_t100_sources_then_rerun_conversion_and_source_cv`

| candidate | official_url | local path found | auto download | blocked reasons |
| --- | --- | ---: | ---: | --- |
| `trajnetpp_epfl_aicrowd` | `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/` | `True` | `False` | ['requires user confirmation of official terms/login/challenge access', 'auto_download_allowed is false by policy'] |
| `opentraj_toolkit` | `https://github.com/crowdbotp/OpenTraj` | `True` | `False` | ['auto_download_allowed is false by policy'] |

## Summary

- ucy_t100_repaired: `True`
- eth_ucy_additional_sources_needed: `2`
- trajnet_additional_sources_needed: `3`
- domains_still_blocked: `['ETH_UCY', 'TrajNet']`
- global_t100_positive_claim_allowed: `False`
- next_stage_recommended: `Stage42-BK legal source acquisition / user path verification for ETH_UCY and TrajNet independent t100 sources`

## Interpretation

- Stage42-BI fixed the UCY independent-source t100 easy blocker, but global t100 remains blocked.
- Under the stricter post-BI protocol, ETH_UCY needs two additional independent t100 sources and TrajNet needs three.
- The current local inventory is exhausted for the blocked independent-source requirements; next progress requires legal official sources or user-provided paths.
- No Stage5C, no SMC, no metric/seconds-level claim, and no auto-download of gated/restricted raw data occurred.
