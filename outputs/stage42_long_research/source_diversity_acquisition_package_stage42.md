# Stage42-CD Source Diversity Acquisition Package

- source: `fresh_stage42_cd_source_diversity_acquisition_package`
- generated_at_utc: `2026-05-26T16:55:35.586098+00:00`
- git_commit: `20f0dad`
- input_hash: `9df7a4242dc8c00e23cede201859c2395d26601ac9cccadd5b2be2725e782106`
- gate: `13 / 13`
- verdict: `stage42_cd_source_diversity_acquisition_package_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CD 是官方源获取与接入准备包，不下载数据，不转换数据，不训练模型。
- Stage42-CB/CC 已证明 t50 source diversity blocker 仍 active；本包不能把 blocker 包装成完成。
- OpenTraj toolkit / wrapper 许可不能自动覆盖 ETH/UCY/TrajNet 等底层第三方数据许可。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C 未执行，SMC 未启用。

## Summary

- official_targets: `5`
- critical_targets: `2`
- auto_download_targets: `0`
- manual_or_terms_targets: `4`
- local_paths_found: `4`
- converted_datasets_now: `0`
- source_diversity_repair_ready_now: `False`
- broad_source_generalization_claim_allowed: `False`

## Official / Manual Targets

| id | priority | blocker | official URL | auto download | local path found | next claim status |
| --- | --- | --- | --- | ---: | ---: | --- |
| `ucy_crowd_original` | `critical` | `UCY_students_t50_source_support` | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | False | True | `blocked_user_action_required` |
| `eth_biwi_original` | `critical` | `ETH_seq_t50_source_support` | https://vision.ee.ethz.ch/datsets.html | False | True | `blocked_user_action_required` |
| `trajnetplusplus_official` | `high` | `TrajNet_raw_long_t100_source_support` | https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/ | False | True | `short_scene_diagnostic_only` |
| `opentraj_toolkit` | `medium` | `loader_and_metadata_support` | https://github.com/crowdbotp/OpenTraj | False | True | `toolkit_only_not_dataset_claim` |
| `aerialmpt_or_other_topdown` | `medium` | `external_source_diversity` | user_or_web_verified_official_url_required | False | False | `future_external_expansion_only` |

## Required Pipeline After User Provides Data

- run a source-specific conversion script for the provided path
- run no-leakage audit with source-level train/internal-val/final-test split
- run protected t50/t100 source-CV only after terms and split are verified
- update paper package only after fresh final-test evidence exists

## Interpretation

- Stage42-CD does not repair source diversity by itself.
- No auto-download was executed because all priority source-diversity targets need manual terms/path verification or are toolkit/challenge references.
- A file path, registry row, toolkit clone, or alternate representation is not counted as a converted/evaluated dataset.
- Broad source-level generalization remains blocked until fresh conversion/no-leakage/source-CV/final-test evidence exists.
