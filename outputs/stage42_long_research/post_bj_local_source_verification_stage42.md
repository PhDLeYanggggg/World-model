# Stage42-BK Post-BJ Local Source Verification

- source: `fresh_post_bj_local_source_verification`
- generated_at_utc: `2026-05-26T13:29:08.075212+00:00`
- git_commit: `f32e126`
- input_hash: `61e5fc1e294b15e5affc6a3914caa7ee2951b1e8f9cad118bec8973cecc60a62`
- gate: `11 / 11`
- verdict: `stage42_bk_local_source_verification_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BK 是 post-BJ 本地 source/path verification 和 loader-gap audit，不训练模型。
- 本步骤检查本地 OpenTraj/ETH/UCY/TrajNet/ETH-Person 文件是否有 t100 conversion potential。
- 本步骤不会把本地路径存在写成 license 已确认，也不会把 conversion candidate 写成 evaluated source。
- future waypoints / endpoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Domain Summary

| domain | parsed files | t100 files | independent t100 groups | potential new vs BJ | after-BJ needed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 18 | 7 | 6 | 5 | 2 |
| `UCY` | 24 | 6 | 4 | 0 | 0 |
| `TrajNet` | 59 | 0 | 0 | 0 | 3 |

## Conversion Candidates

| relative_path | domain | independent_key | format | max track | t100 windows | status |
| --- | --- | --- | --- | ---: | ---: | --- |
| `ETH-Person/data/bahnhof_assc_gt.xml` | `ETH_UCY` | `ETH_UCY::ETH-Person/data/bahnhof_assc_gt` | `xml` | 217 | 377 | `candidate_pending_license_terms_and_source_cv` |
| `ETH-Person/data/jelmoli_assc_gt.xml` | `ETH_UCY` | `ETH_UCY::ETH-Person/data/jelmoli_assc_gt` | `xml` | 161 | 136 | `candidate_pending_license_terms_and_source_cv` |
| `ETH-Person/data/seq0_assc_gt.xml` | `ETH_UCY` | `ETH_UCY::ETH-Person/data/seq0_assc_gt` | `xml` | 303 | 335 | `candidate_pending_license_terms_and_source_cv` |
| `ETH-Person/data/seq0_assc_gt-interp.xml` | `ETH_UCY` | `ETH_UCY::ETH-Person/data/seq0_assc_gt-interp` | `xml` | 353 | 471 | `candidate_pending_license_terms_and_source_cv` |
| `ETH-Person/data/sunnyday_assc_gt.xml` | `ETH_UCY` | `ETH_UCY::ETH-Person/data/sunnyday_assc_gt` | `xml` | 305 | 410 | `candidate_pending_license_terms_and_source_cv` |
| `ETH/seq_eth/biwi_eth_10fps.txt` | `ETH_UCY` | `ETH_UCY::ETH/seq_eth` | `txt` | 114 | 14 | `candidate_pending_license_terms_and_source_cv` |
| `ETH/seq_eth/obsmat.txt` | `ETH_UCY` | `ETH_UCY::ETH/seq_eth` | `txt` | 190 | 91 | `candidate_pending_license_terms_and_source_cv` |
| `UCY/students01/students001.txt` | `UCY` | `UCY::UCY/students01` | `txt` | 352 | 1949 | `candidate_pending_license_terms_and_source_cv` |
| `UCY/students03/obsmat.txt` | `UCY` | `UCY::UCY/students03` | `txt` | 539 | 3413 | `candidate_pending_license_terms_and_source_cv` |
| `UCY/students03/obsmat_px.txt` | `UCY` | `UCY::UCY/students03` | `txt` | 540 | 3415 | `candidate_pending_license_terms_and_source_cv` |
| `UCY/students03/students003.txt` | `UCY` | `UCY::UCY/students03` | `txt` | 289 | 879 | `candidate_pending_license_terms_and_source_cv` |
| `UCY/zara01/obsmat.txt` | `UCY` | `UCY::UCY/zara01` | `txt` | 197 | 97 | `candidate_pending_license_terms_and_source_cv` |
| `UCY/zara02/obsmat.txt` | `UCY` | `UCY::UCY/zara02` | `txt` | 583 | 2095 | `candidate_pending_license_terms_and_source_cv` |

## TrajNet Loader Gap

- trajnet_t100_capable_files: `0`
- trajnet_loader_gap_files_sampled: `59`
- Interpretation: local TrajNet files are fixed short challenge snippets, not raw long-track sources; they cannot repair raw-frame t100 without original longer trajectories.

## Summary

- eth_ucy_t100_capable_files: `7`
- eth_ucy_independent_t100_groups: `6`
- eth_ucy_potential_new_groups_vs_bj: `5`
- eth_person_xml_candidates: `['ETH-Person/data/bahnhof_assc_gt.xml', 'ETH-Person/data/jelmoli_assc_gt.xml', 'ETH-Person/data/seq0_assc_gt.xml', 'ETH-Person/data/seq0_assc_gt-interp.xml', 'ETH-Person/data/sunnyday_assc_gt.xml']`
- can_repair_eth_ucy_with_local_candidates_after_license_confirmation: `True`
- can_repair_trajnet_with_local_candidates: `False`
- global_t100_positive_claim_allowed: `False`

## Interpretation

- ETH-Person XML files close a loader gap and provide local ETH_UCY t100 conversion candidates, pending license/terms confirmation and conversion/no-leakage/source-CV.
- TrajNet local challenge snippets do not provide raw-frame t100 sources; TrajNet still needs official longer sources or a different non-t100 claim.
- This is source/path verification only. It is not a converted dataset, trained model, evaluation result, metric claim, or seconds-level claim.
