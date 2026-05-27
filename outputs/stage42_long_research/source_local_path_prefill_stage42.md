# Stage42-HY Source Local Path Prefill

- source: `fresh_stage42_hy_source_local_path_prefill_from_local_files`
- generated_at_utc: `2026-05-27T20:56:03.908363+00:00`
- git_commit: `22585cf`
- input_hash: `0cb5ebadca4c3c967524afc05d8b2e7fed7efaa1e17177886903357e13e08c95`
- gate: `19 / 19`
- verdict: `stage42_hy_source_local_path_prefill_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HY 只做本地 source path / parseability 预填，不下载、不转换、不训练、不评估。
- local path found 不等于 legal terms accepted，不等于 official source identity confirmed。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Prefill Rows

| dataset | domain | local path found | best path | t50 after terms | t100 after terms | remaining confirmation |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `ucy_crowd_original` | `UCY` | `True` | `external_data/OpenTraj/datasets/UCY` | 9554 | 5605 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `eth_biwi_original` | `ETH_UCY` | `True` | `external_data/OpenTraj/datasets/ETH` | 506 | 91 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `aerialmpt_or_other_topdown` | `other_topdown` | `True` | `data/aerialmpt` | 0 | 0 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `opentraj_toolkit` | `OpenTraj` | `True` | `external_data/OpenTraj` | 0 | 0 | terms_accepted_by_user, terms_acceptance_date, allowed_use |
| `trajnetplusplus_official` | `TrajNet` | `True` | `external_data/OpenTraj/datasets/TrajNet++` | 0 | 0 | terms_accepted_by_user, terms_acceptance_date, allowed_use |

## Gate

| gate | pass |
| --- | --- |
| `gap_input_present` | `True` |
| `validation_input_present` | `True` |
| `all_gap_targets_audited` | `True` |
| `ucy_path_prefilled` | `True` |
| `eth_path_prefilled` | `True` |
| `trajnet_path_checked` | `True` |
| `aerialmpt_path_checked` | `True` |
| `parseability_hints_present` | `True` |
| `technical_windows_preserved` | `True` |
| `legal_block_preserved` | `True` |
| `no_download` | `True` |
| `no_conversion` | `True` |
| `no_training` | `True` |
| `no_evaluation` | `True` |
| `user_action_written` | `True` |
| `no_metric_seconds_claim` | `True` |
| `stage5c_not_executed` | `True` |
| `smc_not_enabled` | `True` |
| `readmes_updated` | `True` |

## Interpretation

- HY reduces a concrete blocker by locating local candidate paths and parseability hints for UCY/ETH/TrajNet/OpenTraj/AerialMPT-like sources.
- It does not accept terms, does not confirm official source identity, and does not run conversion/evaluation.
- The next valid step is user confirmation followed by guarded conversion + no-leakage + source-CV evaluation.
