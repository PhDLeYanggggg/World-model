# Stage42-IA HZ to CG Intake Bridge

- source: `fresh_stage42_ia_hz_to_cg_intake_bridge`
- generated_at_utc: `2026-05-27T21:17:49.956056+00:00`
- git_commit: `4cf6c0d`
- input_hash: `c18a44c1f3ebe7f3e43a6f8c289213a807d1f1243be80b9f04dcd9da8af62930`
- gate: `17 / 17`
- verdict: `stage42_ia_hz_to_cg_intake_bridge_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IA 只把 HZ confirmation template 映射成 CG validator 兼容 intake 文件，不下载、不转换、不训练、不评估。
- Bridge output is not automatically active validator input; user must intentionally review/copy or a future guarded runner must explicitly select it.
- local path found 不等于 legal terms accepted，不等于 official source identity confirmed。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Bridge Rows

| dataset | domain | HZ row | local path | ready if activated | missing confirmation |
| --- | --- | ---: | --- | ---: | --- |
| `ucy_crowd_original` | `UCY` | `True` | `external_data/OpenTraj/datasets/UCY` | `False` | terms_accepted_by_user, terms_acceptance_date, allowed_use, source_identity, confirmed_by_user, official_source_url_confirmed, local_path_confirmed, source_identity_confirmed |
| `eth_biwi_original` | `ETH_UCY` | `True` | `external_data/OpenTraj/datasets/ETH` | `False` | terms_accepted_by_user, terms_acceptance_date, allowed_use, source_identity, confirmed_by_user, official_source_url_confirmed, local_path_confirmed, source_identity_confirmed |
| `aerialmpt_or_other_topdown` | `other_topdown` | `True` | `data/aerialmpt` | `False` | terms_accepted_by_user, terms_acceptance_date, allowed_use, source_identity, confirmed_by_user, official_source_url_confirmed, local_path_confirmed, source_identity_confirmed |
| `opentraj_toolkit` | `OpenTraj` | `True` | `external_data/OpenTraj` | `False` | terms_accepted_by_user, terms_acceptance_date, allowed_use, source_identity, confirmed_by_user, official_source_url_confirmed, local_path_confirmed, source_identity_confirmed |
| `trajnetplusplus_official` | `TrajNet` | `True` | `external_data/OpenTraj/datasets/TrajNet++` | `False` | terms_accepted_by_user, terms_acceptance_date, allowed_use, source_identity, confirmed_by_user, official_source_url_confirmed, local_path_confirmed, source_identity_confirmed |

## Interpretation

- IA prevents HZ and CG from drifting into incompatible source-terms schemas.
- The bridged template is deliberately inactive. It is not used by the validator unless a future command or user action explicitly selects/copies it.
- Current ready-if-activated count is zero because HZ confirmation fields remain blank.
