# Stage42-HZ Source Terms Confirmation Packet

- source: `fresh_stage42_hz_source_terms_confirmation_packet_from_hy_prefill`
- generated_at_utc: `2026-05-27T21:06:59.366620+00:00`
- git_commit: `c340328`
- input_hash: `b069469a8e5087d81d22b4d18e943e0d54df625c1ac0a9417920fe58a19a8247`
- gate: `22 / 22`
- verdict: `stage42_hz_source_terms_confirmation_packet_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HZ 只生成 source/terms confirmation packet 和 readiness validator，不下载、不转换、不训练、不评估。
- local path found 不等于 legal terms accepted，不等于 official source identity confirmed。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Packet Rows

| dataset | domain | local path | parseable hints | t50 after terms | t100 after terms | conversion now |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `ucy_crowd_original` | `UCY` | `external_data/OpenTraj/datasets/UCY` | obsmat, homography, video, reference_image | 9554 | 5605 | `False` |
| `eth_biwi_original` | `ETH_UCY` | `external_data/OpenTraj/datasets/ETH` | obsmat, homography, video, reference_image | 506 | 91 | `False` |
| `aerialmpt_or_other_topdown` | `other_topdown` | `data/aerialmpt` | zip | 0 | 0 | `False` |
| `opentraj_toolkit` | `OpenTraj` | `external_data/OpenTraj` | obsmat, ndjson, homography, video, reference_image | 0 | 0 | `False` |
| `trajnetplusplus_official` | `TrajNet` | `external_data/OpenTraj/datasets/TrajNet++` | ndjson | 0 | 0 | `False` |

## Interpretation

- This packet converts the HY local-path prefill into a user-confirmable source/terms checklist.
- It intentionally keeps every source blocked until the user confirms terms, local path, official/source identity, and allowed use.
- No conversion, no training, no evaluation, no metric/time claim, no Stage5C, and no SMC occurred.
