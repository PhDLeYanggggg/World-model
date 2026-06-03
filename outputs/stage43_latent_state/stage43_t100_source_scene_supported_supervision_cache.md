# Stage43-CQ T100 Source/Scene-Supported Supervision Cache

- source: `fresh_stage43_cq_t100_source_scene_supported_supervision_cache`
- result_source: `fresh_t100_only_source_scene_supported_full_waypoint_supervision_cache`
- verdict: `stage43_cq_t100_source_scene_supported_supervision_cache_pass`
- gate: `15 / 15`
- t100 supported supervised training ready: `True`
- cache dir: `data/stage43_cp_t100_source_scene_support_cache`
- cache committed: `False`

## Split Summary

| split | rows | domains | sources | scenes | source-agents | full waypoint rows | all-waypoint rows | endpoint diff max | hard | failure | easy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 33480 | `{'ETH_UCY': 17024, 'TrajNet': 10448, 'UCY': 6008}` | 13 | 10 | 2804 | 33480 | 33480 | 0.00000000 | 33480 | 8891 | 7211 |
| val | 11444 | `{'ETH_UCY': 5964, 'TrajNet': 3480, 'UCY': 2000}` | 13 | 10 | 938 | 11444 | 11444 | 0.00000000 | 11444 | 2868 | 2553 |
| test | 11820 | `{'ETH_UCY': 6340, 'TrajNet': 3480, 'UCY': 2000}` | 13 | 10 | 954 | 11820 | 11820 | 0.00000000 | 11820 | 2930 | 3085 |

## Cache Files

| split | path | sha256 | row hash |
| --- | --- | --- | --- |
| train | `data/stage43_cp_t100_source_scene_support_cache/stage43_cp_t100_supervision_train.npz` | `f22aa542d9127525ef84ee48d8593675f147fcb2eb049c4a7ebf066e547bf5a7` | `33df49999eaeae10e6ca95ebcdce7871f859623df5d3f84f7a74f61e8449affd` |
| val | `data/stage43_cp_t100_source_scene_support_cache/stage43_cp_t100_supervision_val.npz` | `3414970ec241ad030d3bfec8156e9d1b11494ab9754e3b5d4d5c4d09cdefdb70` | `bd9e89486967af0601a5604fdb9c25470c849dfacdc856d01b7f904f8a2a0995` |
| test | `data/stage43_cp_t100_source_scene_support_cache/stage43_cp_t100_supervision_test.npz` | `742bbce40c513e9c99eb1a41e0ef41fd4553998830f9cb58d336c09d0683c8de` | `e3c0153b44fe98aaeac30c567707001860bf7537535d7a382deb7288b2ed36e9` |

## Protocol Boundary

- CP assignment hash: `a6b33fa427f68e5765fc92cfc6666dfde8252e414d23c95c46a4d661700c97d2`
- source-agent overlap counts: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- row overlap counts: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- source overlap counts: `{'train_val': 13, 'train_test': 13, 'val_test': 13}`
- scene overlap counts: `{'train_val': 10, 'train_test': 10, 'val_test': 10}`
- source/scene overlap is intentional for this supported protocol; source-agent tracks and rows remain disjoint.
- Future endpoints and waypoints are labels/eval only, not inference inputs.

## Interpretation

- Stage43-CQ turns the Stage43-CP support manifest into a t100-only supervised cache.
- This still does not solve t100; it only removes the previous validation-support blocker for the next t100 learner.
- The stricter heldout current split remains floor-only at t100 until a model passes its safety gates.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
