# Stage 7 Scene Data Audit

| dataset | local/path | unit | scene_image | homography | t10 | t25 | t50 | t100 | metric_eval | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stanford_drone_dataset | False | pixel_or_image_coordinate_until_homography | False | False | False | False | False | False | False | Stage 7 does not auto-download SDD because license/manual terms must be handled by the user. |
| opentraj_supported_pedestrian | False | varies | False | False | False | False | False | False | False | Placeholder inspector only; source-specific conversion is required before official benchmark use. |
| full_trajnetplusplus | True | dataset_coordinate | True | False | True | False | False | False | False | Local TrajNet++ source exists, but current quick conversion only supports t+10 pedestrian-like episodes. |
| full_eth_ucy | True | dataset_coordinate_or_meter_depending_source | True | False | True | False | False | False | False | Current converted ETH/UCY fallback is t+10 only; no verified pedestrian t+50/t+100 locally. |
| aerialmpt_long | True | pixel | True | False | False | False | False | False | False | Existing AerialMPT bauma sample is short; t+100 remains qualitative-only. |

No pedestrian/drone t+50/t+100 claim is allowed unless the corresponding verified flags are true.
