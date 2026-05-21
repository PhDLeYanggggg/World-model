# Stage 6 Pedestrian / Drone Long-Horizon Audit

This audit counts only actual local converted/user-path verified data. Registry-estimated t+100 does not count.

| dataset_name | actual_downloaded_or_user_path_verified | license | coordinate_unit | pixel_or_metric | homography_available | max_raw_horizon | max_verified_t50 | max_verified_t100 | eligible_for_pedestrian_drone_long_horizon_gate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stanford Drone Dataset | False | CC BY-NC-SA 3.0 | unknown | unknown | False | 0 | 0 | 0 | False | No pedestrian/drone long-horizon world model claim is allowed. |
| OpenTraj-supported pedestrian datasets | False | unknown_or_not_registered | unknown | unknown | False | 0 | 0 | 0 | False | No pedestrian/drone long-horizon world model claim is allowed. |
| full TrajNet++ | True | benchmark terms; verify individual files | dataset_coordinate | pixel/dataset_coordinate | False | 10 | 0 | 0 | False | No pedestrian/drone long-horizon world model claim is allowed. |
| full ETH/UCY | True | academic dataset terms; verify before redistribution | dataset_coordinate | pixel/dataset_coordinate | True | 10 | 0 | 0 | False | No pedestrian/drone long-horizon world model claim is allowed. |
| UCY original crowd | False | academic dataset terms; verify before redistribution | unknown | unknown | True | 0 | 0 | 0 | False | No pedestrian/drone long-horizon world model claim is allowed. |
| AerialMPT longer sequences | True | unknown_or_not_registered | pixel_or_unknown | pixel/dataset_coordinate | False | 12 | 0 | 0 | False | No pedestrian/drone long-horizon world model claim is allowed. |

No pedestrian/drone long-horizon world model claim is allowed unless at least one actual verified t+50/t+100 source appears here.