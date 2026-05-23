# Stage 20 Dataset Registry

This registry records candidates only. Registry-only data is not counted as converted.

| dataset | category | official | license | local path | auto download | official eval | JEPA | score | next action |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| SyntheticMixedAgents2.5D | simulation_and_synthetic | True | project-generated | False | True | False | True | 100 | Use only as curriculum/stress test, not official real benchmark. |
| SyntheticPhysicalCrowd2.5D | simulation_and_synthetic | True | project-generated | False | True | False | True | 100 | Use only as curriculum/stress test, not official real benchmark. |
| UrbanCrowdSim2.5D | simulation_and_synthetic | True | project-generated | False | True | False | True | 100 | Use only as curriculum/stress test, not official real benchmark. |
| OpenTraj supported datasets | real_topdown_pedestrian_drone_official_benchmark | True | MIT for toolkit; underlying datasets keep their own licenses | True | False | True | True | 95 | Provide OpenTraj local path or allow toolkit-only clone; verify underlying dataset licenses. |
| UCY Crowd original | real_topdown_pedestrian_drone_official_benchmark | True | UCY crowd research terms; verify before use | True | False | True | False | 95 | Verify original UCY path or use local TrajNet-origin UCY subset with source caveat. |
| full ETH/UCY / EWAP | real_topdown_pedestrian_drone_official_benchmark | True | research dataset terms; verify original source | True | False | True | True | 95 | Verify full local ETH/UCY path if user has it; keep t+100 diagnostic until enough rows. |
| Stanford Drone Dataset | real_topdown_pedestrian_drone_official_benchmark | True | Stanford SDD non-commercial / custom access terms | True | False | True | True | 90 | User must provide local SDD path after accepting Stanford non-commercial terms. |
| BIWI Walking Pedestrians | real_topdown_or_fixed_camera_auxiliary | True | research dataset terms; verify before use | False | False | False | True | 85 | Verify local path/license if user wants auxiliary training data. |
| ETH pedestrian original / BIWI | real_topdown_or_fixed_camera_auxiliary | True | research dataset terms; verify before use | False | False | False | True | 85 | Verify local path/license if user wants auxiliary training data. |
| NGSIM | traffic / driving diagnostic only | True | FHWA / ITS Public Data Hub | False | False | False | True | 80 | Use only if legal local path exists and reports keep traffic separate. |
| Edinburgh Informatics Forum pedestrian trajectories | real_topdown_or_fixed_camera_auxiliary | True | research dataset terms; verify before use | False | False | False | True | 75 | Verify local path/license if user wants auxiliary training data. |
| Grand Central pedestrian trajectories | real_topdown_or_fixed_camera_auxiliary | True | research dataset terms; verify before use | False | False | False | True | 75 | Verify local path/license if user wants auxiliary training data. |
| PETS 2009 S2L1 | real_topdown_or_fixed_camera_auxiliary | True | research dataset terms; verify before use | False | False | False | True | 75 | Verify local path/license if user wants auxiliary training data. |
| TownCentre Dataset | real_topdown_or_fixed_camera_auxiliary | True | research dataset terms; verify before use | False | False | False | True | 75 | Verify local path/license if user wants auxiliary training data. |
| Argoverse Motion Forecasting | traffic / driving diagnostic only | True | Argoverse terms; verify | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| INTERACTION dataset | traffic / driving diagnostic only | True | INTERACTION dataset terms | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| OpenDD | traffic / driving diagnostic only | True | CC BY-ND 4.0 | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| TGSIM | traffic / driving diagnostic only | True | trajectory data; official portal required | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| Waymo Open Motion Dataset | traffic / driving diagnostic only | True | Waymo Open Dataset terms | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| exiD | traffic / driving diagnostic only | True | levelXdata terms; verify | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| highD | traffic / driving diagnostic only | True | levelXdata terms; verify | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| inD | traffic / driving diagnostic only | True | levelXdata terms; verify | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| nuScenes prediction | traffic / driving diagnostic only | True | nuScenes license terms | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| rounD | traffic / driving diagnostic only | True | levelXdata terms; verify | False | False | False | True | 70 | Use only if legal local path exists and reports keep traffic separate. |
| TrajNet++ full datasets | real_topdown_pedestrian_drone_official_benchmark | True | dataset-specific / challenge terms | True | False | True | False | 65 | Run Stage20 raw-index conversion and horizon/no-leakage audit. |
| Constrained Stanford Drone Dataset | real_topdown_pedestrian_drone_official_benchmark | False | inherits SDD/non-commercial if derived; verify | False | False | False | False | 45 | Locate official project/source or ignore. |
| EPIC-KITCHENS | human_egocentric_video_pretraining | True | CC BY-NC 4.0 / non-commercial | False | False | False | True | 40 | User must obtain official access and provide local path; no scraping or bypass. |
| Assembly101 | human_egocentric_video_pretraining | True | research terms; verify | False | False | False | True | 30 | User must obtain official access and provide local path; no scraping or bypass. |
| Ego-Exo4D | human_egocentric_video_pretraining | True | Ego-Exo4D license agreement | False | False | False | True | 30 | User must obtain official access and provide local path; no scraping or bypass. |
| Ego4D | human_egocentric_video_pretraining | True | Ego4D data usage/license agreement | False | False | False | True | 30 | User must obtain official access and provide local path; no scraping or bypass. |
| HOI4D | human_egocentric_video_pretraining | True | research terms; verify | False | False | False | True | 30 | User must obtain official access and provide local path; no scraping or bypass. |
| HoloAssist | human_egocentric_video_pretraining | True | research terms; verify | False | False | False | True | 30 | User must obtain official access and provide local path; no scraping or bypass. |
| AerialMPT longer sequences | real_topdown_pedestrian_drone_official_benchmark | False | unknown | False | False | False | False | 5 | User should provide official URL/path if available; do not download mirrors. |
