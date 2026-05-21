# Stage 6 Data Card

Stage 6 counts only actual local converted/user-path verified sources. Registry-only datasets are not benchmark data.

## Pedestrian/Drone Audit
| dataset | downloaded_or_path | unit | t50 | t100 | gate |
| --- | --- | --- | --- | --- | --- |
| Stanford Drone Dataset | False | unknown | 0 | 0 | False |
| OpenTraj-supported pedestrian datasets | False | unknown | 0 | 0 | False |
| full TrajNet++ | True | dataset_coordinate | 0 | 0 | False |
| full ETH/UCY | True | dataset_coordinate | 0 | 0 | False |
| UCY original crowd | False | unknown | 0 | 0 | False |
| AerialMPT longer sequences | True | pixel_or_unknown | 0 | 0 | False |

## HardBench-v1
| field | value |
| --- | --- |
| total_hard_episodes | 53 |
| gate_eligibility | official |
| pedestrian_drone_hard_episodes | 14 |
| traffic_hard_episodes | 39 |

## BaselineFailureBench
| dataset | samples | failure_samples | near_failure_samples | failure_rate | enough_for_training | enough_for_evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | 23 | 8 | 2 | 0.347826 | False | True |
| tgsim | 32 | 3 | 0 | 0.09375 | False | False |
| tgsim_i90 | 31 | 25 | 2 | 0.806452 | True | True |
| trajnet | 32 | 12 | 1 | 0.375 | True | True |

