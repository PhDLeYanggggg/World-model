# Stage 5B.6 Data Card

Stage 5B.6 uses the existing actual converted Stage 5B sources: TGSIM, TGSIM-I90, TrajNet fallback, and ETH/UCY fallback. It does not count registry-only data as converted data.

## Hard Reliability
| dataset | hard | verified_t100_hard | reliability | gate_eligible |
| --- | --- | --- | --- | --- |
| eth_ucy | 2 | 0 | diagnostic_only | False |
| tgsim | 3 | 3 | diagnostic_only | False |
| tgsim_i90 | 8 | 8 | diagnostic_only | False |
| trajnet | 6 | 0 | diagnostic_only | False |

## Pedestrian / Drone Horizon
| dataset | unit | max_raw_horizon | t50 | t100 | official_gate |
| --- | --- | --- | --- | --- | --- |
| Stanford Drone Dataset | unknown | 0 | False | False | False |
| TrajNet++ | dataset_coordinate | 10 | False | False | False |
| ETH/UCY | dataset_coordinate | 10 | False | False | False |
| OpenTraj-compatible pedestrian datasets | unknown | 0 | False | False | False |
| AerialMPT longer sequences | pixel_or_unknown | 12 | False | False | False |

No new verified pedestrian/drone t+50/t+100 source was added.
