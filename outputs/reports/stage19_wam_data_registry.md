# Stage 19 WAM-Style Data Registry

- Data roles are separated: official real top-down eval, representation pretraining, simulation curriculum, diagnostic only.
- No unauthorized downloads or internet video scraping are allowed.

| dataset | category | local path | official eval | pretraining | priority | next action |
| --- | --- | --- | --- | --- | ---: | --- |
| Stanford Drone Dataset | real_topdown_trajectory | False | False | True | 95 | provide local path / accept license |
| OpenTraj datasets | real_topdown_trajectory | False | False | True | 80 | provide local path / accept license |
| full TrajNet++ | real_topdown_trajectory | True | True | True | 80 | run converter |
| full ETH/UCY | real_topdown_trajectory | True | True | True | 80 | run converter |
| AerialMPT longer sequences | real_topdown_trajectory | True | True | True | 80 | run converter |
| Ego4D | human_egocentric_video | False | False | False | 55 | provide local dataset path after license approval |
| Ego-Exo4D | human_egocentric_video | False | False | False | 55 | provide local dataset path after license approval |
| EPIC-KITCHENS | human_egocentric_video | False | False | False | 55 | provide local dataset path after license approval |
| HoloAssist | human_egocentric_video | False | False | False | 55 | provide local dataset path after license approval |
| HOI4D | human_object_interaction / manipulation video | False | False | False | 35 | optional; provide local path only if representation pretraining expands beyond pedestrians |
| EgoDex | human_object_interaction / manipulation video | False | False | False | 35 | optional; provide local path only if representation pretraining expands beyond pedestrians |
| DexYCB | human_object_interaction / manipulation video | False | False | False | 35 | optional; provide local path only if representation pretraining expands beyond pedestrians |
| Assembly101 | human_object_interaction / manipulation video | False | False | False | 35 | optional; provide local path only if representation pretraining expands beyond pedestrians |
| Nymeria | human_object_interaction / manipulation video | False | False | False | 35 | optional; provide local path only if representation pretraining expands beyond pedestrians |
| SyntheticPhysicalCrowd2.5D | simulation data | False | False | True | 45 | none |
| SyntheticTraffic2.5D | simulation data | False | False | True | 45 | none |
| SyntheticMixedAgents2.5D | simulation data | False | False | True | 45 | none |
| UrbanCrowdSim2.5D | simulation data | False | False | True | 75 | none |
| ManiSkill | robotics / WAM auxiliary | False | False | False | 20 | optional and out-of-scope for official pedestrian benchmark |
| RoboCasa | robotics / WAM auxiliary | False | False | False | 20 | optional and out-of-scope for official pedestrian benchmark |
| RoboTwin | robotics / WAM auxiliary | False | False | False | 20 | optional and out-of-scope for official pedestrian benchmark |
| MimicGen | robotics / WAM auxiliary | False | False | False | 20 | optional and out-of-scope for official pedestrian benchmark |
| DexMimicGen | robotics / WAM auxiliary | False | False | False | 20 | optional and out-of-scope for official pedestrian benchmark |
| SynGrasp-style synthetic | robotics / WAM auxiliary | False | False | False | 20 | optional and out-of-scope for official pedestrian benchmark |
