# Stage 20 Web Search Report

- Current model is not true 3D and not a large-scale foundation world model.
- BPSG-MA v1 remains strongest causal baseline fallback + diagnostics.
- SAM-JEPA-2.5D is representation pretraining, not latent rollout; Stage18 did not improve downstream heads or official t+50.
- Stage 5C latent generative remains blocked; SMC remains blocked.

- Candidate sources found/deduplicated: `33`
- Official-source candidates: `31`

## Top Priority Sources

| dataset | category | score | official URL | next action |
| --- | --- | ---: | --- | --- |
| SyntheticMixedAgents2.5D | simulation_and_synthetic | 100 | local_project_generator | Use only as curriculum/stress test, not official real benchmark. |
| SyntheticPhysicalCrowd2.5D | simulation_and_synthetic | 100 | local_project_generator | Use only as curriculum/stress test, not official real benchmark. |
| UrbanCrowdSim2.5D | simulation_and_synthetic | 100 | local_project_generator | Use only as curriculum/stress test, not official real benchmark. |
| UCY Crowd original | real_topdown_pedestrian_drone_official_benchmark | 95 | http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | Verify original UCY path or use local TrajNet-origin UCY subset with source caveat. |
| full ETH/UCY / EWAP | real_topdown_pedestrian_drone_official_benchmark | 95 | https://icu.ee.ethz.ch/research/datsets.html | Verify full local ETH/UCY path if user has it; keep t+100 diagnostic until enough rows. |
| BIWI Walking Pedestrians | real_topdown_or_fixed_camera_auxiliary | 85 | https://icu.ee.ethz.ch/research/datsets.html | Verify local path/license if user wants auxiliary training data. |
| ETH pedestrian original / BIWI | real_topdown_or_fixed_camera_auxiliary | 85 | https://icu.ee.ethz.ch/research/datsets.html | Verify local path/license if user wants auxiliary training data. |
| OpenTraj supported datasets | real_topdown_pedestrian_drone_official_benchmark | 85 | https://github.com/crowdbotp/OpenTraj | Provide OpenTraj local path or allow toolkit-only clone; verify underlying dataset licenses. |
| NGSIM | traffic / driving diagnostic only | 80 | https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm | Use only if legal local path exists and reports keep traffic separate. |
| Stanford Drone Dataset | real_topdown_pedestrian_drone_official_benchmark | 80 | https://cvgl.stanford.edu/projects/uav_data/ | User must provide local SDD path after accepting Stanford non-commercial terms. |
| Edinburgh Informatics Forum pedestrian trajectories | real_topdown_or_fixed_camera_auxiliary | 75 | https://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/ | Verify local path/license if user wants auxiliary training data. |
| Grand Central pedestrian trajectories | real_topdown_or_fixed_camera_auxiliary | 75 | http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html | Verify local path/license if user wants auxiliary training data. |
