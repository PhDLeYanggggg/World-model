# Stage 20 Download Plan

- Default mode is dry-run.
- No gated, login-required, license-required, or large dataset was downloaded.
- User action required is not counted as downloaded.

| dataset | action | auto after audit | license/access |
| --- | --- | --- | --- |
| SyntheticMixedAgents2.5D | local_generator_available | True | login=False, app=False, terms=False |
| SyntheticPhysicalCrowd2.5D | local_generator_available | True | login=False, app=False, terms=False |
| UrbanCrowdSim2.5D | local_generator_available | True | login=False, app=False, terms=False |
| UCY Crowd original | verify_existing_local_path | False | login=False, app=False, terms=False |
| full ETH/UCY / EWAP | verify_existing_local_path | False | login=False, app=False, terms=False |
| BIWI Walking Pedestrians | user_action_required | False | login=False, app=False, terms=False |
| ETH pedestrian original / BIWI | user_action_required | False | login=False, app=False, terms=False |
| OpenTraj supported datasets | user_action_required | False | login=False, app=False, terms=False |
| NGSIM | user_action_required | False | login=False, app=False, terms=False |
| Stanford Drone Dataset | user_action_required | False | login=False, app=False, terms=True |
| Edinburgh Informatics Forum pedestrian trajectories | user_action_required | False | login=False, app=False, terms=False |
| Grand Central pedestrian trajectories | user_action_required | False | login=False, app=False, terms=False |
| PETS 2009 S2L1 | user_action_required | False | login=False, app=False, terms=False |
| TownCentre Dataset | user_action_required | False | login=False, app=False, terms=False |
| Argoverse Motion Forecasting | user_action_required | False | login=True, app=False, terms=True |
| INTERACTION dataset | user_action_required | False | login=True, app=False, terms=True |
| OpenDD | user_action_required | False | login=True, app=False, terms=True |
| TGSIM | user_action_required | False | login=True, app=False, terms=True |
| Waymo Open Motion Dataset | user_action_required | False | login=True, app=False, terms=True |
| exiD | user_action_required | False | login=True, app=False, terms=True |
| highD | user_action_required | False | login=True, app=False, terms=True |
| inD | user_action_required | False | login=True, app=False, terms=True |
| nuScenes prediction | user_action_required | False | login=True, app=False, terms=True |
| rounD | user_action_required | False | login=True, app=False, terms=True |
| TrajNet++ full datasets | verify_existing_local_path | False | login=False, app=False, terms=False |
| Constrained Stanford Drone Dataset | user_action_required | False | login=False, app=False, terms=True |
| EPIC-KITCHENS | user_action_required | False | login=False, app=False, terms=True |
| Assembly101 | user_action_required | False | login=True, app=False, terms=True |
| Ego-Exo4D | user_action_required | False | login=True, app=False, terms=True |
| Ego4D | user_action_required | False | login=True, app=False, terms=True |
| HOI4D | user_action_required | False | login=True, app=False, terms=True |
| HoloAssist | user_action_required | False | login=True, app=False, terms=True |
| AerialMPT longer sequences | user_action_required | False | login=False, app=True, terms=False |
