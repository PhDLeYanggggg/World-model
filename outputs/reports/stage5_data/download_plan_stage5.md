# Stage 5 Download Plan

| dataset | status | action | reason | command |
| --- | --- | --- | --- | --- |
| OpenDD | downloadable | dry_run_download |  | `` |
| TGSIM other public corridors | downloadable | dry_run_download |  | `python scripts/download_stage5_datasets.py --dataset tgsim_all --max-gb 20` |
| TGSIM Foggy Bottom | downloaded | already_available |  | `python scripts/download_stage5_datasets.py --dataset tgsim_foggy_bottom --max-gb 2` |
| SyntheticPhysicalCrowd2.5D | downloaded | already_available |  | `` |
| Argoverse 1 Motion Forecasting | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| Argoverse 2 Motion Forecasting | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| INTERACTION Dataset | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| Waymo Open Motion Dataset | gated | placeholder_only | requires registration, license agreement, or application | `` |
| nuPlan | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| nuScenes Prediction | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| NGSIM | downloadable | dry_run_download |  | `` |
| Stanford Drone Dataset | downloadable | dry_run_download |  | `python scripts/download_stage5_datasets.py --dataset stanford_drone --max-gb 30` |
| exiD | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| inD | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| rounD | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| uniD | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| OpenTraj Toolkit | downloadable | dry_run_download |  | `` |
| SyntheticMixedAgents2.5D | unavailable | skip | download_status=unavailable | `` |
| SyntheticTraffic2.5D | unavailable | skip | download_status=unavailable | `` |
| TrajNet++ | downloadable | dry_run_download |  | `python scripts/download_stage5_datasets.py --dataset trajnet --max-gb 5` |
| UCY Crowd | downloadable | dry_run_download |  | `` |
| highD | requires_application | placeholder_only | requires registration, license agreement, or application | `` |
| ETH Pedestrian | downloadable | dry_run_download |  | `python scripts/download_stage5_datasets.py --dataset eth_ucy --max-gb 5` |
| Google Research Football | downloadable | dry_run_download |  | `` |
| Multi-Agent Particle Environment | downloadable | dry_run_download |  | `` |
| AerialMPT | downloaded | already_available |  | `` |
