# Stage 3 Data Sources For 2.5D Crowd World Model

## Recommendation

Prioritize `TGSIM Foggy Bottom` first because it already provides meter-scale position, speed, acceleration, road-user type, dimensions, an aerial reference image, and region polygons. This directly attacks the current model's weak points: real t+100 evaluation, scene geometry, physical variables, and mixed-agent interactions.

## Ranked Sources

| Priority | Key | Dataset | Coordinate Quality | Scene Geometry | t+100 Readiness | Loader Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `tgsim_foggy_bottom` | [TGSIM Foggy Bottom Trajectories](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories) | meter-scale coordinates; published conversion factor from pixels to meters | aerial reference image plus 49 polygon boundaries for road/crosswalk/intersection regions | strong: 0.1s records over a 2-hour window, enough for many t+100 windows | stage3 adapter planned; no auto-download in this run |
| 2 | `stanford_drone_dataset` | [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) | image-space trajectories; scene-specific scale/homography must be estimated or supplied | top-view videos/reference frames; no full metric scene graph by default | medium-strong: long videos, but metric evaluation needs calibration | metadata/catalog only; avoid automatic 69 GB download |
| 3 | `opentraj_bundle` | [OpenTraj Dataset Toolkit](https://github.com/crowdbotp/OpenTraj) | mixed; includes world-2D datasets such as ETH, inD, DUT, VRU, and image-space datasets | mixed; some datasets include maps or context, many do not | medium: unified loaders help, but every source needs horizon/scene validation | adapter planned through optional external OpenTraj install |
| 4 | `eth_ucy` | [ETH / UCY Pedestrian Trajectories](https://vision.ee.ethz.ch/datsets.html and https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) | commonly used world/pixel pedestrian benchmark coordinates | limited; homographies often available through benchmark preprocessing | medium: good benchmark, but trajectories are shorter and less scene-rich | adapter planned |
| 5 | `trajnet_plus_plus` | [TrajNet++](https://www.epfl.ch/labs/vita/datasets/) | trajectory benchmark coordinates, interaction-centric | usually trajectory-only; scene geometry sparse | medium: interaction benchmark, but standard horizons may differ from t+100 | adapter planned |
| 6 | `opendd` | [OpenDD Roundabout Drone Dataset](https://l3pilot.eu/data/opendd.html) | trajectory database with HD map information | HD map, shapefiles, geo-referenced drone images | strong for mixed traffic; pedestrian-only subset must be checked | later-stage adapter; license restricts derived redistribution |
| 7 | `ind` | [inD Intersection Drone Dataset](https://www.ind-dataset.com/) | world-2D trajectories at German intersections | intersection layouts and recording metadata | strong for mixed-agent intersections; pedestrian subset can evaluate long windows | later-stage adapter |

## Variables By Dataset

### TGSIM Foggy Bottom Trajectories

- URL: https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories
- Agent types: pedestrian, bicycle, scooter, vehicle, bus, truck
- Available variables: position_m, speed_mps, acceleration_mps2, width_m, length_m, road_user_type, region_polygon_id, aerial_reference_image
- Size/access: main CSV around 350 MB plus reference image and region annotations
- Why it matters: Best immediate upgrade for physical world modeling because it includes acceleration, object size, meters, and scene polygons.
- License note: Public data.gov dataset; check resource terms before redistribution.

### Stanford Drone Dataset

- URL: https://cvgl.stanford.edu/projects/uav_data/
- Agent types: pedestrian, bicyclist, skateboarder, cart, car, bus
- Available variables: track_id, bbox, agent_type, image_position, scene_name, video_id
- Size/access: official Stanford Campus download is large, about 69 GB
- Why it matters: Large top-view campus interactions with multiple agent classes; good for social navigation and mixed-agent priors.
- License note: Creative Commons BY-NC-SA 3.0 on official page.

### OpenTraj Dataset Toolkit

- URL: https://github.com/crowdbotp/OpenTraj
- Agent types: pedestrian, cyclist, vehicle, robot/person depending on dataset
- Available variables: position, track_id, dataset_name, fps, coordinate_system
- Size/access: toolkit small; datasets vary
- Why it matters: Fastest way to compare multiple pedestrian trajectory datasets under one API.
- License note: MIT toolkit; individual dataset licenses vary.

### ETH / UCY Pedestrian Trajectories

- URL: https://vision.ee.ethz.ch/datsets.html and https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data
- Agent types: pedestrian
- Available variables: position, track_id, scene, frame
- Size/access: small compared with SDD/TGSIM
- Why it matters: Canonical social trajectory benchmark for sanity-checking learned dynamics.
- License note: Academic benchmark; verify each archive before redistribution.

### TrajNet++

- URL: https://www.epfl.ch/labs/vita/datasets/
- Agent types: pedestrian
- Available variables: position, track_id, interaction_category
- Size/access: benchmark-sized
- Why it matters: Useful for stress-testing social interaction and collision-aware forecasting.
- License note: Challenge/benchmark terms apply.

### OpenDD Roundabout Drone Dataset

- URL: https://l3pilot.eu/data/opendd.html
- Agent types: pedestrian, vehicle, cyclist
- Available variables: trajectory, bounding_box, agent_type, hd_map, utm_coordinates
- Size/access: large multi-part dataset, 62+ hours according to official page
- Why it matters: Excellent for map-aware constraints and mixed-agent physical interaction.
- License note: CC BY-ND 4.0 on official page.

### inD Intersection Drone Dataset

- URL: https://www.ind-dataset.com/
- Agent types: pedestrian, bicyclist, vehicle
- Available variables: position, velocity, heading, agent_type, track_id
- Size/access: moderate/large; access may require registration
- Why it matters: Good mixed-agent interaction data but less immediately convenient than public TGSIM.
- License note: leveLX/inD terms; verify before download/use.
