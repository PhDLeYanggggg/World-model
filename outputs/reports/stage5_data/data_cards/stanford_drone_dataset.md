# Data Card: Stanford Drone Dataset

## Access

- Domain: drone
- Official URL: https://cvgl.stanford.edu/projects/uav_data/
- Download status: downloadable
- License: CC BY-NC-SA 3.0
- Commercial use allowed: no
- Redistribution allowed: share_alike_noncommercial
- Citation required: yes

## World-State Value

- Trajectories: True
- Metric coordinates: False
- Scene map / geometry: True / False / False
- Agent type: True
- Heading / velocity / acceleration: False / False / False
- Can evaluate t+100: True

## Use In This Project

- Loader status: planned
- Download command: `python scripts/download_stage5_datasets.py --dataset stanford_drone --max-gb 30`
- Preprocessing command: ``
- Priority score: 75
- Priority reason: legally accessible, verified t+100 likely/available, scene/map context, pedestrian/crowd dynamics

## Notes

Non-commercial license; do not use for commercial training without permission.
