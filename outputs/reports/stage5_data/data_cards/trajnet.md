# Data Card: TrajNet++

## Access

- Domain: pedestrian
- Official URL: https://www.epfl.ch/labs/vita/datasets/
- Download status: downloadable
- License: benchmark terms; verify individual files
- Commercial use allowed: unknown
- Redistribution allowed: unknown
- Citation required: yes

## World-State Value

- Trajectories: True
- Metric coordinates: True
- Scene map / geometry: False / False / False
- Agent type: False
- Heading / velocity / acceleration: False / False / False
- Can evaluate t+100: True

## Use In This Project

- Loader status: partial_loader
- Download command: `python scripts/download_stage5_datasets.py --dataset trajnet --max-gb 5`
- Preprocessing command: ``
- Priority score: 65
- Priority reason: legally accessible, metric coordinates, verified t+100 likely/available, pedestrian/crowd dynamics

## Notes


