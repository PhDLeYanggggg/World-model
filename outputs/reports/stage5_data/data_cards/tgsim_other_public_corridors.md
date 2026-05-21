# Data Card: TGSIM other public corridors

## Access

- Domain: traffic
- Official URL: https://data.transportation.gov/
- Download status: downloadable
- License: U.S. public data portal; verify each resource
- Commercial use allowed: likely_yes_verify
- Redistribution allowed: likely_yes_verify
- Citation required: yes

## World-State Value

- Trajectories: True
- Metric coordinates: True
- Scene map / geometry: True / False / False
- Agent type: True
- Heading / velocity / acceleration: True / True / True
- Can evaluate t+100: True

## Use In This Project

- Loader status: adapter_partial
- Download command: `python scripts/download_stage5_datasets.py --dataset tgsim_all --max-gb 20`
- Preprocessing command: ``
- Priority score: 95
- Priority reason: legally accessible, metric coordinates, verified t+100 likely/available, scene/map context, traffic dynamics

## Notes

Discover individual Socrata resource ids before bulk download.
