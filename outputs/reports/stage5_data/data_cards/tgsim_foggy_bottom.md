# Data Card: TGSIM Foggy Bottom

## Access

- Domain: traffic
- Official URL: https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories
- Download status: downloaded
- License: U.S. public data portal; verify resource terms before redistribution
- Commercial use allowed: likely_yes_verify
- Redistribution allowed: likely_yes_verify
- Citation required: yes

## World-State Value

- Trajectories: True
- Metric coordinates: True
- Scene map / geometry: True / True / True
- Agent type: True
- Heading / velocity / acceleration: True / True / True
- Can evaluate t+100: True

## Use In This Project

- Loader status: working_quick
- Download command: `python scripts/download_stage5_datasets.py --dataset tgsim_foggy_bottom --max-gb 2`
- Preprocessing command: `python run_stage4p5_dynamics_benchmark.py --dataset tgsim --data <csv-or-url> --quick`
- Priority score: 90
- Priority reason: legally accessible, metric coordinates, verified t+100 likely/available, scene/map context, traffic dynamics

## Notes

Official quick endpoint has no scene polygons loaded yet; causal velocity is official.
