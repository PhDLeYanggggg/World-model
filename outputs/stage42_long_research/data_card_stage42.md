# Stage42 Data Card

- datasets audited: `7`
- external domains ready from existing state: `['opentraj', 'eth_ucy', 'trajnet', 'ucy']`
- metric claim ready datasets: `['tgsim']`
- seconds claim ready datasets: `[]`
- global metric claim allowed: `False`
- global seconds claim allowed: `False`

## Data Roles

- SDD: pixel-space official benchmark evidence from earlier stages.
- External top-down trajectories: dataset-local raw-frame external validation for Stage42.
- TGSIM: diagnostic traffic unit/metric evidence only, not pedestrian official success.

## Leakage Policy

Future endpoints/waypoints are labels/evaluation only. No central velocity, no test endpoint goal construction, and no test threshold tuning are allowed.
