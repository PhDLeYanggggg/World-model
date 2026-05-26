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

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Data-Card Update

No new metric/time calibration was introduced by Stage42-AB/AC. All updated paper-package claims remain dataset-local raw-frame 2.5D.
<!-- STAGE42_AC_REFRESH:END -->
