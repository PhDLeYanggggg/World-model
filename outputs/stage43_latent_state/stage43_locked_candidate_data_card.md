# Stage43 Locked Candidate Data Card

## Units

- SDD remains pixel-space.
- External top-down rows remain dataset-local/raw-frame unless separately calibrated.
- t50 and t100 are raw-frame horizons, not seconds-level claims.

## Source Guard

- external domains in current matrix: `['ETH_UCY', 'TrajNet', 'UCY']`
- source-level test rows: `89736`
- blocked source ready rows: `0`
- blocked source training allowed now: `0`

## Leakage Boundary

- Future endpoint/full-waypoint labels are loss/eval only.
- No central velocity official input.
- No test endpoint goal construction.
- No test statistics normalization.
