# Stage34 Failure Analysis

- source: `fresh_run`
- row geometry complete: `True`
- per-row goal distance lift: `False`
- scene packs lift: `0.0`
- relative target lift: `0.0`
- diagnostic domain-conditioned t+50 lift: `0.06598701408505248`
- diagnostic domain-conditioned hard lift: `0.2512714845174614`
- diagnostic domain-conditioned easy degradation: `0.30290742479725674`
- latent adapter predictive lift: `False`
- latent adapter gap only: `True`
- horizon mismatch main factor: `True`
- agent-type mismatch still factor: `True`
- external data too short: `True`
- M3W still SDD-specific: `True`
- world model status: `not_cross_domain_candidate`

## Shortest Repair Path
- Need more held-out external t+50/t+100 rows; current test t+50 is small and t+100 absent.
- Need image/homography-backed scene packs; current walkable map is scene-bounds proxy.
- Need per-domain held-out-scene training rather than only OpenTraj TrajNet file split.
- Need verify whether goal-directed baseline is genuinely predictive in external data before using it as selector target.
