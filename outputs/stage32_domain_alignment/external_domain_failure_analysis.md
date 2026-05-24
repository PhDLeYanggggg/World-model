# Stage32 External Domain Failure Analysis

- source: `fresh_run`
- zero-shot why collapsed: `SDD selector/latent scale is not calibrated to dataset-local external coordinates; conservative switch policy still selects harmful baselines under external feature distribution shift.`
- normalization fixed: `False`
- latent adapter useful: `True`
- external data too short: `True`
- horizon mismatch: `{10: 2020, 25: 1212, 50: 404}`
- agent type issue: `external is pedestrian-only; SDD training includes mixed agent types.`
- scene/goal missing impact: `scene/goal features are mostly zero-filled externally; full multimodal scene transfer cannot be claimed.`
- needs external scene packs: `True`
- SDD-specific selector: `True`
- world model status: `not_cross_domain_candidate`

## Shortest Repair Path
- Build ETH/UCY/OpenTraj scene packs with train-only goals.
- Use coordinate-invariant trajectory tokens and relative-error selector targets.
- Train a domain-conditioned model with held-out external scenes, not only SDD.
- Add homography/scale audit where available before any metric claim.
