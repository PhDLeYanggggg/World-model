# Stage28 M3W-LAS Evaluation Report

- Test split is evaluated once with the validation-selected LAS policy.
- Oracle selector is diagnostic only; Stage28 reports only trained/fallback-safe policy results.

- best variant: `all_latent`
- t+50 improvement: `0.1686288243790961`
- t+100 raw-frame diagnostic improvement: `0.14519818992776146`
- hard/failure improvement: `0.1336398986813968`
- easy degradation: `0.01928694490688554`
- selector regret: `3.4042292296150327`
- switch rate: `0.05`

## Per-Agent-Type Improvement
- Pedestrian: `0.11016732108610128`
- Biker: `0.20528328459658607`
- Skater: `0.20918500363598935`
- Cart: `0.09415699732781369`
- Car: `-0.017048804306335708`
- Bus: `0.13007872802821607`
- unknown: `0.0`
