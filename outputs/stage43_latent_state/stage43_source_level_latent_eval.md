# Stage43-G Source-Level Protected Latent Model

- source: `fresh_stage43_g_source_level_protected_latent`
- verdict: `stage43_g_source_level_latent_candidate_pass`
- gate: `10 / 10`
- deploy neural: `True`
- mode: `full`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt`
- checkpoint committed: `False`
- latent variance: `0.142447`
- source split row hash: `9c8b4d51e0f7a1618dce410c7dd23fbf7f21da5de587d4ae021257775164c3c5`
- safety floor fallback rate: `0.000000`

## Protected Test Metrics vs Safety Floor

- rows: `89736`
- all improvement: `0.858018`
- t50 improvement: `0.821362`
- t100 raw-frame diagnostic: `0.783976`
- hard/failure improvement: `0.866818`
- easy degradation: `0.000000`
- switch rate: `1.000000`

## Safety Floor Interpretation

- intervention status: `floor_not_active_on_test`
- next required audit: bootstrap and safety stress before replacing the frozen Stage37/Stage42 floor
- Because switch rate is part of the evidence, a full-switch result must not be described as final floor replacement until bootstrap and safety stress pass.

## Domain Metrics

| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | 70585 | 0.845031 | 0.772971 | 0.756211 | 0.849357 | 0.000000 | 1.000000 |
| TrajNet | 9611 | 0.832857 | 0.867602 | 0.903710 | 0.873446 | 0.000000 | 1.000000 |
| UCY | 9540 | 0.939509 | 0.950590 | 0.879977 | 0.945827 | 0.000000 | 1.000000 |

## Ungated Neural Diagnostic

- all improvement: `0.858018`
- t50 improvement: `0.821362`
- hard/failure improvement: `0.866818`
- easy degradation: `0.000000`

No Stage5C, no SMC, no metric/seconds-level claim, no true-3D/foundation claim.
