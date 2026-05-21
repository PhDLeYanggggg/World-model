# Stage 5 Data Discovery Report

## Required Current-State Admission

The current model is a pseudo-3D / 2.5D physics-informed learned residual state-space world model. It is not true 3D and not an exceptional world model. Stage 4.5 score is 64/100, verdict is `prototype_with_repaired_baselines_but_failed_learned_dynamics_gate`, and passed gates are 4/8. TGSIM t+100 is verified; official Stage 4.5 benchmark uses causal_fd velocity; strongest causal baseline is constant_turn_rate_velocity with FDE@100 about 0.04820m; best learned residual FDE@100 is about 1.14498m. Learned residual did not beat the strongest causal baseline, and SMC remains premature.

## Discovery Summary

- Candidate data sources: 26
- Downloaded or legally downloadable according to registry: 13
- Gated / requires application: 11
- Sources that likely support t+100: 25

This is a registry and dry-run discovery stage. Gated datasets are not downloaded, and registry-only datasets are not counted as trained or converted.

## Official Source Notes

- Stanford Drone Dataset: official Stanford UAV data page, non-commercial CC BY-NC-SA 3.0.
- TGSIM: official data.gov / data.transportation.gov public resource.
- Argoverse / Waymo / nuScenes / INTERACTION / leveLX datasets: strong candidates but require terms, application, or large downloads.
- OpenDD: useful map-aware traffic data, but CC BY-ND requires caution around derivative redistribution.

## Top Priority Records

| dataset | domain | status | license | priority | loader |
| --- | --- | --- | --- | --- | --- |
| OpenDD | traffic | downloadable | CC BY-ND 4.0 | 100 | planned |
| TGSIM other public corridors | traffic | downloadable | U.S. public data portal; verify each resource | 95 | adapter_partial |
| TGSIM Foggy Bottom | traffic | downloaded | U.S. public data portal; verify resource terms before redistribution | 90 | working_quick |
| SyntheticPhysicalCrowd2.5D | synthetic | downloaded | project-generated | 85 | working |
| Argoverse 1 Motion Forecasting | driving | requires_application | Argoverse terms | 80 | planned |
| Argoverse 2 Motion Forecasting | driving | requires_application | CC BY-NC-SA 4.0 / Argoverse terms | 80 | planned |
| INTERACTION Dataset | driving | requires_application | INTERACTION dataset terms | 80 | planned |
| Waymo Open Motion Dataset | driving | gated | Waymo Open Dataset terms | 80 | planned |
| nuPlan | driving | requires_application | nuPlan terms | 80 | planned |
| nuScenes Prediction | driving | requires_application | nuScenes terms | 80 | planned |
| NGSIM | traffic | downloadable | U.S. FHWA public research data; verify terms | 75 | planned |
| Stanford Drone Dataset | drone | downloadable | CC BY-NC-SA 3.0 | 75 | planned |
