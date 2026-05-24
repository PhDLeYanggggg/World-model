# Stage30 F World Model Capability Audit

- source: `fresh_run`
- M3W-LAS 是否只是 selector trick：`partly_selector_policy_but_latent_features_improve_selector_decisions`
- latent 是否有贡献：`True`
- latent t50 delta：`0.059488896481791745`
- goal 是否有贡献：`False`
- goal t50 delta：`0.00011240042697294172`
- interaction 是否有贡献：`True`
- interaction hard delta：`0.00021180243131857512`
- interaction high-density delta：`0.0014998137488503567`
- interaction contribution scope：`high_density_subset`
- cross_scene 是否稳定：`True`
- 是否有外部泛化：`False`
- external blocker：`Full M3W-LAS all_latent transfer needs external latent cache and scale calibration; converted base feature store is diagnostic only.`
- 是否仍只是 SDD pixel-space 候选：`True`
- Stage5C executed: `False`
- SMC enabled: `False`
