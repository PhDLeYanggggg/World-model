# Stage28 M3W-LAS Training Report

- M3W-LAS trains cost-aware baseline selectors with frozen M3W latent features.
- It does not train ordinary residual correction, latent generative rollout, or SMC.
- SDD remains pixel-space raw-frame; no metric or seconds-level claim.

- best variant: `all_latent`
- selected policy: `{'variant': 'all_latent', 'confidence_threshold': 0.0, 'predicted_gain_threshold_px': 0.0, 'min_predicted_margin_px': 0.0, 'easy_predicted_strongest_threshold_px': 10.0, 'failure_probability_threshold': None, 'max_switch_rate': 0.05}`
- t+50 improvement: `0.1686288243790961`
- hard/failure improvement: `0.1336398986813968`
- easy degradation: `0.01928694490688554`
- beats Stage26 t+50: `True`
- beats Stage26 hard/failure: `True`
- candidate v2: `True`

| variant | model | feature dim | val t50 | test t50 | test hard | easy degradation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage26_only | extra_trees | 57 | 0.256446 | 0.125127 | 0.115358 | 0.018596 |
| plus_jepa | ridge | 217 | 0.332916 | 0.166755 | 0.131590 | 0.020609 |
| plus_transformer | ridge | 217 | 0.340644 | 0.162020 | 0.125941 | 0.020026 |
| plus_hybrid | extra_trees | 217 | 0.332609 | 0.162655 | 0.130423 | 0.018117 |
| plus_failure_hidden | ridge | 59 | 0.281985 | 0.118883 | 0.081912 | 0.031096 |
| plus_interaction_hidden | ridge | 59 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| all_latent | extra_trees | 541 | 0.271762 | 0.168629 | 0.133640 | 0.019287 |
| all_latent_fallback | extra_trees | 541 | 0.271762 | 0.168629 | 0.133640 | 0.019287 |
