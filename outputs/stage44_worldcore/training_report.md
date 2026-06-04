# Stage44 Training Report

- source: `fresh_stage44_worldcore_latent_state_architecture`
- mode: `small`
- rows: `{'train': 12000, 'val': 5000, 'test': 8000}`
- best variant: `hybrid_no_scene`
- verdict: `stage44_worldcore_latent_state_candidate_pass`

| variant | epochs | checkpoint committed | val selected policy |
| --- | ---: | --- | --- |
| `no_baseline_latent` | `3` | `False` | `{'gain_threshold': 0.5, 'harm_threshold': 0.8, 'failure_threshold': 0.0}` |
| `baseline_aware_protected` | `3` | `False` | `{'gain_threshold': 0.0, 'harm_threshold': 0.8, 'failure_threshold': 0.0}` |
| `hybrid_jepa_transformer` | `3` | `False` | `{'gain_threshold': 0.35, 'harm_threshold': 0.8, 'failure_threshold': 0.0}` |
| `hybrid_no_scene` | `3` | `False` | `{'gain_threshold': 0.0, 'harm_threshold': 0.8, 'failure_threshold': 0.0}` |
| `hybrid_no_interaction` | `3` | `False` | `{'gain_threshold': 0.35, 'harm_threshold': 0.8, 'failure_threshold': 0.0}` |
| `hybrid_no_jepa` | `3` | `False` | `{'gain_threshold': 0.2, 'harm_threshold': 0.8, 'failure_threshold': 0.0}` |
| `hybrid_no_transformer_ssm` | `3` | `False` | `{'gain_threshold': 0.0, 'harm_threshold': 0.8, 'failure_threshold': 0.25}` |
