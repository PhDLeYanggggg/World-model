# Stage44 Ablation Table

| ablation | all delta vs hybrid | t50 delta vs hybrid | hard delta vs hybrid | interaction AUC delta | density MSE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_no_scene` | `-10.74%` | `-10.75%` | `-11.52%` | `-0.029` | `-0.0087` |
| `hybrid_no_interaction` | `-5.44%` | `2.64%` | `-7.28%` | `-0.252` | `-0.0065` |
| `hybrid_no_jepa` | `-5.69%` | `2.68%` | `-7.88%` | `-0.264` | `-0.0037` |
| `hybrid_no_transformer_ssm` | `1.65%` | `3.77%` | `1.54%` | `-0.142` | `0.0028` |
| `baseline_aware_protected` | `-4.09%` | `-5.26%` | `-6.05%` | `-0.217` | `-0.0040` |
| `no_baseline_latent` | `-0.74%` | `3.14%` | `-2.53%` | `-0.169` | `-0.0024` |
