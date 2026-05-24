# Stage32 Domain-Adapted Selector Report

- source: `fresh_run`

| model | all | t50 | hard | easy | regret | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| external_only_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| sdd_only_zero_shot_selector | -0.337476 | -1.018801 | -0.095699 | 1.376132 | 0.063891 | 0.049780 |
| sdd_external_mixed_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| sdd_latent_external_adapter_selector | -0.004313 | -0.013021 | -0.024801 | 0.000000 | 0.017900 | 0.049780 |
| domain_conditioned_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| feature_normalized_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| failure_assisted_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| conservative_fallback_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |

- best model: `external_only_selector`
