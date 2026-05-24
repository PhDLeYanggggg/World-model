# Stage33 Domain-Conditioned Selector Report

- source: `fresh_run`

| model | all | t50 | hard | easy | regret | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sdd_only_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.843625 | 0.000000 |
| sdd_only_zero_shot_external_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| external_only_selector | -0.004160 | 0.000000 | 0.000000 | 2.522678 | 0.017878 | 0.049780 |
| mixed_domain_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| domain_conditioned_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| coordinate_invariant_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.049780 |
| latent_adapted_selector | -0.018541 | 0.000000 | 0.000000 | 3.526676 | 0.019864 | 0.029978 |
| relative_error_selector | -0.004160 | 0.000000 | 0.000000 | 2.522678 | 0.017878 | 0.049780 |
| failure_assisted_domain_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.000000 |
| conservative_fallback_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 | 0.049780 |
| raw_fde_stage32_style_selector | -0.000744 | 0.000000 | 0.000000 | 0.000000 | 0.017407 | 0.029978 |

- best external model: `sdd_only_zero_shot_external_selector`
