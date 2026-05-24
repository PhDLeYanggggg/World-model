# Stage34 Domain-Conditioned Selector v2 Report

- source: `fresh_run`

| model | all | t50 | hard | easy | regret | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| external_strongest_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.518182 | 0.000000 |
| external_oracle_diagnostic | 0.476412 | 0.392410 | 0.431788 | 0.000000 | 0.000000 | 0.565457 |
| external_only_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.518182 | 0.000000 |
| SDD_zero_shot_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.518182 | 0.000000 |
| SDD_external_mixed_selector | -0.015475 | 0.065987 | 0.251271 | 0.302907 | 0.535014 | 0.099835 |
| domain_conditioned_selector | -0.015475 | 0.065987 | 0.251271 | 0.302907 | 0.535014 | 0.099835 |
| domain_mixture_of_experts | -0.028078 | 0.060052 | 0.221798 | 0.217964 | 0.548722 | 0.099835 |
| external_scene_goal_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.518182 | 0.000000 |
| external_relative_error_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.518182 | 0.000000 |
| M3W_latent_plus_external_geometry_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.518182 | 0.000000 |
| conservative_fallback_selector | -0.051910 | 0.066457 | 0.182709 | 0.569532 | 0.574643 | 0.099835 |

- best model: `external_only_selector`
