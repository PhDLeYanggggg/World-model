# Stage35 External Selector v3 Report

- source: `fresh_run`

| model | all | t50 | hard | easy | regret | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| external_strongest_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.556671 | 0.000000 |
| external_oracle_diagnostic | 0.528738 | 0.229828 | 0.522183 | 0.000000 | 0.000000 | 0.997828 |
| external_only_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| domain_conditioned_selector_v2 | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| selective_transfer_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| hard_only_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| easy_safe_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| goal_aware_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| interaction_aware_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |
| M3W_latent_plus_external_geometry_selector | 0.121319 | 0.000000 | 0.139849 | 0.000411 | 0.428943 | 0.049998 |

- best model: `external_only_selector`
- deployable: `False`
