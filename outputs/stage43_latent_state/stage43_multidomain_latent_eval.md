# Stage43-E Multidomain Latent Evaluation

- source: `fresh_stage43_e_multidomain_latent_eval`
- verdict: `stage43_e_multidomain_latent_eval_blocker_mapped`
- gate: `8 / 8`
- multi-domain latent candidate: `False`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt`
- checkpoint committed: `False`

## Heldout Coverage

- train domains: `['ETH_UCY', 'TrajNet']`
- val domains: `['TrajNet']`
- test domains: `['UCY']`
- missing heldout domains: `['ETH_UCY', 'TrajNet']`
- required next split: source-level or scene-level heldout split containing ETH_UCY, TrajNet, and UCY without test endpoint goal leakage

## Split Metrics

| split | role | rows | domains | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `train_seen` | 158942 | `['ETH_UCY', 'TrajNet']` | 0.250650 | 0.140011 | 0.008027 | 0.249244 | 0.037759 | 0.254716 |
| val | `validation_seen` | 112746 | `['TrajNet']` | 0.179296 | 0.082303 | 0.044827 | 0.180618 | 0.054655 | 0.189106 |
| test | `heldout_test` | 66303 | `['UCY']` | 0.163151 | 0.136820 | 0.009722 | 0.164765 | 0.000000 | 0.170113 |

## UCY Heldout Test

- all improvement: `0.163151`
- t50 improvement: `0.136820`
- t100 raw-frame diagnostic: `0.009722`
- hard/failure improvement: `0.164765`
- easy degradation: `0.000000`

## Gate

| gate | passed |
| --- | --- |
| checkpoint_exists | True |
| train_val_test_evaluated | True |
| all_observed_domains_reported | True |
| ucy_heldout_positive | True |
| seen_domain_diagnostics_completed | True |
| multi_domain_heldout_blocker_recorded | True |
| no_multi_domain_overclaim | True |
| no_metric_seconds_stage5c_smc_claim | True |

Conclusion: current Stage43 latent-state model has UCY heldout support and seen/validation domain diagnostics, but it is not yet a multi-domain latent world model candidate because ETH_UCY and TrajNet are not held-out test domains in this split.
