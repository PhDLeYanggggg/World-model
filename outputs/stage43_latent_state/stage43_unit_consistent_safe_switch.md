# Stage43-I Unit-Consistent Safe-Switch Repair

- source: `fresh_stage43_i_unit_consistent_safe_switch`
- verdict: `stage43_i_unit_consistent_safe_switch_pass`
- gate: `13 / 13`
- deploy Stage43-I candidate: `True`
- keep frozen floor as global default: `True`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt`
- checkpoint committed: `False`
- policy: `domain_capped_prior_easy_guard`
- policy hash: `3f8118f6ba5fbc74c9f724deb6237d090bb014569b3c79247701ed64be29f1e2`

## Why This Stage Exists

Stage43-H showed that Stage43-G has real neural dynamics signal but is not deployable after unit-consistent auditing because easy degradation is unsafe. Stage43-I keeps the same neural candidate but only allows it through a fixed prior easy-risk guard and conservative source/domain switch caps. The guard is not selected on test.

## Deployment Candidate Metrics

- rows: `89736`
- all improvement: `0.231071`
- t50 improvement: `0.113648`
- t100 raw-frame diagnostic: `0.013513`
- hard/failure improvement: `0.244058`
- easy degradation: `0.000000`
- switch rate: `0.185255`
- max domain easy degradation: `0.009491`
- min domain all improvement: `0.066122`
- worst source all improvement: `-0.001034` (`83b0417df499ccae`)

## Bootstrap CI

| metric | rows | mean | ci low | ci high |
| --- | ---: | ---: | ---: | ---: |
| unit_all | 89736 | 0.231071 | 0.227754 | 0.234703 |
| unit_t50 | 21754 | 0.113648 | 0.106796 | 0.119873 |
| unit_t100_raw_frame_diagnostic | 18070 | 0.013513 | 0.009959 | 0.016689 |
| unit_hard_failure | 70119 | 0.244058 | 0.239964 | 0.247802 |
| unit_easy_degradation | 26927 | 0.000000 | 0.000000 | 0.000000 |

## Endpoint Proximity Proxy

- selected near@0.05: `0.034751`
- floor near@0.05: `0.037551`
- near@0.05 delta: `-0.002800`
- selected near@0.10: `0.158018`
- floor near@0.10: `0.160795`
- near@0.10 delta: `-0.002778`

## Policy Comparison

| policy | status | val all | val easy | test all | test t50 | test hard | test easy | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| never_switch_floor | diagnostic_not_deployment_policy | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| stage43g_validation_policy_diagnostic | diagnostic_not_deployment_policy | 0.608075 | 0.000000 | 0.351410 | 0.158059 | 0.377402 | 1.597489 | 1.000000 |
| fixed_prior_stage41_easy_guard_0p03 | diagnostic_not_deployment_policy | 0.359035 | 0.000000 | 0.245659 | 0.128030 | 0.264359 | 0.005005 | 0.215387 |
| domain_capped_prior_easy_guard | domain_capped_deployable_candidate | 0.210776 | 0.000000 | 0.231071 | 0.113648 | 0.244058 | 0.000000 | 0.185255 |
| diagnostic_easy_guard_0.05 | diagnostic_not_deployment_policy | 0.395037 | 0.000000 | 0.276437 | 0.150635 | 0.294361 | 0.021316 | 0.261300 |
| diagnostic_easy_guard_0.10 | diagnostic_not_deployment_policy | 0.443572 | 0.000000 | 0.319511 | 0.183101 | 0.334390 | 0.057768 | 0.345068 |
| diagnostic_easy_guard_0.20 | diagnostic_not_deployment_policy | 0.496196 | 0.000000 | 0.359241 | 0.215155 | 0.370148 | 0.123117 | 0.450176 |
| diagnostic_easy_guard_0.80 | diagnostic_not_deployment_policy | 0.618557 | 0.000000 | 0.390929 | 0.204783 | 0.398275 | 0.654113 | 0.747236 |

## Domain Metrics

| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | 70585 | 0.192474 | 0.001618 | -0.007173 | 0.204081 | 0.009491 | 0.149989 |
| TrajNet | 9611 | 0.066122 | 0.000000 | 0.000000 | 0.074663 | 0.000000 | 0.099990 |
| UCY | 9540 | 0.535715 | 0.475292 | 0.156964 | 0.547875 | 0.000000 | 0.532075 |

## Source-Level Caveat

The worst source-level slice is `83b0417df499ccae` with all improvement `-0.001034`, t50 `0.000000`, and easy degradation `0.000000`. Stage43-I therefore supports a unit-consistent domain-level protected candidate, not a uniform per-source success claim.

## Gate

| gate | passed |
| --- | --- |
| stage43_h_precondition_failed_and_floor_kept | True |
| domain_capped_policy_not_test_selected | True |
| test_eval_completed | True |
| unit_all_ci_low_positive | True |
| unit_t50_ci_low_positive | True |
| unit_hard_ci_low_positive | True |
| easy_preservation_gate | True |
| per_domain_easy_preserved | True |
| per_domain_all_positive | True |
| proximity_not_materially_worse | True |
| partial_switch_not_full_replacement | True |
| t100_reported_diagnostic_only | True |
| no_metric_seconds_stage5c_smc_claim | True |

Conclusion: this repairs the Stage43-G deployment failure under unit-consistent auditing by making neural intervention partial and safety-gated. It remains dataset-local/raw-frame 2.5D evidence with a source-level caveat. Stage5C and SMC remain disabled.
