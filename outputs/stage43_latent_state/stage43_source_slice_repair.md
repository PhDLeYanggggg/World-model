# Stage43-K Source-Slice Repair

- source: `fresh_stage43_k_source_slice_repair`
- verdict: `stage43_k_source_slice_negative_repaired`
- gate: `12 / 12`
- source-safe candidate: `True`
- uniform positive source candidate: `False`
- policy: `validation_source_family_guarded_safe_switch`
- policy hash: `e3ba25ea629c538b9e2c704a47acde68b432fcfc4f2e8c68358f4579fcf01732`

## Why This Stage Exists

Stage43-J found that Stage43-I is a domain-level unit-consistent safe-switch candidate, but not a uniform source-level success: one small TrajNet source was slightly negative. Stage43-K repairs that negative source without using test-source threshold tuning.

The repair starts from the Stage43-I safe switch and adds a validation-only source-family support guard. Source families that are missing or unsafe on validation fall back to the frozen floor. This can repair harm, but it does not turn a floored source into positive transfer.

## Deployment Metrics

- rows: `89736`
- all improvement: `0.231096`
- t50 improvement: `0.113648`
- t100 raw-frame diagnostic: `0.013513`
- hard/failure improvement: `0.244058`
- easy degradation: `0.000000`
- switch rate: `0.185199`
- negative source count: `0`
- min source all improvement: `0.000000`
- max source easy degradation: `0.009491`
- blocked test source families: `TrajNet_mot`

## Bootstrap CI

| metric | rows | mean | ci low | ci high |
| --- | ---: | ---: | ---: | ---: |
| unit_all | 89736 | 0.231096 | 0.227633 | 0.234560 |
| unit_t50 | 21754 | 0.113648 | 0.106998 | 0.120688 |
| unit_t100_raw_frame_diagnostic | 18070 | 0.013513 | 0.010352 | 0.016984 |
| unit_hard_failure | 70119 | 0.244058 | 0.240332 | 0.247711 |
| unit_easy_degradation | 26927 | 0.000000 | 0.000000 | 0.000000 |

## Proximity Proxy

- selected near@0.05: `0.034751`
- floor near@0.05: `0.037551`
- near@0.05 delta: `-0.002800`
- selected near@0.10: `0.158018`
- floor near@0.10: `0.160795`
- near@0.10 delta: `-0.002778`

## Validation Source-Family Decision

| family | val rows | val all | val easy | allowed |
| --- | ---: | ---: | ---: | --- |
| ETH_UCY | 16611 | 0.124223 | 0.000000 | True |
| TrajNet_biwi | 459 | 0.100372 | 0.000000 | True |
| TrajNet_crowds | 37153 | 0.114646 | 0.000000 | True |
| UCY | 47223 | 0.338653 | 0.000000 | True |

## Source Metrics

| source | domains | scenes | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 603561046c16aa3a | TrajNet | TrajNet_biwi | 7685 | 0.086855 | 0.000000 | 0.000000 | 0.086855 | 0.000000 | 0.124398 |
| 7c5c7053cb97abc4 | UCY | UCY_crowds | 9540 | 0.535715 | 0.475292 | 0.156964 | 0.547875 | 0.000000 | 0.532075 |
| 83b0417df499ccae | TrajNet | TrajNet_mot | 1926 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| d9c8e12e56a017c8 | ETH_UCY | ETH_UCY_students03 | 70585 | 0.192474 | 0.001618 | -0.007173 | 0.204081 | 0.009491 | 0.149989 |

## Gate

| gate | passed |
| --- | --- |
| stage43_j_precondition_present | True |
| validation_family_policy_used | True |
| negative_source_repaired | True |
| source_easy_preserved | True |
| unit_all_ci_low_positive | True |
| unit_t50_ci_low_positive | True |
| unit_hard_ci_low_positive | True |
| easy_preservation_gate | True |
| proximity_not_materially_worse | True |
| partial_switch_not_full_replacement | True |
| uniform_positive_source_claim_blocked | True |
| no_metric_seconds_stage5c_smc_claim | True |

Conclusion: Stage43-K repairs the negative source-slice harm by adding a validation-only source-family support guard. It supports a source-safe protected candidate, not a uniform positive per-source claim. Stage5C and SMC remain disabled.
