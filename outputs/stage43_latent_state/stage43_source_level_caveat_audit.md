# Stage43-J Source-Level Caveat Audit

- source: `fresh_stage43_j_source_level_caveat_audit`
- verdict: `stage43_j_source_level_caveat_mapped`
- gate: `7 / 7`
- source uniform candidate: `False`
- domain-level candidate: `True`
- repair attempted: `False`

## Finding

Stage43-I is a unit-consistent domain-level protected latent candidate, but it is not a uniform per-source success claim. This audit intentionally blocks that overclaim before the result is used in a paper package.

- source count: `4`
- nonpositive all-improvement source count: `1`
- nonpositive t50 source count: `2`
- worst source: `83b0417df499ccae`
- worst source all improvement: `-0.001034`
- worst source t50 improvement: `0.000000`
- worst source easy degradation: `0.000000`

## Source Metrics

| source | domains | scenes | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 603561046c16aa3a | TrajNet | TrajNet_biwi | 7685 | 0.086855 | 0.000000 | 0.000000 | 0.086855 | 0.000000 | 0.124398 |
| 7c5c7053cb97abc4 | UCY | UCY_crowds | 9540 | 0.535715 | 0.475292 | 0.156964 | 0.547875 | 0.000000 | 0.532075 |
| 83b0417df499ccae | TrajNet | TrajNet_mot | 1926 | -0.001034 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.002596 |
| d9c8e12e56a017c8 | ETH_UCY | ETH_UCY_students03 | 70585 | 0.192474 | 0.001618 | -0.007173 | 0.204081 | 0.009491 | 0.149989 |

## Gate

| gate | passed |
| --- | --- |
| stage43_i_passed | True |
| source_metrics_present | True |
| nonpositive_source_detected | True |
| uniform_source_claim_blocked | True |
| domain_level_claim_preserved | True |
| no_test_tuned_repair_attempted | True |
| claim_boundary_recorded | True |

Conclusion: Stage43-I should be described as a protected domain-level candidate with source-level caveats. The next repair must use train/validation-only source-family gating or source-balanced retraining, not test-source threshold tuning.
