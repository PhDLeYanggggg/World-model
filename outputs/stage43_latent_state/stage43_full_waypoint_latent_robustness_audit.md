# Stage43-N Full-Waypoint Latent Robustness Audit

- source: `fresh_stage43_n_full_waypoint_latent_robustness_audit`
- result_source: `fresh_full_test_replay_from_stage43_m_checkpoint`
- gate: `12 / 12`
- verdict: `stage43_n_full_test_positive_with_source_t100_blockers`
- full-test rows: `89736`

## Full-Test Protected Metrics

- full-waypoint ADE improvement: `29.42%`
- endpoint FDE improvement: `39.28%`
- t50 full-waypoint ADE improvement: `16.60%`
- t100 raw-frame diagnostic: `-16.57%`
- hard/failure full-waypoint ADE improvement: `28.66%`
- easy degradation: `0.00%`
- switch rate: `68.31%`

## Domain Breakdown

| domain | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| ETH_UCY | 70585 | 29.40% | 0.00% | 66.34% |
| TrajNet | 9611 | 6.35% | 19.81% | 66.47% |
| UCY | 9540 | 44.75% | 0.00% | 84.74% |

## Horizon Breakdown

| horizon | rows | ADE lift | ungated ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 26132 | 50.20% | 39.14% | 0.00% | 86.17% |
| 100 | 18070 | -16.57% | -72.28% | 12.58% | 28.57% |
| 25 | 23780 | 40.40% | 33.81% | 0.00% | 82.91% |
| 50 | 21754 | 16.60% | 15.81% | 0.00% | 63.89% |

## Source Caveat

- source count: `4`
- negative source count: `1`
- domains with easy harm >2%: `1`
- uniform source success: `False`

Worst source slices:

| source | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt | 1926 | -115.33% | 164.01% | 96.31% |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt | 70585 | 29.40% | 0.00% | 66.34% |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt | 7685 | 38.13% | 0.00% | 58.99% |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt | 9540 | 44.75% | 0.00% | 84.74% |

## t100 Failure Attribution

- rows: `18070`
- protected t100 improvement: `-16.57%`
- ungated t100 improvement: `-72.28%`
- switch rate: `28.57%`
- diagnosis: t100 remains negative because the neural waypoint shape is worse than the floor on long raw-frame rows; fallback gate still switches too often for t100.

## Boundary

- dataset-local/raw-frame 2.5D only.
- no metric/seconds-level claim.
- no Stage5C execution.
- no SMC.
