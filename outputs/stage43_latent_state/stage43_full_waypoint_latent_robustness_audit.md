# Stage43-N Full-Waypoint Latent Robustness Audit

- source: `fresh_stage43_n_full_waypoint_latent_robustness_audit`
- result_source: `fresh_full_test_replay_from_stage43_m_checkpoint`
- gate: `12 / 12`
- verdict: `stage43_n_full_test_positive_with_source_t100_blockers`
- full-test rows: `89736`

## Full-Test Protected Metrics

- full-waypoint ADE improvement: `36.74%`
- endpoint FDE improvement: `44.96%`
- t50 full-waypoint ADE improvement: `33.30%`
- t100 raw-frame diagnostic: `-27.88%`
- hard/failure full-waypoint ADE improvement: `38.45%`
- easy degradation: `0.47%`
- switch rate: `89.53%`

## Domain Breakdown

| domain | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| ETH_UCY | 70585 | 37.10% | 3.42% | 89.43% |
| TrajNet | 9611 | 9.84% | 46.10% | 89.87% |
| UCY | 9540 | 52.55% | 0.00% | 89.96% |

## Horizon Breakdown

| horizon | rows | ADE lift | ungated ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 26132 | 60.19% | 60.30% | 31.23% | 98.42% |
| 100 | 18070 | -27.88% | -34.17% | 17.11% | 81.11% |
| 25 | 23780 | 47.19% | 47.91% | 0.00% | 94.58% |
| 50 | 21754 | 33.30% | 30.46% | 0.00% | 80.33% |

## Source Caveat

- source count: `4`
- negative source count: `1`
- domains with easy harm >2%: `2`
- uniform source success: `False`

Worst source slices:

| source | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt | 1926 | -98.72% | 155.95% | 97.82% |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt | 70585 | 37.10% | 3.42% | 89.43% |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt | 7685 | 38.20% | 23.82% | 87.87% |
| /Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt | 9540 | 52.55% | 0.00% | 89.96% |

## t100 Failure Attribution

- rows: `18070`
- protected t100 improvement: `-27.88%`
- ungated t100 improvement: `-34.17%`
- switch rate: `81.11%`
- diagnosis: t100 remains negative because the neural waypoint shape is worse than the floor on long raw-frame rows; fallback gate still switches too often for t100.

## Boundary

- dataset-local/raw-frame 2.5D only.
- no metric/seconds-level claim.
- no Stage5C execution.
- no SMC.
