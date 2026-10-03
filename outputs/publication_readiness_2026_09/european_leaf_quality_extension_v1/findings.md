# Leaf-Quality Extension Results

One preregistered extension, no refit or threshold search. The frozen controls are
cached_verified; all extension inference and policy readouts are fresh_run.
This is exposed development evidence, not independent confirmation.

| Arm | Selected | Unknown | Complete support / 72 | Known violations | Upper violations | Worst upper % |
|---|---:|---:|---:|---:|---:|---:|
| original | 95455 | 918 | 33 | 4 | 7 | 5.4058 |
| additive | 112456 | 1143 | 19 | 20 | 42 | 1200.1684 |
| poisson | 111031 | 1050 | 37 | 2 | 11 | 18.0277 |
| cost | 96720 | 926 | 36 | 4 | 7 | 5.7195 |
| extended | 96718 | 926 | 36 | 4 | 7 | 5.7195 |

| Extension contrast | Mean | Nominal 95% locality CI |
|---|---:|---|
| extended_minus_original_signed_MSE | -0.02185213 | [-0.05789952, -0.00017524] |
| extended_minus_original_full_utility_percent | +0.00575491 | [+0.00078285, +0.01342375] |
| extended_minus_original_matched_utility_percent | +0.00271130 | [+0.00038070, +0.00571592] |
| extended_minus_additive_signed_MSE | +0.01029426 | [-0.00153151, +0.02628043] |
| extended_minus_additive_full_utility_percent | -0.28022030 | [-0.48566702, -0.11711521] |
| extended_minus_additive_matched_utility_percent | -0.06185285 | [-0.10867003, -0.01949651] |
| extended_minus_poisson_signed_MSE | -0.15801318 | [-0.29262240, -0.04890369] |
| extended_minus_poisson_full_utility_percent | -0.03806605 | [-0.06551993, -0.01754745] |
| extended_minus_poisson_matched_utility_percent | -0.02092841 | [-0.04064467, -0.00620165] |
| extended_minus_cost_signed_MSE | -0.12600536 | [-0.24678343, -0.03094699] |
| extended_minus_cost_full_utility_percent | +0.00001480 | [-0.00000312, +0.00004134] |
| extended_minus_cost_matched_utility_percent | +0.00000997 | [+0.00000000, +0.00002446] |

Utility is a percentage of full known reference error mass, not ADE/FDE gain.
The 3,000-resample intervals use locality blocks; repeated heads and windows are not
independent units. Intervals are nominal, not adjusted for prior development search.

| Locality | Quality changed fraction | MSE vs cost | MSE vs original | Full utility vs original % |
|---|---:|---:|---:|---:|
| eu-locality-007 | 0.780218 | -0.045923 | +0.012918 | +0.000026 |
| eu-locality-008 | 0.411304 | -0.004423 | -0.006057 | +0.005120 |
| eu-locality-020 | 0.796236 | -0.000738 | -0.026780 | -0.000020 |
| eu-locality-048 | 0.578496 | -0.002334 | -0.011217 | +0.001969 |
| eu-locality-067 | 0.644381 | -0.258998 | -0.002309 | +0.000201 |
| eu-locality-074 | 0.425284 | -0.016719 | -0.010268 | +0.000057 |
| eu-locality-082 | 0.710730 | +0.002909 | -0.017063 | +0.000075 |
| eu-locality-110 | 0.611240 | -0.645618 | +0.007570 | +0.014677 |
| eu-locality-112 | 0.744431 | -0.099948 | -0.008610 | +0.000000 |
| eu-locality-119 | 0.719708 | -0.035170 | +0.011372 | +0.004969 |
| eu-locality-124 | 0.868183 | -0.389122 | -0.200972 | +0.000000 |
| eu-locality-126 | 0.710984 | -0.015980 | -0.010811 | +0.041985 |

## Registered Screen

Advance to transfer: False.
Every head supported and within the absolute easy-risk budget: False.

- MSE_not_supported_vs_additive
- full_utility_not_supported_vs_additive
- matched_utility_not_supported_vs_additive
- full_utility_not_supported_vs_poisson
- matched_utility_not_supported_vs_poisson
- full_utility_not_supported_vs_cost
- matched_utility_not_supported_vs_cost
- worst_upper_risk_not_preserved

No deployment change, Stage5C or SMC. Image-local rawstride12 obs8/pred12, detector-silver.
No seconds, metric, physical safety, true 3D, foundation or submission-ready claim.
