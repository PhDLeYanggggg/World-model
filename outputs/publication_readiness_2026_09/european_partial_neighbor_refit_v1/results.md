# Matched Partial-Neighbor Neural Results

Fresh source-development neural refit, not independent confirmation. Positive is lower error.

| Endpoint/subset | vs legacy neural (%) | 95% locality interval | vs training-selected baseline (%) | vs CV (%) |
|---|---:|---|---:|---:|
| ADE_all | +0.012 | [-0.374, +0.412] | +2.927 | +2.719 |
| ADE_positive_easy | +0.335 | [+0.015, +0.637] | +0.264 | -12.888 |
| ADE_hard | -0.152 | [-0.628, +0.324] | +1.899 | +9.286 |
| ADE_zero_CV | undefined | undefined (fixed roster unsupported) | undefined | undefined |
| FDE_all | -0.181 | [-0.604, +0.258] | +3.187 | +5.159 |
| FDE_positive_easy | +0.557 | [+0.369, +0.751] | +0.384 | -8.229 |
| FDE_hard | -0.380 | [-0.870, +0.092] | +2.048 | +13.519 |
| FDE_zero_CV | undefined | undefined (fixed roster unsupported) | undefined | undefined |

These are averages of locality-specific percentage gains, not a pooled-pixel ratio.
Each locality averages two producer contexts and three seeds before 3,000 resamples.
The 12 source localities were already development-exposed; producer fits overlap.
ADE uses available requested future labels; FDE requires the final requested label.
Easy/hard membership uses training-derived CV-ADE cuts, including for FDE reporting.
FDE gains compare FDE against FDE. Zero-reference percentages remain undefined.

## By Locality

| Locality | ADE vs legacy (%) | easy gain (%) | hard gain (%) | FDE gain (%) |
|---|---:|---:|---:|---:|
| eu-locality-007 | +0.353 | +0.143 | +0.265 | +0.399 |
| eu-locality-008 | +0.593 | +0.602 | +0.308 | +0.352 |
| eu-locality-020 | +0.219 | +0.154 | -0.694 | +0.930 |
| eu-locality-048 | -0.172 | +0.867 | -0.398 | -0.604 |
| eu-locality-067 | -0.935 | +0.920 | -1.229 | -1.098 |
| eu-locality-074 | +0.170 | -0.116 | +0.250 | +0.167 |
| eu-locality-082 | -1.539 | -0.246 | -2.111 | -1.668 |
| eu-locality-110 | -0.362 | +0.208 | -0.473 | -0.902 |
| eu-locality-112 | +0.229 | +0.914 | +0.393 | +0.246 |
| eu-locality-119 | +1.359 | +1.116 | +1.399 | +0.857 |
| eu-locality-124 | -0.142 | -0.885 | -0.097 | -0.496 |
| eu-locality-126 | +0.373 | +0.346 | +0.564 | -0.360 |

## By Training Seed

| Seed | ADE gain (%) | Locality interval |
|---|---:|---|
| 17 | -0.122 | [-0.577, +0.318] |
| 29 | +0.137 | [-0.254, +0.558] |
| 43 | +0.021 | [-0.400, +0.532] |

## By Producer

| Fold | Seed | Held localities | ADE gain (%) |
|---|---|---:|---:|
| 0 | 17 | 8 | -0.236 |
| 0 | 29 | 8 | +0.183 |
| 0 | 43 | 8 | +0.112 |
| 1 | 17 | 8 | -0.345 |
| 1 | 29 | 8 | -0.297 |
| 1 | 43 | 8 | -0.397 |
| 2 | 17 | 8 | +0.215 |
| 2 | 29 | 8 | +0.524 |
| 2 | 43 | 8 | +0.350 |
