# Agent-Track Topology Results

Fresh source-development inference and scoring; cached_verified controls. Positive means lower error.
No result is independent confirmation or an intervention-policy safety guarantee.

| Endpoint/subset | vs matched flat (%) | 95% locality interval | vs original neural (%) | vs train baseline (%) | vs CV (%) |
|---|---:|---|---:|---:|---:|
| ADE_all | +0.447 | [+0.156, +0.748] | +0.459 | +3.388 | +3.126 |
| ADE_positive_easy | -0.161 | [-0.410, +0.076] | +0.172 | +0.124 | -13.080 |
| ADE_hard | +0.502 | [+0.251, +0.816] | +0.353 | +2.408 | +9.747 |
| ADE_zero_CV | undefined | undefined (fixed roster unsupported) | undefined | undefined | undefined |
| FDE_all | +0.461 | [+0.071, +0.869] | +0.285 | +3.674 | +5.533 |
| FDE_positive_easy | -0.234 | [-0.611, +0.118] | +0.321 | +0.173 | -8.532 |
| FDE_hard | +0.340 | [+0.014, +0.708] | -0.034 | +2.403 | +13.806 |
| FDE_zero_CV | undefined | undefined (fixed roster unsupported) | undefined | undefined | undefined |

Means are equal-locality means of percentage gains, not pooled source-coordinate ratios.
Each locality averages two producer contexts and three seeds before 3,000 locality resamples.
Source localities were already exposed for development; fitting sets and windows overlap.
ADE uses supported requested labels; FDE requires the final requested label.
Easy/hard use fixed training CV-ADE thresholds, including for FDE. Zero-CV gains are undefined.
Absolute costs and positive gain/harm remain reported for zero-reference rows.

## By Locality

| Locality | ADE gain (%) | Easy gain (%) | Hard gain (%) | FDE gain (%) |
|---|---:|---:|---:|---:|
| eu-locality-007 | +0.226 | +0.257 | -0.008 | +0.137 |
| eu-locality-008 | +0.893 | +0.310 | +0.700 | +0.960 |
| eu-locality-020 | -0.589 | -0.955 | +0.034 | -0.902 |
| eu-locality-048 | +1.561 | +0.062 | +1.803 | +2.038 |
| eu-locality-067 | +0.905 | -0.253 | +1.122 | +0.904 |
| eu-locality-074 | +0.529 | +0.325 | +0.123 | +0.467 |
| eu-locality-082 | +0.076 | -0.698 | +0.224 | +0.678 |
| eu-locality-110 | +0.266 | -0.094 | +0.315 | +0.154 |
| eu-locality-112 | -0.020 | -0.397 | +0.144 | -0.134 |
| eu-locality-119 | +0.459 | +0.348 | +0.444 | +0.752 |
| eu-locality-124 | +0.465 | -0.612 | +0.506 | -0.108 |
| eu-locality-126 | +0.588 | -0.223 | +0.624 | +0.593 |

## By Seed

| Seed | ADE vs flat (%) | 95% locality interval |
|---|---:|---|
| 17 | +0.422 | [-0.006, +0.797] |
| 29 | +0.622 | [+0.183, +1.091] |
| 43 | +0.296 | [+0.177, +0.465] |

## By Producer

| Fold | Seed | ADE vs flat (%) |
|---|---|---:|
| 0 | 17 | +0.741 |
| 0 | 29 | +0.117 |
| 0 | 43 | +0.444 |
| 1 | 17 | +0.393 |
| 1 | 29 | +1.581 |
| 1 | 43 | +0.345 |
| 2 | 17 | +0.131 |
| 2 | 29 | +0.168 |
| 2 | 43 | +0.098 |
