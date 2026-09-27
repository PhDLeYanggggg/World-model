# Dimensionless Correction-Fraction Results

Fresh source-development inference and scoring; cached_verified controls. Positive means lower error.
No result is independent confirmation or an intervention-policy safety guarantee.

| Endpoint/subset | vs matched grouped (%) | 95% locality interval | vs prior flat neural (%) | vs train baseline (%) | vs CV (%) |
|---|---:|---|---:|---:|---:|
| ADE_all | +4.721 | [+2.531, +7.885] | +5.154 | +7.916 | +8.462 |
| ADE_positive_easy | +0.950 | [-0.707, +3.355] | +0.789 | +0.961 | -11.169 |
| ADE_hard | +3.874 | [+2.565, +5.083] | +4.352 | +6.328 | +13.355 |
| ADE_zero_CV | undefined | undefined (fixed roster unsupported) | undefined | undefined | undefined |
| FDE_all | +6.492 | [+3.376, +11.099] | +6.945 | +9.915 | +13.398 |
| FDE_positive_easy | +1.886 | [-1.252, +7.097] | +1.655 | +1.718 | -3.839 |
| FDE_hard | +4.499 | [+2.852, +6.194] | +4.827 | +7.001 | +17.913 |
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
| eu-locality-007 | +1.016 | +0.262 | +1.895 | +1.637 |
| eu-locality-008 | +5.613 | +0.350 | +4.959 | +7.998 |
| eu-locality-020 | +19.462 | +12.061 | +5.957 | +28.333 |
| eu-locality-048 | +6.022 | +0.040 | +6.254 | +8.011 |
| eu-locality-067 | +6.251 | -1.075 | +7.121 | +8.539 |
| eu-locality-074 | +4.386 | -0.459 | +5.056 | +5.175 |
| eu-locality-082 | +3.118 | -4.124 | +5.163 | +3.261 |
| eu-locality-110 | +2.224 | +0.323 | +2.195 | +3.022 |
| eu-locality-112 | +3.686 | -0.050 | +3.362 | +5.096 |
| eu-locality-119 | -0.453 | +1.672 | -1.183 | +0.090 |
| eu-locality-124 | +2.471 | +1.697 | +2.615 | +3.039 |
| eu-locality-126 | +2.853 | +0.704 | +3.093 | +3.706 |

## By Seed

| Seed | ADE vs grouped (%) | 95% locality interval |
|---|---:|---|
| 17 | +4.663 | [+2.331, +8.073] |
| 29 | +4.225 | [+2.101, +7.416] |
| 43 | +5.274 | [+3.058, +8.225] |

## By Producer

| Fold | Seed | ADE vs grouped (%) |
|---|---|---:|
| 0 | 17 | +4.703 |
| 0 | 29 | +5.534 |
| 0 | 43 | +6.113 |
| 1 | 17 | +7.839 |
| 1 | 29 | +6.161 |
| 1 | 43 | +8.551 |
| 2 | 17 | +1.448 |
| 2 | 29 | +0.980 |
| 2 | 43 | +1.159 |
