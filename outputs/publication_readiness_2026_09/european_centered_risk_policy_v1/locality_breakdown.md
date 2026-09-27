# Locality and Seed Breakdown

No locality is silently removed when one of its repeated views has an undefined metric.

| Arm | Locality | Centered ADE/floor % | Same-count raw ADE/floor % | Centered hard/floor % | Matched ADE contrast % | Fixed-denominator harm reduction pp |
|---|---|---:|---:|---:|---:|---:|
| subset_pointwise | eu-locality-007 | 0.39309030 | 0.40023705 | 0.71243675 | -0.00716424 | 0.00458623 |
| subset_pointwise | eu-locality-008 | 0.00010951 | 0.00097063 | -0.00000098 | -0.00086126 | 0.00041835 |
| subset_pointwise | eu-locality-020 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_pointwise | eu-locality-048 | 0.00359758 | 0.00461143 | 0.00002625 | -0.00101470 | 0.00018040 |
| subset_pointwise | eu-locality-067 | 0.00264907 | 0.00360590 | 0.00006853 | -0.00095735 | 0.00021418 |
| subset_pointwise | eu-locality-074 | 0.00000045 | 0.00002408 | 0.00000048 | -0.00002363 | 0.00000019 |
| subset_pointwise | eu-locality-082 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_pointwise | eu-locality-110 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_pointwise | eu-locality-112 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_pointwise | eu-locality-119 | 0.63513285 | 0.66513554 | 0.70870154 | -0.03023052 | 0.00654631 |
| subset_pointwise | eu-locality-124 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_pointwise | eu-locality-126 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_aggregate | eu-locality-007 | 0.36193672 | 0.38377154 | 0.65290873 | -0.02194093 | 0.00375201 |
| subset_aggregate | eu-locality-008 | 0.00016191 | 0.00062117 | -0.00000190 | -0.00045932 | 0.00046371 |
| subset_aggregate | eu-locality-020 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_aggregate | eu-locality-048 | 0.00222882 | 0.00340236 | -0.00002462 | -0.00117426 | 0.00007355 |
| subset_aggregate | eu-locality-067 | 0.00162111 | 0.00270660 | 0.00007636 | -0.00108579 | 0.00011087 |
| subset_aggregate | eu-locality-074 | 0.00620030 | 0.00938884 | 0.00780467 | -0.00319393 | 0.00105821 |
| subset_aggregate | eu-locality-082 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_aggregate | eu-locality-110 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_aggregate | eu-locality-112 | 0.00042184 | 0.00083863 | 0.00000000 | -0.00041685 | 0.00021200 |
| subset_aggregate | eu-locality-119 | 0.60255646 | 0.63676074 | 0.67605210 | -0.03443479 | 0.00644762 |
| subset_aggregate | eu-locality-124 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| subset_aggregate | eu-locality-126 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

| Seed | Arm | Centered ADE/floor % | Same-count raw ADE/floor % |
|---|---|---:|---:|
| 17 | subset_pointwise | 0.167768 [0.000048, 0.434069] | 0.168006 [0.000142, 0.434112] |
| 17 | subset_aggregate | 0.181439 [0.000071, 0.469412] | 0.181664 [0.000158, 0.469531] |
| 29 | subset_pointwise | 0.086307 [0.000899, 0.207688] | 0.086777 [0.001153, 0.207905] |
| 29 | subset_aggregate | 0.058629 [0.000557, 0.141002] | 0.059109 [0.000851, 0.141189] |
| 43 | subset_pointwise | 0.004570 [0.000000, 0.011866] | 0.013863 [0.000000, 0.038181] |
| 43 | subset_aggregate | 0.003714 [0.000216, 0.007909] | 0.018599 [0.000420, 0.041381] |
