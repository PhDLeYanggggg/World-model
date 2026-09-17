# Offline Annotated Visual Forecasting: Full Fit Cohort

Exploratory leave-one-physical-fit-scene-out evaluation. Not independent test evidence.

| Arm | Equal-scene/seed gain vs CV (%) | Vs training-selected strongest (%) | Positive CV folds |
| --- | ---: | ---: | ---: |
| geometry | -0.5767 | -0.5767 | 0/9 |
| mask_only | -0.5692 | -0.5692 | 0/9 |
| current_rgb | -0.6627 | -0.6627 | 0/9 |
| past_rgb | -0.7401 | -0.7401 | 0/9 |

| Arm | Seed | Held fold | Primary gain vs CV (%) | Easy degradation (%) |
| --- | ---: | ---: | ---: | ---: |
| geometry | 17 | 0 | -0.0209 | 178.16461084992733 |
| mask_only | 17 | 0 | -0.0185 | 179.93164052505026 |
| current_rgb | 17 | 0 | -0.3069 | 251.4981104590492 |
| past_rgb | 17 | 0 | -0.3463 | 290.27464996606966 |
| geometry | 17 | 1 | -0.1515 | 2090.5903971839257 |
| mask_only | 17 | 1 | -0.1646 | 2179.51564631636 |
| current_rgb | 17 | 1 | -0.1711 | 2050.7791903787793 |
| past_rgb | 17 | 1 | -0.2796 | 1734.0096869594136 |
| geometry | 17 | 2 | -6.9918 | 6449.884093150995 |
| mask_only | 17 | 2 | -6.6817 | 6050.859349494462 |
| current_rgb | 17 | 2 | -5.9373 | 5085.951209024091 |
| past_rgb | 17 | 2 | -6.1077 | 4892.594710650715 |
| geometry | 29 | 0 | -0.0938 | 170.8443785776265 |
| mask_only | 29 | 0 | -0.1083 | 186.01407607997925 |
| current_rgb | 29 | 0 | -0.2961 | 187.66798666815228 |
| past_rgb | 29 | 0 | -0.2700 | 181.37342161692428 |
| geometry | 29 | 1 | -0.2442 | 2526.320196234911 |
| mask_only | 29 | 1 | -0.2332 | 2466.745641074596 |
| current_rgb | 29 | 1 | -0.3245 | 2463.337764306364 |
| past_rgb | 29 | 1 | -0.3097 | 2012.4493749799033 |
| geometry | 29 | 2 | -6.2712 | 5519.969704913989 |
| mask_only | 29 | 2 | -6.1468 | 5378.476405423633 |
| current_rgb | 29 | 2 | -4.8605 | 3654.890012305979 |
| past_rgb | 29 | 2 | -5.9227 | 3963.519355384767 |
| geometry | 43 | 0 | -0.0537 | 215.94219013137356 |
| mask_only | 43 | 0 | -0.0611 | 216.0827306717828 |
| current_rgb | 43 | 0 | -0.3484 | 314.2654441065902 |
| past_rgb | 43 | 0 | -0.3498 | 313.3324887065866 |
| geometry | 43 | 1 | -0.0894 | 1999.8076895217378 |
| mask_only | 43 | 1 | -0.0795 | 1995.131751781859 |
| current_rgb | 43 | 1 | -0.2008 | 1739.9097435548824 |
| past_rgb | 43 | 1 | -0.2980 | 1655.044236698275 |
| geometry | 43 | 2 | -6.0363 | 5240.9016435812755 |
| mask_only | 43 | 2 | -6.0662 | 5303.232043401559 |
| current_rgb | 43 | 2 | -5.1294 | 3830.449811129207 |
| past_rgb | 43 | 2 | -5.9297 | 4337.1335118794605 |

Offline interpolated annotations are not strict online observations. Zara03 missing imagery is retained.
Three physical fit scenes are not a confirmation set. No development/test tuning, metric/seconds or deployment claim.
