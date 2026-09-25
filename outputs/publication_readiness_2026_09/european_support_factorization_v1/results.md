# Fixed-Support Factorization Results

All 936 registered development views completed. No thresholds or neural weights refitted.
Forecasts, heads and population: cached_verified. Decisions, factor attribution and evaluations: fresh_run.
Ranges span 36 correlated views per policy; CI counts are not independent tests or corrected significance.

| Parent / policy | All ADE gain vs floor (%) | All gain vs stop (%) | Positive / negative CI vs stop | Hard gain vs stop (%) | Worst easy locality gain vs CV (%) |
|---|---:|---:|---:|---:|---:|
| cv_targets__stop | 0.1249 to 1.7634 | 0 to 0 | 0 / 0 | 0 to 0 | -0.3001 |
| cv_targets__history | 0.1096 to 1.6647 | -0.1047 to -0.0153 | 0 / 30 | -0.1317 to 0.0061 | -0.4362 |
| cv_targets__history_risk | 0.1096 to 1.7081 | -0.1286 to -0.0151 | 0 / 29 | -0.1250 to 0.0096 | -0.4362 |
| cv_targets__history_random | 0.1096 to 1.6949 | -0.0967 to -0.0144 | 0 / 30 | -0.0861 to 0.0053 | -0.4362 |
| cv_targets__disagreement | 0.1119 to 1.7245 | -0.0669 to 0.0055 | 0 / 27 | -0.1059 to 0.0103 | -0.3699 |
| cv_targets__disagreement_risk | 0.1119 to 1.7486 | -0.0533 to 0.0055 | 0 / 25 | -0.0565 to 0.0132 | -0.3699 |
| cv_targets__disagreement_random | 0.1119 to 1.7359 | -0.0569 to 0.0058 | 0 / 30 | -0.0653 to 0.0123 | -0.3699 |
| cv_targets__separate | 0.1053 to 1.6510 | -0.1244 to -0.0179 | 0 / 31 | -0.1491 to 0.0023 | -0.4362 |
| cv_targets__separate_risk | 0.1053 to 1.7006 | -0.1290 to -0.0188 | 0 / 31 | -0.1253 to 0.0053 | -0.4362 |
| cv_targets__separate_random | 0.1053 to 1.6937 | -0.0976 to -0.0180 | 0 / 31 | -0.0929 to 0.0000 | -0.4362 |
| cv_targets__joint | 0.0883 to 1.6472 | -0.1279 to -0.0181 | 0 / 36 | -0.1560 to 0.0020 | -0.4369 |
| cv_targets__joint_risk | 0.0883 to 1.6976 | -0.1392 to -0.0188 | 0 / 36 | -0.1385 to 0.0053 | -0.4369 |
| cv_targets__joint_random | 0.0883 to 1.6909 | -0.1046 to -0.0180 | 0 / 36 | -0.0977 to 0.0000 | -0.4369 |
| floor_both__stop | 0.1225 to 1.6410 | 0 to 0 | 0 / 0 | 0 to 0 | 0.0777 |
| floor_both__history | 0.1064 to 1.5412 | -0.1041 to 0.0059 | 0 / 28 | -0.1159 to 0.0092 | -0.0501 |
| floor_both__history_risk | 0.1064 to 1.5814 | -0.0947 to 0.0106 | 0 / 25 | -0.0858 to 0.0138 | -0.0501 |
| floor_both__history_random | 0.1064 to 1.5559 | -0.0936 to 0.0087 | 0 / 27 | -0.1034 to 0.0144 | 0.0200 |
| floor_both__disagreement | 0.1124 to 1.6031 | -0.0642 to 0.0054 | 0 / 26 | -0.0878 to 0.0111 | 0.0076 |
| floor_both__disagreement_risk | 0.1124 to 1.6214 | -0.0530 to 0.0059 | 0 / 24 | -0.0588 to 0.0143 | 0.0076 |
| floor_both__disagreement_random | 0.1124 to 1.6090 | -0.0571 to 0.0063 | 0 / 26 | -0.0646 to 0.0140 | 0.0777 |
| floor_both__separate | 0.1051 to 1.5278 | -0.1190 to -0.0149 | 0 / 29 | -0.1335 to 0.0092 | -0.0501 |
| floor_both__separate_risk | 0.1051 to 1.5670 | -0.1045 to -0.0143 | 0 / 31 | -0.0917 to 0.0131 | -0.0501 |
| floor_both__separate_random | 0.1051 to 1.5491 | -0.1010 to -0.0139 | 0 / 30 | -0.1115 to 0.0130 | 0.0200 |
| floor_both__joint | 0.0880 to 1.5241 | -0.1225 to -0.0149 | 0 / 33 | -0.1440 to 0.0092 | -0.0509 |
| floor_both__joint_risk | 0.0881 to 1.5633 | -0.1079 to -0.0146 | 0 / 33 | -0.0991 to 0.0131 | -0.0509 |
| floor_both__joint_random | 0.0880 to 1.5457 | -0.1041 to -0.0139 | 0 / 33 | -0.1158 to 0.0130 | 0.0192 |

## Comparison With The Joint Box

Recovering benefit relative to a harmful filter is not improvement over the unchanged stop policy.

| Policy | All gain vs joint (%) | Positive / negative CI | Complete-label gain vs joint (%) |
|---|---:|---:|---:|
| cv_targets__history | 0.0001 to 0.0274 | 28 / 0 | 0.0001 to 0.0305 |
| cv_targets__disagreement | 0.0093 to 0.0798 | 33 / 0 | 0.0037 to 0.0572 |
| cv_targets__separate | 0.0000 to 0.0181 | 22 / 0 | -0.0001 to 0.0049 |
| floor_both__history | 0.0001 to 0.0305 | 25 / 0 | 0.0001 to 0.0256 |
| floor_both__disagreement | 0.0089 to 0.0803 | 28 / 0 | 0.0016 to 0.0609 |
| floor_both__separate | -0.0000 to 0.0193 | 18 / 0 | -0.0002 to 0.0147 |

## Same-Recording, Same-Frame Count-Matched Controls

| Policy | Control | All gain (%) | Positive / negative CI | Easy gain (%) | Hard gain (%) |
|---|---|---:|---:|---:|---:|
| cv_targets__history | risk | -0.0473 to 0.0426 | 6 / 6 | -0.0008 to 0.1428 | -0.0811 to 0.0505 |
| cv_targets__history | random | -0.0388 to 0.0045 | 5 / 4 | -0.0006 to 0.1902 | -0.0662 to 0.0032 |
| cv_targets__disagreement | risk | -0.0476 to 0.0004 | 2 / 5 | -0.0060 to 0.0256 | -0.0711 to 0.0000 |
| cv_targets__disagreement | random | -0.0470 to 0.0049 | 2 / 3 | -0.0003 to 0.0259 | -0.0552 to 0.0062 |
| cv_targets__separate | risk | -0.0507 to 0.0426 | 6 / 5 | -0.0007 to 0.1429 | -0.0918 to 0.0505 |
| cv_targets__separate | random | -0.0475 to 0.0043 | 4 / 4 | -0.0006 to 0.1935 | -0.0761 to 0.0046 |
| cv_targets__joint | risk | -0.0513 to 0.0524 | 8 / 6 | -0.0100 to 0.1876 | -0.0912 to 0.0635 |
| cv_targets__joint | random | -0.0481 to 0.0041 | 5 / 5 | -0.0012 to 0.2205 | -0.0769 to 0.0043 |
| floor_both__history | risk | -0.0417 to 0.0286 | 6 / 5 | -0.0007 to 0.0935 | -0.0742 to 0.0192 |
| floor_both__history | random | -0.0374 to 0.0027 | 4 / 4 | -0.0006 to 0.1511 | -0.0669 to 0.0004 |
| floor_both__disagreement | risk | -0.0415 to 0.0063 | 2 / 9 | -0.0091 to 0.0235 | -0.0643 to 0.0062 |
| floor_both__disagreement | random | -0.0403 to 0.0015 | 2 / 4 | -0.0091 to 0.0218 | -0.0565 to 0.0001 |
| floor_both__separate | risk | -0.0454 to 0.0284 | 6 / 4 | -0.0007 to 0.1017 | -0.0770 to 0.0191 |
| floor_both__separate | random | -0.0437 to 0.0027 | 4 / 4 | -0.0006 to 0.1604 | -0.0773 to 0.0004 |
| floor_both__joint | risk | -0.0457 to 0.0397 | 6 / 4 | -0.0008 to 0.1143 | -0.0778 to 0.0318 |
| floor_both__joint | random | -0.0473 to 0.0032 | 4 / 4 | -0.0032 to 0.1750 | -0.0793 to 0.0004 |

## Rejected Benefit Attribution

Mutually exclusive categories 1-4 exactly sum to joint-box removals. Each locality uses the same floor-error denominator.
Avoided harm minus lost benefit is a percentage-point change in gain vs floor, not relative gain vs stop.

| Parent | Reason | Known rows / view | Avoided harm (pp) | Lost benefit (pp) | Net removal change (pp) |
|---|---|---:|---:|---:|---:|
| cv_targets | history_only_failure | 44.0000 to 3731.0000 | 0.0008 to 0.0176 | 0.0091 to 0.0859 | -0.0751 to -0.0066 |
| cv_targets | disagreement_only_failure | 1.0000 to 702.0000 | 0.0000 to 0.0076 | 0.0000 to 0.0241 | -0.0239 to -0.0000 |
| cv_targets | both_marginals_fail | 12.0000 to 408.0000 | 0.0000 to 0.0108 | 0.0023 to 0.0611 | -0.0550 to 0.0077 |
| cv_targets | source_overlap_failure | 3.0000 to 561.0000 | 0.0000 to 0.0017 | 0.0000 to 0.0191 | -0.0180 to -0.0000 |
| floor_both | history_only_failure | 39.0000 to 2922.0000 | 0.0007 to 0.0124 | 0.0091 to 0.0842 | -0.0753 to -0.0054 |
| floor_both | disagreement_only_failure | 0.0000 to 556.0000 | 0.0000 to 0.0041 | 0.0000 to 0.0261 | -0.0259 to 0.0000 |
| floor_both | both_marginals_fail | 12.0000 to 383.0000 | 0.0000 to 0.0616 | 0.0016 to 0.0501 | -0.0472 to 0.0181 |
| floor_both | source_overlap_failure | 1.0000 to 468.0000 | 0 to 0.000864687 | 0.0000 to 0.0195 | -0.0192 to 0.0000 |

Every mode, fold, seed, event, source locality and error tail is retained in adjacent JSON/CSV.
Unknown future labels are not zero error. Complete-window and endpoint metrics retain only their actual label support.
The indexed-agent query cohort is not every visible agent. Forecast-disagreement projections do not prove producer transport causality.
No new deployment or independent confirmation. Image-pixel obs8/pred12 rawstride12, not historical raw-t50 or seconds/metric evidence.
