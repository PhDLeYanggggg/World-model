# Where the Frozen Policy Underestimates Selected Risk

## Material Passport

Fresh frozen-model inference and accounting:144/144 views in86.30s.
All prediction hashes and original selections reproduce. No new training,
threshold selection or deployment. Original TRAIN packets only; held
recordings are already exposed. This is not independent confirmation.

## Easy-Constraint Error Accounting

Sums below are source-scale-normalized and repeat rows across contexts/seeds.
They are not ADE/FDE, independent samples, calibrated probabilities, or causal
effects. Each row obeys observed=predicted+harm error+reference error.

### Fit Own-Selected Cohort

| Arm | Selected | Unknown | Predicted excess | Harm underestimated | Reference overestimated x0.02 | Observed excess |
|---|---:|---:|---:|---:|---:|---:|
| local | 123490 | 1436 | -608.270923 | -21.240677 | 557.111627 | -72.399973 |
| anchor | 337984 | 5363 | -135.300061 | 1174.935582 | -119.908820 | 919.726701 |
| old_risk | 67774 | 908 | -265.311549 | -16.108042 | 239.751677 | -41.667914 |
| utility_continue | 401255 | 6723 | -149.933615 | 1478.020193 | -140.364346 | 1187.722231 |
| joint_risk | 76621 | 1068 | -188.098932 | 22.136987 | 123.451602 | -42.510342 |
| isolated_risk | 45395 | 592 | -30.084950 | 55.983983 | -22.444722 | 3.454311 |

| Arm | Violating known rows | Still allowed with perfect reference | Still allowed with perfect harm | Violating views |
|---|---:|---:|---:|---:|
| local | 7639 | 6777 | 3597 | 1 |
| anchor | 23320 | 23284 | 471 | 62 |
| old_risk | 4141 | 3561 | 1689 | 9 |
| utility_continue | 26216 | 26197 | 540 | 65 |
| joint_risk | 4449 | 4192 | 1189 | 6 |
| isolated_risk | 3334 | 3298 | 251 | 21 |

Perfect-component replacements use future outcomes, are diagnostic only,
and are never legal inference inputs or deployable policies.

| Arm | Known selected in fit-radius bins <=p50 / p50-p90 / p90-p99 / >=p99 | Known easy-violating rows in same bins |
|---|---|---|
| local | [74346, 43163, 4175, 370] | [4485, 2940, 210, 4] |
| anchor | [164469, 138902, 26881, 2369] | [12260, 9516, 1437, 107] |
| old_risk | [40714, 22360, 3082, 710] | [2526, 1493, 116, 6] |
| utility_continue | [197636, 161939, 32056, 2901] | [13935, 10555, 1615, 111] |
| joint_risk | [46410, 24890, 3454, 799] | [2749, 1570, 123, 7] |
| isolated_risk | [23748, 15861, 4335, 859] | [1843, 1256, 217, 18] |

Bin assignment uses right-sided boundaries: values equal to a boundary go
in the higher bin. Labels in the header are shorthand, not exact quantile
occupancy. Ties/empty bins retained. Radius is not a density/support proof.

| Isolated branch cohort | Observed easy harm | Harm on rows exceeding model supremum | Sum minimum shortfall | Such rows |
|---|---:|---:|---:|---:|
| eligible | 5060.485293 | 3105.751404 | 2298.462190 | 25590.000000 |
| anchor | 1183.325429 | 1004.479998 | 802.971652 | 15592.000000 |
| isolated | 58.614455 | 45.380316 | 42.998012 | 2096.000000 |

A realized label above a conditional-mean ceiling does not prove the true
conditional expectation is above it. This is a feasibility stress diagnosis,
not a reason to introduce unbounded corrections or claim safety.

### Held Own-Selected Cohort

| Arm | Selected | Unknown | Predicted excess | Harm underestimated | Reference overestimated x0.02 | Observed excess |
|---|---:|---:|---:|---:|---:|---:|
| local | 57903 | 934 | -217.336139 | 57.324158 | 191.089761 | 31.077781 |
| anchor | 154772 | 2711 | -56.473702 | 630.856886 | -42.884506 | 531.498678 |
| old_risk | 27826 | 510 | -95.741707 | 22.178739 | 84.602254 | 11.039287 |
| utility_continue | 182163 | 3307 | -59.315721 | 792.583316 | -52.841588 | 680.426007 |
| joint_risk | 36085 | 599 | -72.821331 | 46.054314 | 47.811087 | 21.044069 |
| isolated_risk | 24204 | 421 | -12.735580 | 38.375982 | -7.564989 | 18.075413 |

| Arm | Violating known rows | Still allowed with perfect reference | Still allowed with perfect harm | Violating views |
|---|---:|---:|---:|---:|
| local | 3921 | 3618 | 1506 | 46 |
| anchor | 10677 | 10643 | 250 | 61 |
| old_risk | 1835 | 1616 | 736 | 40 |
| utility_continue | 11951 | 11933 | 335 | 61 |
| joint_risk | 2401 | 2282 | 584 | 45 |
| isolated_risk | 1589 | 1573 | 133 | 47 |

Perfect-component replacements use future outcomes, are diagnostic only,
and are never legal inference inputs or deployable policies.

| Arm | Known selected in fit-radius bins <=p50 / p50-p90 / p90-p99 / >=p99 | Known easy-violating rows in same bins |
|---|---|---|
| local | [25412, 18961, 8046, 4550] | [1895, 1486, 371, 169] |
| anchor | [61985, 63910, 21056, 5110] | [4795, 4593, 1097, 192] |
| old_risk | [14523, 8558, 2974, 1261] | [1027, 684, 99, 25] |
| utility_continue | [75437, 73450, 24186, 5783] | [5661, 4954, 1141, 195] |
| joint_risk | [17401, 10662, 4982, 2441] | [1227, 873, 205, 96] |
| isolated_risk | [8352, 8193, 5178, 2060] | [748, 595, 192, 54] |

Bin assignment uses right-sided boundaries: values equal to a boundary go
in the higher bin. Labels in the header are shorthand, not exact quantile
occupancy. Ties/empty bins retained. Radius is not a density/support proof.

| Isolated branch cohort | Observed easy harm | Harm on rows exceeding model supremum | Sum minimum shortfall | Such rows |
|---|---:|---:|---:|---:|
| eligible | 2600.888246 | 1722.739885 | 1300.972340 | 12540.000000 |
| anchor | 634.180691 | 560.392369 | 463.826530 | 7535.000000 |
| isolated | 39.489891 | 32.050564 | 30.113113 | 1185.000000 |

A realized label above a conditional-mean ceiling does not prove the true
conditional expectation is above it. This is a feasibility stress diagnosis,
not a reason to introduce unbounded corrections or claim safety.

## Boundaries and Reproduction

All six controls, all sources and all seeds are retained. Exact raw sums,
fixed anchor cohorts, unknown counts and fit-only edges remain in group JSON.
The prior completion-bound safety gates remain failed. This diagnosis has no
new success gate and does not upgrade the model or recertify Stage37.

```bash
.venv-pytorch/bin/python -m scripts.audit_m3w_selected_tail check
```

Complete outputs are immutable. Check verifies artifact hashes, identities,
arithmetic and counts, not a new model fit. Full execution used native arm64
CPU4/workers0, CREATE read-only TRAIN streaming, no new scheduler job.
Obs8/pred12 raw stride12, dataset-local/pixel coordinates, detector tracks.
No metric, seconds, physical safety, human gold, true3D or foundation claim.
Stage5C and SMC disabled. No reserved role opened.
