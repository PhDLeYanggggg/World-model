# Separate Development and TRAIN Evidence

| SDD comparison | ADE gain difference (pp) | Nominal 95% CI |
|---|---:|---:|
| Forest minus log neural (original primary) | +0.132728 | [-0.062063, +0.481336] |
| Forest minus log neural (same-count risk) | +0.723314 | [+0.610586, +0.827277] |
| Square minus log neural (strict primary) | +0.049515 | [-0.005267, +0.138640] |
| Square minus log neural (same-count risk) | -0.207277 | [-0.253252, -0.140711] |

| ID | EuropeanSquares comparison | Change | Nominal 95% CI | Localities |
|---|---|---:|---:|---:|
| E1 | Cost minus original, TRAIN | -0.124240 | [-0.150777, -0.099481] | 12 |
| E2 | Cost minus original, validation | +0.104153 | [+0.018611, +0.220039] | 12 |
| E3 | Extension minus cost, validation | -0.126005 | [-0.246783, -0.030947] | 12 |
| E4 | Extension minus original, validation | -0.021852 | [-0.057900, -0.000175] | 12 |
| E5 | Extension minus additive, validation | +0.010294 | [-0.001532, +0.026280] | 12 |
| E6 | Temporal minus row-mean leaf, validation | -0.094342 | [-0.158824, -0.038614] | 12 |
| E7 | Temporal minus global temporal, validation | -0.137388 | [-0.232556, -0.047422] | 12 |
| E8 | Temporal minus row-mean leaf, complete labels | -0.098473 | [-0.159674, -0.047870] | 12 |
| E9 | Temporal minus row-mean leaf, original selected | +0.000043 | [-0.000140, +0.000256] | 11 |
| E10 | Extension minus cost, full utility | +0.000015 | [-0.000003, +0.000041] | 12 |
| E11 | Extension minus cost, matched utility | +0.000010 | [+0.000000, +0.000024] | 12 |

| European policy | Selected occurrences | Unknown selected | Complete support / 72 | Defined easy risk / 72 | Known violations | Upper-bound violations | Worst easy upper % |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 95,455 | 918 | 33 | 43 | 4 | 7 | 5.4058 |
| additive | 112,456 | 1,143 | 19 | 61 | 20 | 42 | 1200.1684 |
| poisson | 111,031 | 1,050 | 37 | 51 | 2 | 11 | 18.0277 |
| cost | 96,720 | 926 | 36 | 48 | 4 | 7 | 5.7195 |
| extended | 96,718 | 926 | 36 | 47 | 4 | 7 | 5.7195 |

| Comparator | Signed-score MSE | Full paired lower utility | Same-count paired lower utility |
|---|---:|---:|---:|
| original | +0.261571 [+0.104999, +0.451020] | +1.390725 [+0.566962, +2.193647] | +0.261338 [+0.034097, +0.566836] |
| additive | +0.293717 [+0.126013, +0.504089] | +1.053742 [+0.239029, +1.861736] | +0.220231 [-0.024922, +0.521718] |
| poisson | +0.125410 [-0.070909, +0.319224] | +1.345108 [+0.527166, +2.147304] | +0.270922 [+0.023377, +0.589251] |
| cost | +0.157417 [-0.056953, +0.367987] | +1.384831 [+0.561310, +2.187567] | +0.257274 [+0.017003, +0.570425] |
| none | -0.062288 [-0.151167, +0.005691] | -1.026746 [-1.557310, -0.574154] | -0.567678 [-0.939022, -0.286494] |
| rowmean | -0.024422 [-0.054556, -0.001405] | -0.353836 [-0.545292, -0.186078] | -0.231955 [-0.372209, -0.112760] |

| Auxiliary arm | TRAIN risk violations /72 | Median risk (%) | Median predicted/actual harm: all known | Median predicted/actual harm: selected known | Unknown selected |
|---|---:|---:|---:|---:|---:|
| none | 58 | 5.371021 | 0.718954 | 0.024865 | 2,496 |
| rowmean | 57 | 4.567470 | 0.676995 | 0.037706 | 1,911 |
| temporal | 52 | 4.304727 | 0.705898 | 0.039213 | 1,828 |

| Policy | Selected occurrences | Unknown selected | Undefined easy risk /72 | Upper-risk violations /72 | Complete finite support /72 |
|---|---:|---:|---:|---:|---:|
| original | 95,455 | 918 | 29 | 7 | 33 |
| additive | 112,456 | 1,143 | 11 | 42 | 19 |
| poisson | 111,031 | 1,050 | 21 | 11 | 37 |
| cost | 96,720 | 926 | 24 | 7 | 36 |
| none | 107,596 | 1,487 | 0 | 72 | 0 |
| rowmean | 84,251 | 1,201 | 0 | 72 | 0 |
| temporal | 83,168 | 1,191 | 0 | 72 | 0 |
