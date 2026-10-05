# Separate Development Studies

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
