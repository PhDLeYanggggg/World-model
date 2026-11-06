# Prefix-Balanced Trajectory Producer: Actual Matched Fitting

## Result

Independent computational replay is complete: PID13422 exits0 in524.029483s.
All six prediction banks covering125540total rows,all analysis scalars and the public-file
snapshot match exactly,with no optimizer updates. This is repeatability,not
independent scientific confirmation. The two scalar-only plots are local and
were visually checked. See failure_analysis.md for the final negative decision.

Component progression screen: **FAIL**.
Equal-source paired available12ADE gain difference: **0.0976pp**.
Conservative paired completion-lower: **-9.4738pp**,
exploratory source bootstrap95% interval **[-17.8089, -3.0825]pp**.
6/14sources improve on the whole available12-step outcome.
Producer-fold known mean differences: {'0': 1.3002686780716803, '1': -0.804423605317734}.

This is a fitting-donor component contrast, not an independent test or
a deployment result. The source bootstrap is exploratory because training
roles overlap across folds and sources have prior exposure. One producer
seed does not meet the main three-seed paper requirement. Prefix-only
improvement cannot replace a failed whole12 contrast.

## What Was Actually Trained

Fresh_run:six3,027,268-parameter EqMotion K1 trajectory producers,4000updates
each,plus one full4000update old-loss control refit. The100+100vs200pilot
exactly reproduces model,optimizer,sampler,Torch RNG and trace. One pilot
prefix is resumed into its corresponding full treatment,not counted as
another completed model. All six controls/125540rows are cached_verified.
No new own-B,C/H/reserved data or new controller training. No rollout
ramp,ordinary residual head,threshold sweep or changed safety certificate.
Control scientific state and common trace exact: True;
old pilot-only extra log steps: [20, 40].
Recovery exact: True.

The sole intervention is a fixed loss:half available12ADE,half mean
supported1/3/6prefix ADEs. Rows with only late labels keep full loss;
wholly unknown rows stay in the sampler with zero gradient,not zero-error
labels. Inputs remain original past geometry476. Deep supervision is a
conventional backbone repair,not a novel paper-method claim.

## Every Source And Prefix

| Source | Prefix | Rows | Known | Partial | Unknown | Control gain % | New gain % | Paired pp | Paired lower pp | Control easy +harm % | New easy +harm % | New signed easy loss % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| moving_source_component_048 | 1 | 2038 | 1990 | 0 | 48 | -14.4947 | -2.6030 | 11.8917 | 9.5325 | 45.0842 | 28.1314 | 8.7383 |
| moving_source_component_048 | 3 | 2038 | 1991 | 98 | 47 | 14.2643 | 18.6164 | 4.3521 | 2.4463 | 28.6286 | 20.6215 | 1.1756 |
| moving_source_component_048 | 6 | 2038 | 1992 | 265 | 46 | 26.2247 | 26.6310 | 0.4062 | -1.5778 | 31.2518 | 25.4342 | 7.1454 |
| moving_source_component_048 | 12 | 2038 | 1997 | 618 | 41 | 32.8183 | 32.8418 | 0.0235 | -2.7625 | 47.2830 | 37.9125 | 22.2761 |
| moving_source_component_067 | 1 | 4159 | 3987 | 0 | 172 | -21.4334 | 4.7396 | 26.1729 | 20.0748 | 52.9427 | 27.5352 | 6.4182 |
| moving_source_component_067 | 3 | 4159 | 3997 | 328 | 162 | 19.9281 | 23.1669 | 3.2388 | -0.7280 | 34.2355 | 20.2825 | -1.1922 |
| moving_source_component_067 | 6 | 4159 | 4004 | 757 | 155 | 30.6916 | 28.6459 | -2.0457 | -6.2618 | 34.1733 | 24.5256 | 3.1181 |
| moving_source_component_067 | 12 | 4159 | 4006 | 1538 | 153 | 39.2407 | 35.8589 | -3.3818 | -10.4494 | 36.5508 | 30.5544 | 11.6332 |
| moving_source_component_047 | 1 | 2198 | 2155 | 0 | 43 | -36.4500 | -13.5067 | 22.9433 | 20.9699 | 43.5543 | 27.2079 | 5.3887 |
| moving_source_component_047 | 3 | 2198 | 2162 | 93 | 36 | -15.6151 | -10.7110 | 4.9042 | 3.3365 | 28.8056 | 22.9991 | 0.4520 |
| moving_source_component_047 | 6 | 2198 | 2163 | 213 | 35 | -10.4021 | -8.7001 | 1.7021 | 0.1796 | 34.2951 | 31.4118 | 6.5109 |
| moving_source_component_047 | 12 | 2198 | 2167 | 450 | 31 | -11.6184 | -10.2494 | 1.3690 | -0.6388 | 55.3906 | 53.7145 | 33.4586 |
| moving_source_component_052 | 1 | 536 | 515 | 0 | 21 | 20.9007 | 25.7523 | 4.8516 | 1.4949 | 64.4539 | 66.0237 | 40.7824 |
| moving_source_component_052 | 3 | 536 | 516 | 28 | 20 | 39.4923 | 39.5009 | 0.0086 | -4.8304 | 43.6679 | 50.1346 | 19.4542 |
| moving_source_component_052 | 6 | 536 | 518 | 41 | 18 | 51.7048 | 49.9821 | -1.7227 | -7.2999 | 44.0569 | 50.5144 | 23.1887 |
| moving_source_component_052 | 12 | 536 | 518 | 73 | 18 | 61.3592 | 60.8670 | -0.4922 | -20.3617 | 35.7089 | 33.4873 | 11.1362 |
| moving_source_component_051 | 1 | 7196 | 7112 | 0 | 84 | -55.4358 | -18.4420 | 36.9938 | 33.0331 | 81.2792 | 45.3910 | 26.8036 |
| moving_source_component_051 | 3 | 7196 | 7116 | 168 | 80 | -15.9299 | -6.1557 | 9.7742 | 6.1250 | 51.9701 | 38.9386 | 21.8697 |
| moving_source_component_051 | 6 | 7196 | 7121 | 416 | 75 | -3.9272 | 4.3483 | 8.2755 | 2.8249 | 53.7034 | 47.3555 | 30.6887 |
| moving_source_component_051 | 12 | 7196 | 7128 | 874 | 68 | 5.4075 | 16.6032 | 11.1957 | 4.9676 | 58.7505 | 50.3837 | 34.0524 |
| moving_source_component_062 | 1 | 37829 | 37419 | 0 | 410 | -20.9459 | -9.1129 | 11.8330 | 10.6883 | 77.0215 | 55.9966 | 41.9450 |
| moving_source_component_062 | 3 | 37829 | 37437 | 817 | 392 | -8.0217 | -2.7759 | 5.2458 | 4.0470 | 73.8441 | 61.3091 | 51.9137 |
| moving_source_component_062 | 6 | 37829 | 37458 | 1995 | 371 | -5.9837 | -1.8842 | 4.0995 | 2.6951 | 105.7961 | 95.3469 | 87.8774 |
| moving_source_component_062 | 12 | 37829 | 37472 | 4129 | 357 | -20.1659 | -21.0786 | -0.9127 | -2.7466 | 186.9278 | 185.3794 | 180.8856 |
| moving_source_component_053 | 1 | 1137 | 1085 | 0 | 52 | 50.5429 | 50.2843 | -0.2586 | -4.6617 | 49.9998 | 39.5095 | 10.6924 |
| moving_source_component_053 | 3 | 1137 | 1089 | 113 | 48 | 42.4706 | 45.8192 | 3.3485 | -0.7864 | 75.2407 | 63.7378 | 47.4678 |
| moving_source_component_053 | 6 | 1137 | 1091 | 286 | 46 | 35.7092 | 38.5805 | 2.8713 | -0.6908 | 107.9862 | 81.8525 | 70.6616 |
| moving_source_component_053 | 12 | 1137 | 1095 | 570 | 42 | 30.0694 | 34.3653 | 4.2959 | -1.2956 | 166.9285 | 124.3933 | 116.8986 |
| moving_source_component_065 | 1 | 1042 | 940 | 0 | 102 | -17.0867 | -6.3744 | 10.7123 | 4.1390 | 55.7653 | 36.9177 | 20.7880 |
| moving_source_component_065 | 3 | 1042 | 940 | 204 | 102 | 6.3881 | 8.8178 | 2.4296 | -6.0610 | 35.9734 | 27.2723 | 8.9624 |
| moving_source_component_065 | 6 | 1042 | 941 | 499 | 101 | 13.6045 | 13.2119 | -0.3926 | -9.8399 | 36.6697 | 31.3474 | 11.9241 |
| moving_source_component_065 | 12 | 1042 | 941 | 853 | 101 | 14.6994 | 13.5010 | -1.1984 | -18.6808 | 43.5091 | 37.7322 | 19.3406 |
| moving_source_component_076 | 1 | 3342 | 3175 | 0 | 167 | -53.9510 | -22.0855 | 31.8656 | 25.9464 | 90.4179 | 54.1966 | 40.7193 |
| moving_source_component_076 | 3 | 3342 | 3181 | 333 | 161 | -25.8177 | -16.5370 | 9.2807 | 3.6118 | 74.9888 | 59.5240 | 48.4737 |
| moving_source_component_076 | 6 | 3342 | 3182 | 796 | 160 | -16.9082 | -13.5039 | 3.4043 | -1.7479 | 87.8353 | 74.7245 | 64.6113 |
| moving_source_component_076 | 12 | 3342 | 3184 | 1604 | 158 | -12.4842 | -9.8207 | 2.6635 | -3.4195 | 116.3070 | 102.8652 | 94.1308 |
| moving_source_component_080 | 1 | 10670 | 10497 | 0 | 173 | 3.4498 | 7.1457 | 3.6959 | 2.4027 | 35.0692 | 30.5513 | 13.6025 |
| moving_source_component_080 | 3 | 10670 | 10513 | 362 | 157 | 14.0782 | 15.8478 | 1.7696 | 0.5212 | 31.1448 | 28.9880 | 12.2312 |
| moving_source_component_080 | 6 | 10670 | 10519 | 876 | 151 | 15.3293 | 16.9726 | 1.6433 | 0.1067 | 41.0485 | 38.9044 | 24.4553 |
| moving_source_component_080 | 12 | 10670 | 10523 | 1806 | 147 | 14.4769 | 15.9746 | 1.4977 | -0.4145 | 53.8508 | 49.2852 | 37.2730 |
| moving_source_component_082 | 1 | 6225 | 5920 | 0 | 305 | -60.9648 | -42.2238 | 18.7409 | 15.5686 | 116.9571 | 86.0958 | 72.1229 |
| moving_source_component_082 | 3 | 6225 | 5925 | 624 | 300 | -29.9026 | -24.8159 | 5.0867 | 2.2018 | 97.9221 | 82.1218 | 70.6536 |
| moving_source_component_082 | 6 | 6225 | 5930 | 1595 | 295 | -17.2718 | -18.7436 | -1.4717 | -4.8486 | 99.5594 | 102.7540 | 93.8289 |
| moving_source_component_082 | 12 | 6225 | 5930 | 3572 | 295 | -15.4158 | -19.7262 | -4.3104 | -8.6051 | 129.5593 | 158.4419 | 151.7452 |
| moving_source_component_087 | 1 | 2847 | 2640 | 0 | 207 | 8.5863 | -4.3248 | -12.9111 | -49.7162 | 215.3688 | 201.6381 | 174.4230 |
| moving_source_component_087 | 3 | 2847 | 2643 | 366 | 204 | 14.8632 | 6.3646 | -8.4986 | -37.1991 | 132.7148 | 129.7931 | 113.1454 |
| moving_source_component_087 | 6 | 2847 | 2645 | 903 | 202 | 16.9514 | 10.9191 | -6.0323 | -28.4007 | 131.6555 | 129.4516 | 114.4008 |
| moving_source_component_087 | 12 | 2847 | 2653 | 1778 | 194 | 17.6411 | 13.0502 | -4.5909 | -56.4978 | 132.5464 | 130.1875 | 115.1527 |
| moving_source_component_077 | 1 | 6127 | 5923 | 0 | 204 | -3.2669 | 2.6195 | 5.8864 | 4.2419 | 32.1249 | 22.8700 | 8.9777 |
| moving_source_component_077 | 3 | 6127 | 5933 | 415 | 194 | 10.0199 | 11.0978 | 1.0779 | -0.6295 | 31.2343 | 24.6010 | 9.7017 |
| moving_source_component_077 | 6 | 6127 | 5936 | 995 | 191 | 18.6072 | 17.8985 | -0.7087 | -3.0081 | 37.3076 | 34.6351 | 20.1994 |
| moving_source_component_077 | 12 | 6127 | 5943 | 2111 | 184 | 23.5650 | 21.4134 | -2.1516 | -4.8707 | 55.8779 | 58.7125 | 46.8037 |
| moving_source_component_078 | 1 | 40194 | 38980 | 0 | 1214 | 8.9937 | 10.7231 | 1.7294 | -0.4691 | 22.3805 | 15.5702 | -1.8699 |
| moving_source_component_078 | 3 | 40194 | 39047 | 2406 | 1147 | 27.3451 | 26.1440 | -1.2011 | -3.8825 | 16.9803 | 13.7981 | -9.5804 |
| moving_source_component_078 | 6 | 40194 | 39091 | 5755 | 1103 | 36.1940 | 35.6504 | -0.5436 | -3.4734 | 18.5402 | 16.8770 | -8.1807 |
| moving_source_component_078 | 12 | 40194 | 39159 | 11775 | 1035 | 35.6952 | 33.0540 | -2.6412 | -6.8571 | 14.6648 | 12.4438 | -15.6151 |

## Every Recording: Whole Available12

| Recording | Rows | Unknown | Old gain % | New gain % | Paired pp | New positive harm % | New easy +harm % | New signed easy loss % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2038 | 41 | 32.8183 | 32.8418 | 0.0235 | 5.2554 | 37.9125 | 22.2761 |
| 34 | 4159 | 153 | 39.2407 | 35.8589 | -3.3818 | 2.2314 | 30.5544 | 11.6332 |
| 0 | 1599 | 24 | -13.8188 | -11.9485 | 1.8703 | 24.1927 | 72.9990 | 59.0434 |
| 1 | 599 | 7 | 15.9310 | 11.0233 | -4.9078 | 20.0081 | 30.3863 | 2.5089 |
| 8 | 536 | 18 | 61.3592 | 60.8670 | -0.4922 | 0.6379 | 33.4873 | 11.1362 |
| 5 | 3030 | 29 | 7.4903 | 20.0491 | 12.5588 | 9.1291 | 33.9439 | 13.2408 |
| 6 | 2730 | 29 | 16.3650 | 24.9174 | 8.5524 | 15.4509 | 71.5453 | 60.8170 |
| 7 | 1436 | 10 | -20.2239 | -6.5680 | 13.6559 | 20.5283 | 44.5526 | 26.7098 |
| 26 | 3861 | 47 | 3.7260 | 8.0865 | 4.3605 | 19.4097 | 149.3121 | 146.2274 |
| 27 | 3712 | 34 | -43.3277 | -56.4099 | -13.0823 | 68.9872 | 339.4388 | 337.5314 |
| 28 | 14262 | 129 | -1.9874 | 2.1471 | 4.1345 | 23.6464 | 102.6092 | 96.7270 |
| 29 | 6307 | 37 | -17.1319 | -12.3540 | 4.7779 | 33.1514 | 115.4600 | 107.9134 |
| 30 | 5544 | 49 | -45.9007 | -54.4661 | -8.5654 | 67.6674 | 332.0667 | 330.5036 |
| 31 | 4143 | 61 | -17.7509 | -12.3265 | 5.4244 | 32.4417 | 219.7911 | 217.3932 |
| 10 | 39 | 0 | 49.3897 | 42.8088 | -6.5808 | 0.2142 | 25.2795 | 17.4247 |
| 11 | 108 | 5 | -0.5046 | -3.7677 | -3.2630 | 17.0465 | 67.2880 | 64.0545 |
| 12 | 898 | 37 | 38.2676 | 41.5129 | 3.2453 | 2.3487 | 70.2296 | 48.5090 |
| 13 | 68 | 0 | -20.3374 | -11.7317 | 8.6058 | 28.3740 | 215.8715 | 215.8715 |
| 14 | 24 | 0 | -16.6535 | 21.2792 | 37.9327 | 13.9055 | 256.4928 | 256.4928 |
| 32 | 1042 | 101 | 14.6994 | 13.5010 | -1.1984 | 13.7089 | 37.7322 | 19.3406 |
| 44 | 2412 | 125 | -7.5573 | -6.2797 | 1.2777 | 24.5516 | 76.6120 | 65.5982 |
| 45 | 930 | 33 | -19.1810 | -14.6339 | 4.5471 | 30.6746 | 146.3579 | 141.3998 |
| 51 | 10670 | 147 | 14.4769 | 15.9746 | 1.4977 | 8.9701 | 49.2852 | 37.2730 |
| 52 | 6225 | 295 | -15.4158 | -19.7262 | -4.3104 | 28.3086 | 158.4419 | 151.7452 |
| 53 | 1062 | 106 | -0.0531 | -7.9775 | -7.9244 | 10.1143 | 421.1060 | 421.0756 |
| 54 | 607 | 17 | 11.1610 | 25.1394 | 13.9784 | 13.2059 | 66.2492 | 50.5190 |
| 55 | 640 | 40 | 43.8590 | 39.2846 | -4.5744 | 1.4616 | 10.5905 | -24.7911 |
| 57 | 414 | 29 | 22.5413 | 17.0667 | -5.4747 | 4.5968 | 202.6327 | 199.5738 |
| 58 | 124 | 2 | 27.0837 | 25.2740 | -1.8097 | 1.5005 | 134.1744 | 134.1744 |
| 46 | 3193 | 83 | 31.8008 | 31.0872 | -0.7136 | 4.9637 | 44.9302 | 32.3804 |
| 47 | 2934 | 101 | 13.0279 | 9.0365 | -3.9914 | 14.8091 | 64.3396 | 52.6925 |
| 48 | 13417 | 274 | 36.2620 | 33.7857 | -2.4762 | 1.4583 | 8.9007 | -21.7916 |
| 49 | 18416 | 483 | 29.9087 | 28.1340 | -1.7747 | 3.1045 | 17.2771 | -6.8203 |
| 50 | 8361 | 278 | 42.4764 | 38.5982 | -3.8782 | 0.5226 | 8.4233 | -23.6709 |

## Available Endpoint FDE: Secondary Diagnostic

| Source | Endpoint known | Endpoint unknown | Old FDE gain % | New FDE gain % | Paired FDE pp |
|---|---:|---:|---:|---:|---:|
| moving_source_component_048 | 1412 | 626 | 32.1325 | 33.7347 | 1.6022 |
| moving_source_component_067 | 2559 | 1600 | 44.9495 | 43.3288 | -1.6206 |
| moving_source_component_047 | 1803 | 395 | -16.9854 | -11.4615 | 5.5239 |
| moving_source_component_052 | 450 | 86 | 65.4462 | 64.3419 | -1.1043 |
| moving_source_component_051 | 6338 | 858 | 11.5307 | 25.3141 | 13.7834 |
| moving_source_component_062 | 33659 | 4170 | -37.7569 | -44.3364 | -6.5795 |
| moving_source_component_053 | 578 | 559 | 22.3366 | 30.9692 | 8.6326 |
| moving_source_component_065 | 90 | 952 | 3.5025 | 8.8477 | 5.3452 |
| moving_source_component_076 | 1604 | 1738 | -8.4088 | -4.7441 | 3.6647 |
| moving_source_component_080 | 8869 | 1801 | 12.6580 | 14.6084 | 1.9504 |
| moving_source_component_082 | 2409 | 3816 | -15.5229 | -22.0312 | -6.5084 |
| moving_source_component_087 | 891 | 1956 | 19.8598 | 17.6300 | -2.2298 |
| moving_source_component_077 | 3908 | 2219 | 25.7183 | 22.7057 | -3.0126 |
| moving_source_component_078 | 28246 | 11948 | 31.3132 | 27.2178 | -4.0954 |

Endpoint FDE uses a different supported subset from available-label ADE.
It is not substituted for the failed registered primary contrast.

Known-label raw-actor harm columns are diagnostics,not the full policy
certificate. Original recording2% all/easy positive harm,signed easy-net
and support requirements remain unchanged. NA is not zero or pass. No
actor is directly deployed; new risk heads would need new causal labels
and the complete original policy checks before any later deployment.

## Compute And Reproduction

Six treatment cumulative fitting seconds:7854.136111.
Full control fitting seconds:1298.300801.
loss_trace.csv retains the original logging field names. The legacy
mean_past_normalized_ADE column is an unweighted minibatch available ADE
in the packed image coordinates,not a metric or scene-normalized score.
The two objective loss columns use different weights and cannot be
compared directly as evidence of improved forecast accuracy.
Native arm64 CPU4/inter-op1/workers0. Atomic200update checkpoints and
50update heartbeats; checkpoints,arrays and private caches stay local.

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py control --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py forecast
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py replay
```

Completed pilot is immutable;verify its receipt rather than re-running.
Replay regenerates predictions and scalars without optimizer updates.
Current task is native observation8/prediction12 annotation stride1 image
coordinates. No meters,seconds,true3D,foundation,human-gold,physical-safety
or submission-ready claim. Stage5C execution and SMC remain off.
