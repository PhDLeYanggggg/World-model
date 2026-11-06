# Does Explicit Past-Image Viewpoint Conditioning Help?

I kept the same trajectory network and trained it in a query-view frame
derived from past images. Predicted camera motion restores native output
coordinates before loss. No new learned parameters or residual decoder.

Component screen:**FAIL**.
Whole12primary known difference:**-5.1331pp**.
Paired completion lower:**-21.4474pp**,exploratory95%
source-bootstrap interval**[-33.6823,-9.9109]**.
Known-positive sources:3/14;producer-fold means:{'0': -5.225888117730981, '1': -5.063449324958821}.

**No new deployment,independent confirmation or complete selected-policy
risk pass follows.** The original2%all/easy,signed easy-net,support/NA
rules remain. A failed lower bound does not identify the actual unknown
effect. The experiment uses exposed A donors,not a new test.

## Matched Prefix Comparisons

| Contrast | Prefix | Control gain vs CV % | New gain vs CV % | Paired known pp | Lower pp | Upper pp |
|---|---:|---:|---:|---:|---:|---:|
| primary | 1 | -13.6826 | -14.8079 | -1.1253 | -8.4442 | +6.1936 |
| primary | 3 | +6.6831 | +1.4205 | -5.2626 | -13.3551 | +2.8299 |
| primary | 6 | +13.6088 | +7.2581 | -6.3508 | -17.3335 | +4.6319 |
| primary | 12 | +15.3777 | +10.2447 | -5.1331 | -21.4474 | +11.1813 |
| camera_cv_effect | 1 | +0.0000 | +0.5444 | +0.5444 | +0.0571 | +1.0317 |
| camera_cv_effect | 3 | +0.0000 | +1.6963 | +1.6963 | +0.2506 | +3.1421 |
| camera_cv_effect | 6 | +0.0000 | +2.5838 | +2.5838 | -0.1746 | +5.3421 |
| camera_cv_effect | 12 | +0.0000 | +2.8275 | +2.8275 | -2.6114 | +8.2665 |
| neural_over_camera_cv | 1 | +0.5444 | -14.8079 | -15.3522 | -21.6825 | -9.0220 |
| neural_over_camera_cv | 3 | +1.6963 | +1.4205 | -0.2759 | -9.9853 | +9.4335 |
| neural_over_camera_cv | 6 | +2.5838 | +7.2581 | +4.6743 | -9.1909 | +18.5396 |
| neural_over_camera_cv | 12 | +2.8275 | +10.2447 | +7.4171 | -12.0501 | +26.8843 |

Primary is the camera-conditioned actor minus the cached identity actor.
Camera-CV minus raw CV and neural minus camera-CV are secondary controls.
A useful camera baseline alone is not proof of neural added value.
Weights balance queries within each source,then sources. Missing references
remain NA;no defined-only averaging or outcome-selected primary.

## Causal Baseline Diagnostics

| Baseline | Prefix | Known gain vs CV % | Hard gain % | Easy positive harm % |
|---|---:|---:|---:|---:|
| camera_compensated_cv | 1 | +0.5444 | +1.9763 | +3.5355 |
| camera_compensated_cv | 3 | +1.6963 | +3.6988 | +5.2534 |
| camera_compensated_cv | 6 | +2.5838 | +4.8235 | +8.3515 |
| camera_compensated_cv | 12 | +2.8275 | +5.2186 | +14.0694 |
| constant_position | 1 | -589.3265 | -427.7091 | +983.7434 |
| constant_position | 3 | -561.4350 | -366.9508 | +1106.0160 |
| constant_position | 6 | -489.6865 | -293.4622 | +1210.4604 |
| constant_position | 12 | -391.9643 | -220.0769 | +1278.6621 |
| constant_velocity_causal_fd | 1 | +0.0000 | +0.0000 | +0.0000 |
| constant_velocity_causal_fd | 3 | +0.0000 | +0.0000 | +0.0000 |
| constant_velocity_causal_fd | 6 | +0.0000 | +0.0000 | +0.0000 |
| constant_velocity_causal_fd | 12 | +0.0000 | +0.0000 | +0.0000 |
| damped_velocity_090 | 1 | +0.0000 | +0.0000 | +0.0000 |
| damped_velocity_090 | 3 | -19.3895 | -14.7610 | +35.5226 |
| damped_velocity_090 | 6 | -47.4918 | -30.5575 | +96.5345 |
| damped_velocity_090 | 12 | -74.1374 | -44.4520 | +208.8942 |
| damped_velocity_097 | 1 | +0.0000 | +0.0000 | +0.0000 |
| damped_velocity_097 | 3 | -4.3327 | -4.0099 | +8.4008 |
| damped_velocity_097 | 6 | -11.5338 | -8.5151 | +23.0497 |
| damped_velocity_097 | 12 | -20.4572 | -12.8235 | +59.1357 |
| history_ols4_velocity | 1 | -8.2249 | -4.6602 | +51.1689 |
| history_ols4_velocity | 3 | -6.6142 | -0.5747 | +51.6056 |
| history_ols4_velocity | 6 | -5.5589 | +1.4116 | +57.4500 |
| history_ols4_velocity | 12 | -3.9587 | +2.6200 | +67.6789 |
| history_ols8_velocity | 1 | -44.8118 | -41.6824 | +104.6426 |
| history_ols8_velocity | 3 | -41.2978 | -30.7077 | +108.6703 |
| history_ols8_velocity | 6 | -33.8881 | -20.1793 | +118.9347 |
| history_ols8_velocity | 12 | -24.4621 | -11.4942 | +130.6706 |

All seven fixed causal baselines are reported. An aggregate best score
on these exposed donors is diagnostic,not a new selected deployment floor.

## Every Source, Whole12

| Source | Rows | Active camera rows | Known | Partial | Unknown | Known pp | Lower pp | Upper pp | New easy +harm % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| moving_source_component_047 | 2198 | 1640 | 2167 | 450 | 31 | -11.1875 | -15.6584 | -6.7167 | +96.3212 |
| moving_source_component_048 | 2038 | 2038 | 1997 | 618 | 41 | -10.2196 | -20.7063 | +0.2671 | +22.0238 |
| moving_source_component_051 | 7196 | 6851 | 7128 | 874 | 68 | -1.8529 | -14.5835 | +10.8776 | +94.7397 |
| moving_source_component_052 | 536 | 0 | 518 | 73 | 18 | -24.9845 | -36.9777 | -12.9913 | +98.0409 |
| moving_source_component_053 | 1137 | 861 | 1095 | 570 | 42 | -30.0289 | -46.4967 | -13.5610 | +107.0606 |
| moving_source_component_062 | 37829 | 36678 | 37472 | 4129 | 357 | +19.2007 | +15.4639 | +22.9376 | +117.0057 |
| moving_source_component_065 | 1042 | 1042 | 941 | 853 | 101 | -4.4738 | -62.9093 | +53.9617 | +34.5183 |
| moving_source_component_067 | 4159 | 1196 | 4006 | 1538 | 153 | -2.3115 | -15.3076 | +10.6846 | +37.5454 |
| moving_source_component_076 | 3342 | 3342 | 3184 | 1604 | 158 | +9.9885 | -4.4931 | +24.4701 | +60.5262 |
| moving_source_component_077 | 6127 | 6127 | 5943 | 2111 | 184 | -6.7572 | -13.2593 | -0.2551 | +57.0234 |
| moving_source_component_078 | 40194 | 17984 | 39159 | 11775 | 1035 | -10.0063 | -17.7450 | -2.2677 | +46.2947 |
| moving_source_component_080 | 10670 | 8823 | 10523 | 1806 | 147 | -2.5341 | -6.1360 | +1.0679 | +57.1819 |
| moving_source_component_082 | 6225 | 6225 | 5930 | 3572 | 295 | +14.9963 | +2.5941 | +27.3986 | +74.7966 |
| moving_source_component_087 | 2847 | 1435 | 2653 | 1778 | 194 | -11.6922 | -64.0488 | +40.6644 | +137.9619 |

Raw camera actor known-easy +harm exceeds2%in33/34recordings;
0undefined easy references remain NA.
This is a raw actor diagnostic,not a complete selected-policy certificate.

## Camera Support Strata, Whole12

| Fitting design | Camera active | Rows | Contrast | Known pp | Lower pp |
|---|---|---:|---|---:|---:|
| eqmotion_A0_inner0_seed17 | False | 2963 | primary | +0.3971 | -10.2644 |
| eqmotion_A0_inner0_seed17 | True | 3234 | primary | -9.1583 | -22.3079 |
| eqmotion_A0_inner0_seed17 | False | 2963 | camera_cv_effect | +0.0000 | +0.0000 |
| eqmotion_A0_inner0_seed17 | True | 3234 | camera_cv_effect | +20.6970 | +5.8882 |
| eqmotion_A0_inner0_seed17 | False | 2963 | neural_over_camera_cv | +37.2232 | +4.5755 |
| eqmotion_A0_inner0_seed17 | True | 3234 | neural_over_camera_cv | +7.1853 | +1.2344 |
| eqmotion_A0_inner1_seed17 | False | 1094 | primary | -15.7109 | -23.2671 |
| eqmotion_A0_inner1_seed17 | True | 1640 | primary | -11.2006 | -15.0587 |
| eqmotion_A0_inner1_seed17 | False | 1094 | camera_cv_effect | +0.0000 | +0.0000 |
| eqmotion_A0_inner1_seed17 | True | 1640 | camera_cv_effect | -7.2073 | -9.4909 |
| eqmotion_A0_inner1_seed17 | False | 1094 | neural_over_camera_cv | -8.0638 | -41.9079 |
| eqmotion_A0_inner1_seed17 | True | 1640 | neural_over_camera_cv | -10.8783 | -14.1813 |
| eqmotion_A0_inner2_seed17 | False | 1496 | primary | +25.2806 | +20.2386 |
| eqmotion_A0_inner2_seed17 | True | 43529 | primary | +14.1546 | +8.4163 |
| eqmotion_A0_inner2_seed17 | False | 1496 | camera_cv_effect | +0.0000 | +0.0000 |
| eqmotion_A0_inner2_seed17 | True | 43529 | camera_cv_effect | -2.0890 | -6.4330 |
| eqmotion_A0_inner2_seed17 | False | 1496 | neural_over_camera_cv | +17.4953 | +12.2098 |
| eqmotion_A0_inner2_seed17 | True | 43529 | neural_over_camera_cv | +1.4267 | -3.0413 |
| eqmotion_A1_inner0_seed17 | False | 276 | primary | -23.2293 | -30.3816 |
| eqmotion_A1_inner0_seed17 | True | 5245 | primary | -11.5435 | -34.8590 |
| eqmotion_A1_inner0_seed17 | False | 276 | camera_cv_effect | +0.0000 | +0.0000 |
| eqmotion_A1_inner0_seed17 | True | 5245 | camera_cv_effect | +0.6403 | -11.9917 |
| eqmotion_A1_inner0_seed17 | False | 276 | neural_over_camera_cv | +9.9658 | -9.3071 |
| eqmotion_A1_inner0_seed17 | True | 5245 | neural_over_camera_cv | -1.2215 | -15.7784 |
| eqmotion_A1_inner1_seed17 | False | 3259 | primary | -8.0849 | -63.5490 |
| eqmotion_A1_inner1_seed17 | True | 16483 | primary | +7.0400 | -4.9980 |
| eqmotion_A1_inner1_seed17 | False | 3259 | camera_cv_effect | +0.0000 | +0.0000 |
| eqmotion_A1_inner1_seed17 | True | 16483 | camera_cv_effect | -0.4283 | -4.0420 |
| eqmotion_A1_inner1_seed17 | False | 3259 | neural_over_camera_cv | +6.4262 | -55.6503 |
| eqmotion_A1_inner1_seed17 | True | 16483 | neural_over_camera_cv | +3.2808 | -2.2747 |
| eqmotion_A1_inner2_seed17 | False | 22210 | primary | -2.2420 | -10.7563 |
| eqmotion_A1_inner2_seed17 | True | 24111 | primary | -15.4588 | -22.0657 |
| eqmotion_A1_inner2_seed17 | False | 22210 | camera_cv_effect | +0.0000 | +0.0000 |
| eqmotion_A1_inner2_seed17 | True | 24111 | camera_cv_effect | +12.6341 | +2.6674 |
| eqmotion_A1_inner2_seed17 | False | 22210 | neural_over_camera_cv | +36.0286 | +16.3344 |
| eqmotion_A1_inner2_seed17 | True | 24111 | neural_over_camera_cv | +0.8274 | -4.6089 |

Active and unsupported populations differ;their separate means do not
identify a causal effect of camera-support status. No slice is removed.

## What Actually Ran

Fresh_run:six complete4000update treatment fits,one full identity-control
refit and one full first-design treatment refit. Not six full refits.
Real100+100vs200recovery is exact;its200prefix seeds its own treatment.
Direct200is additional compute. Uninterrupted real-data budget32,200updates.
A later tool interruption required resuming the fifth treatment from
update1400,last observed heartbeat1550. At least150updates were repeated;
the exact discarded count and old process exit code were not observed.
32,200is therefore the planned budget,not a measured total including
that interruption. The four completed fits were reused unchanged.
All6actor and6camera-CV prediction banks freeze before scoring;all original
predictions and scalars reproduce exactly. The generic loader opens/hashes
target arrays for integrity,but targets/masks do not enter model inputs.

Fresh causal composition retains125540rows,1613queries.
94242rows use complete seven-pair support;all remaining rows
remain with explicit identity imputation. That is not measured zero camera
motion or a frozen raw-model fallback. The newly trained weights can change
unsupported forecasts. No donor/test statistics normalize model inputs.

Cached_verified:the six raw controls,source split,packed geometry and
previously measured past image-affine proxies. Not_run:new own-B/C/H
readout,new controller fitting,three forecaster seeds or independent
confirmation. Conventional viewpoint conditioning is not a new method
contribution by itself and does not inherit a metric/equivariance proof.

Nativearm64CPU4/inter-op1/workers0,full4000endpoint,checkpoint/heartbeat/
resume,no quick downgrade or threshold search. Native observation8/
prediction12,stride1,pixel coordinates,not SDDt50,meters,verified seconds,
true3D,foundation,human-gold or physical-safety success. Stage5C/SMCoff.

[Protocol](protocol.md) | [Model scope](model_scope.md) | [Causal inputs](input_operator_report.md) |
[Loss trace](loss_trace.csv) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
