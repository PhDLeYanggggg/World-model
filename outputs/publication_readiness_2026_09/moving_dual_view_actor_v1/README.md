# Does Additional Per-Node Observed-View Motion Help?

I retain native geometry and supply aligned histories as additional node
features. There is no hard future-camera restoration or trajectory residual
decoder. The matched native-node control has the same extra capacity.

Overall component screen:**FAIL**.

| Fixed comparison | Known pp | Completion lower pp | Exploratory95%source interval | Positive sources | Screen |
|---|---:|---:|---|---:|---|
| Primary:camera-native node | +0.8276 | -4.9788 | [-7.6915,-2.3328] | 10/14 | False |
| Required camera-raw bundle | -0.3208 | -5.9952 | [-8.4610,-3.5302] | 6/14 | False |

Primary fold known means:{'0': -0.12156664862828581, '1': 1.5394937220892906};camera/raw:{'0': -1.5113125165347434, '1': 0.5720526943955268}.
Both screens are required;beating a worse capacity control alone cannot
license controller fitting. No new policy,test readout or deployment.

## Every Fixed Contrast And Prefix

| Contrast | Prefix | Control gain vs CV % | New gain vs CV % | Paired known pp | Lower pp | Upper pp |
|---|---:|---:|---:|---:|---:|---:|
| primary | 1 | -12.8570 | -11.7342 | +1.1228 | -4.1807 | +6.4262 |
| primary | 3 | +6.3623 | +6.9320 | +0.5697 | -4.3681 | +5.5075 |
| primary | 6 | +12.6837 | +13.5493 | +0.8656 | -3.7193 | +5.4505 |
| primary | 12 | +14.2293 | +15.0569 | +0.8276 | -4.9788 | +6.6340 |
| camera_bundle | 1 | -13.6826 | -11.7342 | +1.9484 | -2.3472 | +6.2439 |
| camera_bundle | 3 | +6.6831 | +6.9320 | +0.2489 | -4.5198 | +5.0176 |
| camera_bundle | 6 | +13.6088 | +13.5493 | -0.0596 | -4.6390 | +4.5199 |
| camera_bundle | 12 | +15.3777 | +15.0569 | -0.3208 | -5.9952 | +5.3536 |
| native_bundle | 1 | -13.6826 | -12.8570 | +0.8256 | -4.0388 | +5.6899 |
| native_bundle | 3 | +6.6831 | +6.3623 | -0.3208 | -4.4809 | +3.8393 |
| native_bundle | 6 | +13.6088 | +12.6837 | -0.9252 | -4.9960 | +3.1457 |
| native_bundle | 12 | +15.3777 | +14.2293 | -1.1484 | -6.4529 | +4.1560 |
| both_effect | 1 | -13.6826 | -14.8079 | -1.1253 | -8.4442 | +6.1936 |
| both_effect | 3 | +6.6831 | +1.4205 | -5.2626 | -13.3551 | +2.8299 |
| both_effect | 6 | +13.6088 | +7.2581 | -6.3508 | -17.3335 | +4.6319 |
| both_effect | 12 | +15.3777 | +10.2447 | -5.1331 | -21.4474 | +11.1813 |
| camera_over_camera_cv | 1 | +0.5444 | -11.7342 | -12.2786 | -19.7608 | -4.7964 |
| camera_over_camera_cv | 3 | +1.6963 | +6.9320 | +5.2356 | -7.7391 | +18.2104 |
| camera_over_camera_cv | 6 | +2.5838 | +13.5493 | +10.9655 | -7.1140 | +29.0450 |
| camera_over_camera_cv | 12 | +2.8275 | +15.0569 | +12.2294 | -14.7361 | +39.1948 |
| camera_cv_effect | 1 | +0.0000 | +0.5444 | +0.5444 | +0.0571 | +1.0317 |
| camera_cv_effect | 3 | +0.0000 | +1.6963 | +1.6963 | +0.2506 | +3.1421 |
| camera_cv_effect | 6 | +0.0000 | +2.5838 | +2.5838 | -0.1746 | +5.3421 |
| camera_cv_effect | 12 | +0.0000 | +2.8275 | +2.8275 | -2.6114 | +8.2665 |

The primary is matched camera-node minus native-node. Camera/raw and
native/raw are extra-capacity bundle contrasts. Old hard-restoration
and camera-CV controls are cached_verified,not fresh model fits. Their
readout and all seven causal baseline statistics match the parent exactly.
Means balance queries within source then all sources;NA is not dropped.

## Every Source,Primary Whole12

| Source | Rows | Known | Partial | Unknown | Known pp | Lower pp | New easy +harm % |
|---|---:|---:|---:|---:|---:|---:|---:|
| moving_source_component_047 | 2198 | 2167 | 450 | 31 | -4.0536 | -5.8157 | +56.8524 |
| moving_source_component_048 | 2038 | 1997 | 618 | 41 | -5.9043 | -8.4047 | +45.8452 |
| moving_source_component_051 | 7196 | 7128 | 874 | 68 | +2.0163 | -6.4718 | +69.5870 |
| moving_source_component_052 | 536 | 518 | 73 | 18 | +0.2155 | -12.4284 | +41.8287 |
| moving_source_component_053 | 1137 | 1095 | 570 | 42 | -2.4191 | -5.9601 | +157.9267 |
| moving_source_component_062 | 37829 | 37472 | 4129 | 357 | +5.3905 | +3.6977 | +203.9206 |
| moving_source_component_065 | 1042 | 941 | 853 | 101 | +5.3966 | -9.3228 | +39.2283 |
| moving_source_component_067 | 4159 | 4006 | 1538 | 153 | +1.6062 | -1.9279 | +35.5239 |
| moving_source_component_076 | 3342 | 3184 | 1604 | 158 | +1.2869 | -3.9422 | +110.1055 |
| moving_source_component_077 | 6127 | 5943 | 2111 | 184 | -0.7936 | -3.0645 | +68.5963 |
| moving_source_component_078 | 40194 | 39159 | 11775 | 1035 | +3.5608 | -0.1268 | +17.2678 |
| moving_source_component_080 | 10670 | 10523 | 1806 | 147 | +1.8323 | +0.3691 | +50.8029 |
| moving_source_component_082 | 6225 | 5930 | 3572 | 295 | +2.6047 | -0.1158 | +142.6605 |
| moving_source_component_087 | 2847 | 2653 | 1778 | 194 | +0.8472 | -16.1897 | +122.3838 |

Raw camera-node known-easy +harm exceeds2%in33/34recordings;
0undefined references remain NA. This is not a full policy certificate.
All recordings,camera support strata and baseline diagnostics remain in
the scalar receipt;no favorable prefix or source replaces the primary.

## Actual Execution And Limits

Fresh_run:twelve full4000update treatment models,one full frozen-encoder
raw control and two complete first-design refits,one per learning arm.
Original shared core,active optimizer state,sampler/RNG/factors/common
trace match the old control. Two extra inactive optimizer parameter entries
are expected,not identical complete parameter lists. Both learning arms
have3,029,060nominal/582,468trainable parameters,1,792extra node parameters.

Real100+100and actual in-forward SIGTERM1+199both match continuous200.
The immutable step1stop snapshot is preserved. Three recovery trajectories
add400updates beyond the prefix reused by its own treatment. The actual
total is60,400updates. The disk-reserve stop retained3935committed steps
and resumed for exactly65more;there was no budget reduction. A partial-
update failure retains a coherent earlier
checkpoint;it never certifies an advanced incomplete sampler as committed.

All twelve new prediction banks freeze before scoring and replay exactly;
all scalar evidence replays exactly. Eighteen cached raw/old-both/camera-CV
banks are hash/row-alignment verified,not freshly inferred here. Generic
loading opens/hashes label arrays for integrity;targets/masks enter only
loss/evaluation. No future-camera operator,endpoint,central velocity or
test endpoint goal enters the model. Native geometry is always retained.

Previously exposed125540producer-Awindows,14sources,34recordings,1613queries.
One forecaster seed17. Native observation8/prediction12,stride1,image
coordinates;not SDDt50,verified seconds,meters,homography,true3D,foundation,
human-gold labels or physical-safety success. Original per-recording2%all/
easy positive-harm,signed easy-net,reference/support/NA rules unchanged.
No identity certification,epsilon,new own-B/C/Hreadout,new controller or
deployment. A two-view encoder alone is not a new method contribution.
Formal three-forecaster-seed main results,independent confirmation and
full selected-policy pass remain not_run. Submission not ready;Stage5C/
SMCoff. Earlier completed failures and the unscored stopped trial remain.

[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Loss traces](loss_trace.csv) | [Reproduction](operation_zh.md)
