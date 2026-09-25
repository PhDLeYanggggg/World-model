# Bridge Attribution Results

Fresh motion-only cost training; full-pair fits and inputs cached_verified; fresh fixed readout.
Six reused opened model-selection localities, not independent confirmation. Three seeds;
18 dependent producer/controller/seed configurations; 396 policy views. No selected winner.

All gains below compare delivered predictions with the preceding full neural all-risk bridge.

| Pair and scoring policy | All gain | Hard gain | Gain vs training-selected motion | Easy degradation | Easy passes |
|---|---:|---:|---:|---:|---:|
| full__candidate | -0.380963% to +1.469546% | +0.072497% to +2.580782% | +3.094065% to +11.734089% | +0.000000% to +6.076432% | 12/18 |
| full__hash_matched | -3.884623% to -0.145760% | -4.655640% to -0.207382% | +2.476065% to +10.928285% | +0.000000% to +0.000000% | 18/18 |
| full__neural | +0.000000% to +0.000000% | +0.000000% to +0.000000% | +2.893989% to +11.358978% | +0.000000% to +0.730345% | 18/18 |
| full__neural_common | -0.119401% to +0.001182% | -0.017477% to +0.012987% | +2.883481% to +11.340590% | +0.000000% to +0.715792% | 18/18 |
| full__neural_matched | -3.739113% to -0.026194% | -4.598576% to -0.030149% | +2.768368% to +11.050161% | +0.000000% to +0.000000% | 18/18 |
| full__neural_utility_ridge_risk | -3.860022% to +0.270608% | -4.754675% to +0.370739% | +2.964674% to +11.411528% | +0.000000% to +0.000000% | 18/18 |
| full__reference | -6.084049% to -4.245729% | -6.810174% to -4.638224% | -1.829906% to +6.360054% | +0.000000% to +0.000000% | 18/18 |
| full__ridge | -3.865930% to +0.269596% | -4.746578% to +0.373026% | +2.965437% to +11.426894% | +0.000000% to +0.000000% | 18/18 |
| full__ridge_common | -3.864319% to +0.264472% | -4.746918% to +0.369348% | +2.964322% to +11.422180% | +0.000000% to +0.000000% | 18/18 |
| full__ridge_matched | -3.888884% to -0.017349% | -4.753558% to +0.062107% | +2.822738% to +11.139206% | +0.000000% to +0.000000% | 18/18 |
| full__ridge_utility_neural_risk | -0.109346% to +0.015215% | -0.016498% to +0.028913% | +2.889661% to +11.340754% | +0.000000% to +0.474504% | 18/18 |
| motion_only__candidate | -0.631798% to +0.755988% | -0.665004% to +1.235142% | +2.958371% to +11.091545% | +0.000000% to +0.000000% | 18/18 |
| motion_only__hash_matched | -2.635879% to -0.036172% | -2.852171% to -0.029722% | +2.694781% to +10.659541% | +0.000000% to +0.000000% | 18/18 |
| motion_only__neural | -0.697843% to +0.818339% | -0.768451% to +0.804902% | +2.892281% to +10.956458% | +0.000000% to +0.000000% | 18/18 |
| motion_only__neural_common | -0.701072% to +0.829597% | -0.769087% to +0.791372% | +2.878588% to +10.956966% | +0.000000% to +0.000000% | 18/18 |
| motion_only__neural_matched | -2.453905% to +0.010612% | -2.685763% to +0.007423% | +2.778346% to +10.700525% | +0.000000% to +0.000000% | 18/18 |
| motion_only__neural_utility_ridge_risk | -2.315233% to +0.097395% | -2.526774% to +0.118224% | +2.876382% to +10.921456% | +0.000000% to +0.000000% | 18/18 |
| motion_only__reference | -6.292869% to -4.386211% | -6.949371% to -4.701116% | -1.901913% to +6.217606% | +0.000000% to +0.000000% | 18/18 |
| motion_only__ridge | -2.315203% to +0.094026% | -2.527173% to +0.143996% | +2.880434% to +10.943493% | +0.000000% to +0.000000% | 18/18 |
| motion_only__ridge_common | -2.315203% to +0.094831% | -2.527173% to +0.129556% | +2.876832% to +10.920413% | +0.000000% to +0.000000% | 18/18 |
| motion_only__ridge_matched | -2.316693% to +0.079274% | -2.529059% to +0.068416% | +2.837767% to +10.811980% | +0.000000% to +0.000000% | 18/18 |
| motion_only__ridge_utility_neural_risk | -0.690536% to +0.821482% | -0.767480% to +0.781780% | +2.897297% to +10.962400% | +0.000000% to +0.000000% | 18/18 |

Ranges cover all configurations, not best seeds. Easy means worst-locality net
degradation relative to causal CV on the producer-training-defined positive-easy event.
Positive harm is separate and remains uncalibrated. Full FDE, complete-label, p95/p99,
per-locality errors and predicted/realized moment accounting are retained in group JSON.

## Primary Three-Seed Contrasts

Seeds are averaged within locality before 3,000 locality resamples. All six localities
remain the independent unit. Intervals are exploratory, without multiplicity adjustment.

| Producer/controller | Contrast | All gain | 95% locality CI | Hard gain |
|---|---|---:|---:|---:|
| producer0_controller1 | full__neural_vs_ridge | -0.076226% | -0.380462% to +0.186595% | -0.336173% |
| producer0_controller1 | full__neural_matched_vs_ridge_matched | -0.073700% | -0.153501% to +0.007949% | -0.152749% |
| producer0_controller1 | motion_only__neural_vs_ridge | -0.036198% | -0.194754% to +0.117178% | -0.142337% |
| producer0_controller1 | motion_only__neural_matched_vs_ridge_matched | -0.106406% | -0.229882% to -0.034769% | -0.162821% |
| producer0_controller1 | full_neural_vs_motion_neural | +0.560254% | +0.148008% to +0.973160% | +0.439642% |
| producer0_controller2 | full__neural_vs_ridge | +2.700990% | +1.955858% to +3.567825% | +2.853406% |
| producer0_controller2 | full__neural_matched_vs_ridge_matched | -0.090299% | -0.158613% to -0.019709% | -0.172245% |
| producer0_controller2 | motion_only__neural_vs_ridge | +1.468537% | +1.138014% to +1.906989% | +1.650556% |
| producer0_controller2 | motion_only__neural_matched_vs_ridge_matched | -0.142935% | -0.256492% to -0.057691% | -0.161053% |
| producer0_controller2 | full_neural_vs_motion_neural | +0.492477% | +0.155782% to +0.829146% | +0.337818% |
| producer1_controller0 | full__neural_vs_ridge | +1.039894% | +0.492807% to +1.606978% | +1.424607% |
| producer1_controller0 | full__neural_matched_vs_ridge_matched | +0.071141% | -0.002545% to +0.140144% | +0.047243% |
| producer1_controller0 | motion_only__neural_vs_ridge | +0.103199% | -0.275744% to +0.408112% | +0.351282% |
| producer1_controller0 | motion_only__neural_matched_vs_ridge_matched | -0.067320% | -0.098808% to -0.029617% | -0.072685% |
| producer1_controller0 | full_neural_vs_motion_neural | +0.106963% | -0.370134% to +0.554330% | +0.511910% |
| producer1_controller2 | full__neural_vs_ridge | +3.099757% | +2.282385% to +3.935878% | +3.444678% |
| producer1_controller2 | full__neural_matched_vs_ridge_matched | +0.031228% | -0.032788% to +0.116696% | +0.005527% |
| producer1_controller2 | motion_only__neural_vs_ridge | +1.364644% | +1.073870% to +1.802384% | +1.384177% |
| producer1_controller2 | motion_only__neural_matched_vs_ridge_matched | -0.146654% | -0.317140% to -0.016691% | -0.206677% |
| producer1_controller2 | full_neural_vs_motion_neural | -0.322571% | -1.390724% to +0.580030% | -0.152995% |
| producer2_controller0 | full__neural_vs_ridge | -0.064680% | -0.146215% to -0.007911% | -0.059942% |
| producer2_controller0 | full__neural_matched_vs_ridge_matched | -0.007778% | -0.035063% to +0.015064% | -0.007140% |
| producer2_controller0 | motion_only__neural_vs_ridge | +0.052787% | +0.010222% to +0.097965% | +0.039815% |
| producer2_controller0 | motion_only__neural_matched_vs_ridge_matched | -0.008130% | -0.014247% to -0.002333% | -0.004625% |
| producer2_controller0 | full_neural_vs_motion_neural | +0.009912% | -0.125360% to +0.148060% | +0.002557% |
| producer2_controller1 | full__neural_vs_ridge | -0.145470% | -0.216996% to -0.076480% | -0.173478% |
| producer2_controller1 | full__neural_matched_vs_ridge_matched | -0.075553% | -0.140024% to -0.023781% | -0.102414% |
| producer2_controller1 | motion_only__neural_vs_ridge | +0.005349% | -0.082620% to +0.086301% | -0.053450% |
| producer2_controller1 | motion_only__neural_matched_vs_ridge_matched | -0.087808% | -0.139296% to -0.038671% | -0.100553% |
| producer2_controller1 | full_neural_vs_motion_neural | -0.021661% | -0.104167% to +0.067724% | -0.048247% |

## Matched Query Coverage

| Group | Pair | Queries | Zero-count queries | Matched actions | Different rows | Different queries |
|---|---|---:|---:|---:|---:|---:|
| fold0_seed17_controller1 | full | 7087 | 3342 | 13388 | 1034 | 441 |
| fold0_seed17_controller1 | motion_only | 7087 | 3532 | 11042 | 738 | 323 |
| fold0_seed17_controller2 | full | 7087 | 4481 | 5320 | 2072 | 863 |
| fold0_seed17_controller2 | motion_only | 7087 | 4410 | 6152 | 1478 | 653 |
| fold0_seed29_controller1 | full | 7087 | 3119 | 12495 | 1626 | 684 |
| fold0_seed29_controller1 | motion_only | 7087 | 3339 | 10536 | 764 | 339 |
| fold0_seed29_controller2 | full | 7087 | 3891 | 7507 | 2754 | 1122 |
| fold0_seed29_controller2 | motion_only | 7087 | 3978 | 7412 | 1556 | 683 |
| fold0_seed43_controller1 | full | 7087 | 3133 | 12341 | 1974 | 803 |
| fold0_seed43_controller1 | motion_only | 7087 | 3587 | 9562 | 1282 | 532 |
| fold0_seed43_controller2 | full | 7087 | 4313 | 4849 | 2072 | 870 |
| fold0_seed43_controller2 | motion_only | 7087 | 4469 | 5763 | 1708 | 735 |
| fold1_seed17_controller0 | full | 7087 | 3643 | 7540 | 2336 | 966 |
| fold1_seed17_controller0 | motion_only | 7087 | 3582 | 10140 | 566 | 270 |
| fold1_seed17_controller2 | full | 7087 | 5679 | 1738 | 502 | 245 |
| fold1_seed17_controller2 | motion_only | 7087 | 4276 | 5918 | 1500 | 625 |
| fold1_seed29_controller0 | full | 7087 | 3533 | 8495 | 2734 | 1103 |
| fold1_seed29_controller0 | motion_only | 7087 | 3549 | 9615 | 746 | 340 |
| fold1_seed29_controller2 | full | 7087 | 5477 | 1950 | 958 | 443 |
| fold1_seed29_controller2 | motion_only | 7087 | 4225 | 5867 | 948 | 438 |
| fold1_seed43_controller0 | full | 7087 | 3281 | 10239 | 2084 | 910 |
| fold1_seed43_controller0 | motion_only | 7087 | 3639 | 9423 | 672 | 321 |
| fold1_seed43_controller2 | full | 7087 | 4866 | 3177 | 1050 | 497 |
| fold1_seed43_controller2 | motion_only | 7087 | 4239 | 5871 | 1338 | 584 |
| fold2_seed17_controller0 | full | 7087 | 3294 | 12644 | 698 | 321 |
| fold2_seed17_controller0 | motion_only | 7087 | 3328 | 12046 | 134 | 66 |
| fold2_seed17_controller1 | full | 7087 | 3253 | 11914 | 1722 | 703 |
| fold2_seed17_controller1 | motion_only | 7087 | 3332 | 11238 | 556 | 256 |
| fold2_seed29_controller0 | full | 7087 | 3263 | 12216 | 228 | 112 |
| fold2_seed29_controller0 | motion_only | 7087 | 3274 | 11885 | 122 | 61 |
| fold2_seed29_controller1 | full | 7087 | 3358 | 11000 | 974 | 429 |
| fold2_seed29_controller1 | motion_only | 7087 | 3344 | 10870 | 598 | 279 |
| fold2_seed43_controller0 | full | 7087 | 3367 | 11532 | 428 | 199 |
| fold2_seed43_controller0 | motion_only | 7087 | 3378 | 11649 | 142 | 68 |
| fold2_seed43_controller1 | full | 7087 | 3330 | 11134 | 1712 | 703 |
| fold2_seed43_controller1 | motion_only | 7087 | 3448 | 10716 | 608 | 272 |

Matched counts include unknown-label rows. They are not a common estimated risk
budget: each scorer has its own moments; hash ranking is not necessarily risk-guarded.
Motion-only excludes neural trajectories and policy bits, not learned floor/cost scorers.
Neural risk uses hurdle/ranking losses and ridge uses moment regression; their contrast
does not isolate architecture nonlinearity. Full vs motion-only changes the forecast pair.
Image-pixel native annotation-step results only. No physical-safety or deployment claim.
