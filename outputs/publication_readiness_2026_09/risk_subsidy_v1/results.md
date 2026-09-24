# Query Risk Credit and Denominator Results

All fixed controls; no winner selected or deployed.
Obs8/pred12 native SDD pixels, four development-exposed sites, three seeds. Not historical t+50.
CI: 3000 physical-site bootstrap draws, not independent confirmation or calibrated safety.

| Predictor / rule | ADE gain % | CI95 | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed |
|---|---:|---|---:|---:|---:|---:|
| damped_velocity_005 / old_strict | 3.634683 | [2.6422692738600757, 4.685964407975451] | 4.723212 | 2.494391 | 12.2520 | 0 |
| damped_velocity_005 / positive_population | 0.738313 | [0.3306794225313908, 1.3552036301571369] | 0.379018 | 0.000000 | 5.8295 | 0 |
| damped_velocity_005 / net_point | 0.535988 | [0.20981980884233753, 1.1107291930470304] | 0.023266 | 0.000000 | 5.2122 | 0 |
| damped_velocity_005 / parent_net_population | 1.094468 | [0.5540007230557503, 1.903244902867582] | 0.852556 | 0.000000 | 8.0238 | 0 |
| damped_velocity_005 / net_population | 1.094416 | [0.5539488654765917, 1.903244902867582] | 0.852434 | 0.000000 | 8.0238 | 0 |
| damped_velocity_005 / net_clipped_population | 0.870828 | [0.4285331330633996, 1.5507709052989926] | 0.434909 | 0.000000 | 7.2933 | 0 |
| damped_velocity_005 / net_selected | 0.755697 | [0.2857264734210885, 1.5189957253223225] | 0.352249 | 0.000000 | 6.1936 | 0 |
| damped_velocity_005 / net_clipped_selected | 0.556687 | [0.219359713888595, 1.1493363937652794] | 0.031335 | 0.000000 | 5.3834 | 0 |
| damped_velocity_005 / matched_net_population | 0.759011 | [0.3230084251442167, 1.453420061916419] | 0.460273 | 0.000000 | 5.3834 | 0 |
| damped_velocity_005 / matched_net_clipped_population | 0.660399 | [0.28770847299133817, 1.278162376649744] | 0.226643 | 0.000000 | 5.3834 | 0 |
| damped_velocity_005 / matched_net_selected | 0.633525 | [0.2476767529812962, 1.291518581925205] | 0.207667 | 0.000000 | 5.3834 | 0 |
| transformer / old_strict | 2.436827 | [1.6792440292285882, 3.111364306600029] | 2.291813 | 1.066900 | 6.6755 | 0 |
| transformer / positive_population | 1.042923 | [0.4253962459562055, 1.7490544468590108] | 0.585153 | 0.312359 | 9.5623 | 2 |
| transformer / net_point | 1.326354 | [0.34466469059601934, 2.3080431081873467] | 0.565312 | 0.000000 | 5.4403 | 0 |
| transformer / parent_net_population | 2.939195 | [1.1159886535132224, 4.762401606464062] | 2.434926 | 0.863564 | 19.5843 | 3 |
| transformer / net_population | 2.939203 | [1.1159886535132224, 4.762416424190674] | 2.434926 | 0.865803 | 19.5843 | 3 |
| transformer / net_clipped_population | 2.273613 | [0.9592876987406629, 3.5879382063655427] | 1.435497 | 0.367567 | 16.5165 | 2 |
| transformer / net_selected | 2.182812 | [0.5410664980406932, 3.8245566730082947] | 1.628906 | 0.000000 | 11.1332 | 1 |
| transformer / net_clipped_selected | 1.395384 | [0.3635632184295634, 2.4272049912624496] | 0.595417 | 0.000000 | 6.4648 | 0 |
| transformer / matched_net_population | 1.854965 | [0.5892918599455177, 3.1206383240151347] | 1.417820 | 0.000000 | 6.4648 | 0 |
| transformer / matched_net_clipped_population | 1.632007 | [0.5341189209952968, 2.729894554535223] | 0.936426 | 0.000000 | 6.4648 | 0 |
| transformer / matched_net_selected | 1.694423 | [0.43916457453712643, 2.949682154850203] | 1.136177 | 0.000000 | 6.4648 | 0 |
| eqmotion / old_strict | 1.609305 | [0.7162681896853484, 3.039280619820001] | 0.527132 | 0.452161 | 3.9265 | 0 |
| eqmotion / positive_population | 0.790537 | [0.28550148837307443, 1.3558636588587163] | 0.282740 | 0.641763 | 3.8954 | 0 |
| eqmotion / net_point | 1.234337 | [0.2790658400730839, 2.8586285586657243] | 0.039694 | 0.000000 | 4.0742 | 0 |
| eqmotion / parent_net_population | 2.826823 | [0.8690457377094563, 5.581929709546868] | 1.680746 | 1.838357 | 13.7124 | 6 |
| eqmotion / net_population | 2.826836 | [0.8690457377094563, 5.581929709546868] | 1.680746 | 1.838357 | 13.7124 | 6 |
| eqmotion / net_clipped_population | 1.983687 | [0.6827746079242081, 3.860011896534349] | 0.474680 | 0.769668 | 10.0653 | 0 |
| eqmotion / net_selected | 2.261437 | [0.5227781159801381, 4.994356395058631] | 1.218582 | 0.269947 | 8.9493 | 5 |
| eqmotion / net_clipped_selected | 1.299489 | [0.29967721485433807, 3.000434589662673] | 0.049721 | 0.000000 | 4.4391 | 0 |
| eqmotion / matched_net_population | 1.525671 | [0.3844310658072059, 3.369958907437737] | 0.595736 | 0.208981 | 4.4391 | 0 |
| eqmotion / matched_net_clipped_population | 1.433764 | [0.37453925011883826, 3.192681271382092] | 0.199024 | 0.000000 | 4.4391 | 0 |
| eqmotion / matched_net_selected | 1.450271 | [0.3260654860045842, 3.3200410488623215] | 0.470427 | 0.000000 | 4.4391 | 0 |

Missing labels are unknown. Solver failures, failed-reference matched queries, tails, partial-label bounds
and full contrasts remain in analysis.json. Signed and clipped-net risk are not positive-harm guarantees.
