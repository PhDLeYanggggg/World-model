# Signed Easy-Risk Results

Fresh 36-head training; frozen source-excluded forecasts and outcomes cached-verified. Four exposed sites, three seeds.
Obs8/pred12 annotation pixels, equal-site gains vs CV. NOT t+50, confirmation or safety certification.
Worst easy degradation is the maximum over individual site/seed views, not a pooled mean.

| Predictor / policy | ADE gain % | Site CI95 | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed |
|---|---:|---|---:|---:|---:|---:|
| damped_velocity_005 / old_net | 5.281168 | [3.9558913413655254, 6.606443924091549] | 8.241735 | 13.523159 | 26.8933 | 3 |
| damped_velocity_005 / old_strict | 3.634683 | [2.6422692738600757, 4.685964407975451] | 4.723212 | 2.494391 | 12.2520 | 0 |
| damped_velocity_005 / old_positive_point | 0.124052 | [0.02317214654763422, 0.2850477403262275] | 0.000363 | 0.000000 | 1.4592 | 0 |
| damped_velocity_005 / old_positive_population | 0.921295 | [0.4881816678120654, 1.533319829689861] | 0.622880 | 0.000000 | 6.0279 | 0 |
| damped_velocity_005 / positive_point | 0.081779 | [0.019179146331865793, 0.18847259997168964] | 0.000114 | 0.000000 | 1.0027 | 0 |
| damped_velocity_005 / net_point | 0.535988 | [0.20981980884233753, 1.1107291930470304] | 0.023266 | 0.000000 | 5.2122 | 0 |
| damped_velocity_005 / positive_population | 0.738313 | [0.3306794225313908, 1.3552036301571369] | 0.379018 | 0.000000 | 5.8295 | 0 |
| damped_velocity_005 / net_population | 1.094468 | [0.5540007230557503, 1.903244902867582] | 0.852556 | 0.000000 | 8.0238 | 0 |
| damped_velocity_005 / net_matched_positive | 0.844391 | [0.39569528122564424, 1.5194909467287965] | 0.628800 | 0.117444 | 5.8295 | 0 |
| transformer / old_net | 8.379146 | [6.729993859539871, 9.935973210495353] | 10.251325 | 21.732634 | 54.3650 | 21 |
| transformer / old_strict | 2.436827 | [1.6792440292285882, 3.111364306600029] | 2.291813 | 1.066900 | 6.6755 | 0 |
| transformer / old_positive_point | 0.013424 | [0.0023143968750360955, 0.031447857944547075] | 0.000048 | 0.000000 | 0.2321 | 0 |
| transformer / old_positive_population | 1.280156 | [0.5647475654103257, 2.1160313342163284] | 0.642462 | 0.112018 | 10.7592 | 2 |
| transformer / positive_point | 0.002736 | [0.000648661632765557, 0.005267083608515022] | 0.000055 | 0.000000 | 0.1318 | 0 |
| transformer / net_point | 1.326354 | [0.34466469059601934, 2.3080431081873467] | 0.565312 | 0.000000 | 5.4403 | 0 |
| transformer / positive_population | 1.042923 | [0.4253962459562055, 1.7490544468590108] | 0.585153 | 0.312359 | 9.5623 | 2 |
| transformer / net_population | 2.939195 | [1.1159886535132224, 4.762401606464062] | 2.434926 | 0.863564 | 19.5843 | 3 |
| transformer / net_matched_positive | 1.459616 | [0.6588644630896368, 2.260367492974508] | 1.198893 | 0.728746 | 9.5617 | 2 |
| eqmotion / old_net | 11.006527 | [8.853700100060797, 13.159353941080205] | 13.330014 | 40.944883 | 69.1616 | 21 |
| eqmotion / old_strict | 1.609305 | [0.7162681896853484, 3.039280619820001] | 0.527132 | 0.452161 | 3.9265 | 0 |
| eqmotion / old_positive_point | 0.001340 | [0.0005001781941182948, 0.0024170343474100298] | 0.000000 | 0.000000 | 0.0099 | 0 |
| eqmotion / old_positive_population | 0.955560 | [0.35360620976072066, 1.6290514688420576] | 0.250405 | 0.038061 | 4.4234 | 0 |
| eqmotion / positive_point | 0.000000 | [0.0, 0.0] | 0.000000 | 0.000000 | 0.0000 | 0 |
| eqmotion / net_point | 1.234337 | [0.2790658400730839, 2.8586285586657243] | 0.039694 | 0.000000 | 4.0742 | 0 |
| eqmotion / positive_population | 0.790537 | [0.28550148837307443, 1.3558636588587163] | 0.282740 | 0.641763 | 3.8954 | 0 |
| eqmotion / net_population | 2.826823 | [0.8690457377094563, 5.581929709546868] | 1.680746 | 1.838357 | 13.7124 | 6 |
| eqmotion / net_matched_positive | 0.888999 | [0.32606440440344775, 1.4990258637819305] | 0.462465 | 0.734738 | 3.8954 | 0 |

All counts, missing-label support, tails and partial-label bounds remain in analysis.json.
Signed risk permits cancellation. Predicted-budget feasibility is not observed safety; all arms remain diagnostic.
No external readout, new forecaster, threshold selection, deployment, Stage5C or SMC.
