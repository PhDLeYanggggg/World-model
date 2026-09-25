# Nested Calibration Results

All 72 pointwise views; no outer-result model/threshold/seed selection. Three seeds and both predefined role rotations.
Forecasts and all scoring-head preprocessing exclude the internal calibration and outer localities.
Four fitting, four calibration and four outer localities per assignment; all are opened development sources.

| Seed/rotation/candidate/event/rule | ADE vs CV (%) | Conditional 95% CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harmed | Switch (%) |
|---|---:|---|---:|---:|---:|---:|---:|
| 17_direction1_neural_all_none | 0.926705 | [0.150360, 1.877154] | 1.448116 | 1.521737 | 2.458339 | 0/4 | 4.190689 |
| 17_direction1_neural_all_population_rescale | 0.655000 | [0.090157, 1.394382] | 1.077032 | 1.100410 | 1.574235 | 0/4 | 3.090269 |
| 17_direction1_neural_all_selected_risk_grid | 0.376817 | [0.065760, 0.833638] | 0.571782 | 0.627771 | 0.532497 | 0/4 | 1.916174 |
| 17_direction1_neural_easy_none | 0.628019 | [0.258904, 1.089575] | 0.966836 | 0.272359 | 0.577141 | 1/4 | 11.804282 |
| 17_direction1_neural_easy_population_rescale | 0.219238 | [0.112796, 0.339372] | 0.346980 | 0.076888 | 0.106542 | 0/4 | 4.138647 |
| 17_direction1_neural_easy_selected_risk_grid | 0.389052 | [0.182000, 0.635115] | 0.605313 | 0.150942 | 0.577141 | 1/4 | 8.145933 |
| 17_direction2_neural_all_none | -0.144910 | [-1.298171, 1.031334] | 0.012295 | 0.691957 | 3.530751 | 0/4 | 4.561572 |
| 17_direction2_neural_all_population_rescale | -0.087076 | [-1.070038, 0.960230] | 0.044778 | 0.604632 | 2.238071 | 0/4 | 3.644241 |
| 17_direction2_neural_all_selected_risk_grid | 0.146671 | [-0.435235, 0.893040] | 0.199182 | 0.673185 | 1.677363 | 0/4 | 1.891406 |
| 17_direction2_neural_easy_none | 0.445788 | [0.182624, 0.776163] | 0.652682 | 0.263741 | 0.410574 | 3/4 | 16.606316 |
| 17_direction2_neural_easy_population_rescale | 0.303871 | [0.076807, 0.596069] | 0.464205 | 0.170350 | 0.239263 | 2/4 | 13.078073 |
| 17_direction2_neural_easy_selected_risk_grid | 0.411728 | [0.160530, 0.722239] | 0.609935 | 0.235345 | 0.410574 | 2/4 | 15.256342 |
| 17_direction1_damping097_all_none | 3.036234 | [2.228237, 3.768987] | 4.765892 | 3.526461 | 0.839814 | 0/4 | 46.379742 |
| 17_direction1_damping097_all_population_rescale | 2.483888 | [1.764138, 3.174220] | 3.849143 | 2.876886 | 0.553170 | 0/4 | 35.584022 |
| 17_direction1_damping097_all_selected_risk_grid | 2.793842 | [2.025602, 3.495376] | 4.363077 | 3.276378 | 0.543400 | 0/4 | 42.027909 |
| 17_direction1_damping097_easy_none | 1.456501 | [0.805423, 2.214652] | 2.383988 | 1.019068 | -0.789304 | 0/4 | 38.190232 |
| 17_direction1_damping097_easy_population_rescale | 1.063632 | [0.350821, 1.889432] | 1.747145 | 0.882476 | -0.040368 | 0/4 | 25.553580 |
| 17_direction1_damping097_easy_selected_risk_grid | 1.233056 | [0.729865, 1.839328] | 2.018210 | 0.816052 | -0.789304 | 0/4 | 37.345008 |
| 17_direction2_damping097_all_none | 2.840245 | [2.120389, 3.552716] | 4.535971 | 3.405630 | 2.138760 | 0/4 | 29.916387 |
| 17_direction2_damping097_all_population_rescale | 2.090172 | [1.348020, 2.890778] | 3.320883 | 2.602755 | 1.351509 | 0/4 | 22.927620 |
| 17_direction2_damping097_all_selected_risk_grid | 2.265624 | [1.494118, 3.106516] | 3.604600 | 2.818135 | 2.138760 | 0/4 | 27.680433 |
| 17_direction2_damping097_easy_none | 1.259106 | [0.650264, 1.965292] | 2.000251 | 0.966900 | -0.337275 | 0/4 | 45.329170 |
| 17_direction2_damping097_easy_population_rescale | 0.965381 | [0.328995, 1.740059] | 1.539330 | 0.825945 | -0.035863 | 0/4 | 39.143302 |
| 17_direction2_damping097_easy_selected_risk_grid | 1.238534 | [0.645579, 1.921850] | 1.967429 | 0.944455 | -0.552086 | 0/4 | 45.058924 |
| 29_direction1_neural_all_none | 1.035108 | [0.287512, 1.956355] | 1.639364 | 1.869188 | 3.317388 | 1/4 | 12.029382 |
| 29_direction1_neural_all_population_rescale | 0.897707 | [0.340807, 1.580699] | 1.304978 | 1.590656 | 2.008120 | 1/4 | 10.778163 |
| 29_direction1_neural_all_selected_risk_grid | 0.530828 | [0.249862, 0.870343] | 0.742635 | 0.946569 | 1.227951 | 1/4 | 9.365800 |
| 29_direction1_neural_easy_none | 0.396767 | [0.238862, 0.573160] | 0.576252 | 0.194467 | 2.207477 | 1/4 | 10.597268 |
| 29_direction1_neural_easy_population_rescale | 0.208570 | [0.088586, 0.346756] | 0.293038 | 0.123045 | 1.029982 | 0/4 | 5.519659 |
| 29_direction1_neural_easy_selected_risk_grid | 0.396767 | [0.238862, 0.573160] | 0.576252 | 0.194467 | 2.207477 | 1/4 | 10.597268 |
| 29_direction2_neural_all_none | 0.375006 | [-0.627318, 1.574896] | 0.577499 | 1.370155 | 5.532697 | 0/4 | 4.610793 |
| 29_direction2_neural_all_population_rescale | 0.272248 | [-0.619304, 1.368725] | 0.477163 | 1.144437 | 5.825242 | 0/4 | 3.421022 |
| 29_direction2_neural_all_selected_risk_grid | 0.424325 | [-0.303040, 1.370122] | 0.559850 | 1.117901 | 3.071300 | 0/4 | 2.358850 |
| 29_direction2_neural_easy_none | 0.425427 | [0.246654, 0.626749] | 0.566317 | 0.265183 | 0.496603 | 0/4 | 8.335920 |
| 29_direction2_neural_easy_population_rescale | 0.242660 | [0.093912, 0.439331] | 0.300772 | 0.136566 | 0.317946 | 0/4 | 6.967762 |
| 29_direction2_neural_easy_selected_risk_grid | 0.333218 | [0.191919, 0.513698] | 0.431727 | 0.178652 | 0.351661 | 0/4 | 7.806088 |
| 29_direction1_damping097_all_none | 2.786548 | [1.881614, 3.613844] | 4.423992 | 3.211901 | 1.007278 | 0/4 | 39.757782 |
| 29_direction1_damping097_all_population_rescale | 2.208825 | [1.481485, 2.906964] | 3.440457 | 2.587334 | 1.007278 | 0/4 | 27.569450 |
| 29_direction1_damping097_all_selected_risk_grid | 2.603820 | [1.736521, 3.381739] | 4.119692 | 3.022195 | 1.007278 | 0/4 | 33.776637 |
| 29_direction1_damping097_easy_none | 1.370444 | [0.833966, 2.030810] | 2.243505 | 0.961280 | -0.859740 | 0/4 | 38.099000 |
| 29_direction1_damping097_easy_population_rescale | 0.998804 | [0.385774, 1.716093] | 1.635157 | 0.824883 | -0.071347 | 0/4 | 27.012029 |
| 29_direction1_damping097_easy_selected_risk_grid | 1.193737 | [0.746103, 1.733892] | 1.940909 | 0.788183 | -0.859740 | 0/4 | 37.696767 |
| 29_direction2_damping097_all_none | 2.692411 | [2.069374, 3.303875] | 4.331390 | 3.236671 | 1.494669 | 0/4 | 26.930830 |
| 29_direction2_damping097_all_population_rescale | 1.823442 | [1.053295, 2.633330] | 2.883595 | 2.334354 | 1.494669 | 0/4 | 19.142299 |
| 29_direction2_damping097_all_selected_risk_grid | 2.060336 | [1.310825, 2.842209] | 3.303392 | 2.590600 | 1.494669 | 0/4 | 24.408642 |
| 29_direction2_damping097_easy_none | 1.215078 | [0.664399, 1.839934] | 1.954098 | 0.915410 | -0.511253 | 0/4 | 44.676442 |
| 29_direction2_damping097_easy_population_rescale | 0.975751 | [0.378346, 1.675344] | 1.574846 | 0.802685 | -0.000000 | 0/4 | 41.516887 |
| 29_direction2_damping097_easy_selected_risk_grid | 1.215078 | [0.664399, 1.839934] | 1.954098 | 0.915410 | -0.511253 | 0/4 | 44.676442 |
| 43_direction1_neural_all_none | 1.521345 | [0.491923, 2.758405] | 2.393638 | 2.383153 | 5.198429 | 0/4 | 9.136311 |
| 43_direction1_neural_all_population_rescale | 1.028673 | [0.319717, 1.876235] | 1.627381 | 1.667862 | 2.582776 | 0/4 | 6.176149 |
| 43_direction1_neural_all_selected_risk_grid | 0.265343 | [0.025627, 0.583679] | 0.395729 | 0.353611 | 1.032168 | 0/4 | 2.777699 |
| 43_direction1_neural_easy_none | 0.510576 | [0.254434, 0.825829] | 0.768674 | 0.195471 | 0.015697 | 0/4 | 12.253228 |
| 43_direction1_neural_easy_population_rescale | 0.126335 | [0.069803, 0.191991] | 0.188671 | 0.046597 | -0.115083 | 0/4 | 3.145447 |
| 43_direction1_neural_easy_selected_risk_grid | 0.249931 | [0.115984, 0.420990] | 0.377117 | 0.113087 | 0.015697 | 0/4 | 7.547442 |
| 43_direction2_neural_all_none | 0.390407 | [-0.760567, 1.919470] | 0.911728 | 1.492183 | 5.446808 | 1/4 | 7.717364 |
| 43_direction2_neural_all_population_rescale | 0.268244 | [-0.759679, 1.619041] | 0.641950 | 1.234349 | 5.505686 | 0/4 | 5.608382 |
| 43_direction2_neural_all_selected_risk_grid | 0.409690 | [-0.412311, 1.369154] | 0.637939 | 0.989034 | 5.842501 | 0/4 | 2.005524 |
| 43_direction2_neural_easy_none | 0.437851 | [0.206439, 0.710879] | 0.657373 | 0.200462 | 0.142487 | 1/4 | 14.362211 |
| 43_direction2_neural_easy_population_rescale | 0.240298 | [0.062624, 0.469192] | 0.348319 | 0.097272 | 0.071070 | 1/4 | 10.937113 |
| 43_direction2_neural_easy_selected_risk_grid | 0.431561 | [0.196453, 0.706297] | 0.649593 | 0.195739 | 0.073852 | 1/4 | 14.252796 |
| 43_direction1_damping097_all_none | 2.880497 | [1.938513, 3.754598] | 4.571850 | 3.307858 | 1.121608 | 0/4 | 39.297549 |
| 43_direction1_damping097_all_population_rescale | 2.478469 | [1.652286, 3.271145] | 3.895469 | 2.916418 | 0.633725 | 0/4 | 30.993608 |
| 43_direction1_damping097_all_selected_risk_grid | 2.535061 | [1.684351, 3.353768] | 3.993622 | 2.971140 | 0.677599 | 0/4 | 32.558650 |
| 43_direction1_damping097_easy_none | 1.652369 | [0.936840, 2.506037] | 2.683500 | 1.240314 | -0.793081 | 0/4 | 35.599384 |
| 43_direction1_damping097_easy_population_rescale | 1.234512 | [0.458178, 2.138063] | 2.001832 | 1.036906 | -0.145500 | 0/4 | 25.249162 |
| 43_direction1_damping097_easy_selected_risk_grid | 1.333990 | [0.804207, 1.947907] | 2.181897 | 0.932662 | -0.793081 | 0/4 | 34.829090 |
| 43_direction2_damping097_all_none | 2.671415 | [1.881108, 3.506622] | 4.258738 | 3.214952 | 3.049421 | 0/4 | 29.756810 |
| 43_direction2_damping097_all_population_rescale | 2.075940 | [1.351263, 2.924804] | 3.294750 | 2.600925 | 1.482548 | 0/4 | 23.094407 |
| 43_direction2_damping097_all_selected_risk_grid | 2.447811 | [1.724250, 3.241597] | 3.893527 | 2.968924 | 1.482548 | 0/4 | 25.105261 |
| 43_direction2_damping097_easy_none | 1.468347 | [0.755548, 2.291941] | 2.366517 | 1.191537 | -0.399674 | 0/4 | 47.704636 |
| 43_direction2_damping097_easy_population_rescale | 1.238468 | [0.488492, 2.134212] | 1.992010 | 1.051986 | -0.007587 | 0/4 | 44.946688 |
| 43_direction2_damping097_easy_selected_risk_grid | 1.438179 | [0.746202, 2.241236] | 2.305094 | 1.159746 | -0.399674 | 0/4 | 47.478595 |

## Neural Versus Matched Damping

| View | Subset | Paired gain (%) | Conditional CI |
|---|---|---:|---|
| 17_direction1_neural_all_none | all | -2.184536 | [-3.073948, -1.317184] |
| 17_direction1_neural_all_none | hard | -2.078164 | [-3.261279, -0.795498] |
| 17_direction1_neural_all_none | easy | -0.691034 | [-1.173617, -0.230192] |
| 17_direction1_neural_all_population_rescale | all | -1.888239 | [-2.796620, -0.999551] |
| 17_direction1_neural_all_population_rescale | hard | -1.839725 | [-2.956586, -0.581700] |
| 17_direction1_neural_all_population_rescale | easy | -0.345487 | [-0.703913, 0.011649] |
| 17_direction1_neural_all_selected_risk_grid | all | -2.503172 | [-3.308172, -1.661080] |
| 17_direction1_neural_all_selected_risk_grid | hard | -2.754602 | [-3.651443, -1.804069] |
| 17_direction1_neural_all_selected_risk_grid | easy | -0.442047 | [-0.845331, -0.083746] |
| 17_direction1_neural_easy_none | all | -0.859166 | [-1.825909, 0.000473] |
| 17_direction1_neural_easy_none | hard | -0.774335 | [-1.698053, 0.031085] |
| 17_direction1_neural_easy_none | easy | -1.959106 | [-3.395172, -0.654301] |
| 17_direction1_neural_easy_population_rescale | all | -0.875034 | [-1.793195, -0.054225] |
| 17_direction1_neural_easy_population_rescale | hard | -0.831700 | [-1.680825, -0.084614] |
| 17_direction1_neural_easy_population_rescale | easy | -1.592947 | [-3.015451, -0.276413] |
| 17_direction1_neural_easy_selected_risk_grid | all | -0.865360 | [-1.564750, -0.291251] |
| 17_direction1_neural_easy_selected_risk_grid | hard | -0.682635 | [-1.383473, -0.088543] |
| 17_direction1_neural_easy_selected_risk_grid | easy | -2.142700 | [-3.228268, -1.129861] |
| 17_direction2_neural_all_none | all | -3.076347 | [-4.164310, -2.097136] |
| 17_direction2_neural_all_none | hard | -2.806285 | [-3.564529, -1.823876] |
| 17_direction2_neural_all_none | easy | -1.509079 | [-2.481736, -0.661217] |
| 17_direction2_neural_all_population_rescale | all | -2.232786 | [-3.289758, -1.273191] |
| 17_direction2_neural_all_population_rescale | hard | -2.056671 | [-2.869553, -1.133883] |
| 17_direction2_neural_all_population_rescale | easy | -0.715521 | [-1.285286, -0.209754] |
| 17_direction2_neural_all_selected_risk_grid | all | -2.180343 | [-2.967486, -1.452511] |
| 17_direction2_neural_all_selected_risk_grid | hard | -2.219784 | [-2.930922, -1.502061] |
| 17_direction2_neural_all_selected_risk_grid | easy | -0.512591 | [-0.979485, -0.076503] |
| 17_direction2_neural_easy_none | all | -0.840293 | [-1.703854, -0.066828] |
| 17_direction2_neural_easy_none | hard | -0.729155 | [-1.611774, 0.035967] |
| 17_direction2_neural_easy_none | easy | -1.398362 | [-2.579250, -0.157145] |
| 17_direction2_neural_easy_population_rescale | all | -0.685863 | [-1.551774, 0.056819] |
| 17_direction2_neural_easy_population_rescale | hard | -0.679563 | [-1.541641, 0.052089] |
| 17_direction2_neural_easy_population_rescale | easy | -0.514110 | [-1.404209, 0.317878] |
| 17_direction2_neural_easy_selected_risk_grid | all | -0.852671 | [-1.675881, -0.109156] |
| 17_direction2_neural_easy_selected_risk_grid | hard | -0.733680 | [-1.576472, -0.002001] |
| 17_direction2_neural_easy_selected_risk_grid | easy | -1.566145 | [-2.716886, -0.389015] |
| 29_direction1_neural_all_none | all | -1.813437 | [-2.671356, -0.969934] |
| 29_direction1_neural_all_none | hard | -1.389493 | [-2.455394, -0.206821] |
| 29_direction1_neural_all_none | easy | -0.717755 | [-1.356310, -0.103905] |
| 29_direction1_neural_all_population_rescale | all | -1.355755 | [-2.281203, -0.404870] |
| 29_direction1_neural_all_population_rescale | hard | -1.035423 | [-2.148501, 0.213986] |
| 29_direction1_neural_all_population_rescale | easy | -0.308852 | [-0.778201, 0.134706] |
| 29_direction1_neural_all_selected_risk_grid | all | -2.147746 | [-2.911370, -1.319709] |
| 29_direction1_neural_all_selected_risk_grid | hard | -2.160073 | [-2.971270, -1.341665] |
| 29_direction1_neural_all_selected_risk_grid | easy | -0.335022 | [-0.783211, 0.087247] |
| 29_direction1_neural_easy_none | all | -0.998650 | [-1.665610, -0.448000] |
| 29_direction1_neural_easy_none | hard | -0.786231 | [-1.462729, -0.246376] |
| 29_direction1_neural_easy_none | easy | -2.063659 | [-3.510301, -1.008856] |
| 29_direction1_neural_easy_population_rescale | all | -0.812220 | [-1.527537, -0.228250] |
| 29_direction1_neural_easy_population_rescale | hard | -0.720563 | [-1.413714, -0.168357] |
| 29_direction1_neural_easy_population_rescale | easy | -1.675257 | [-2.952447, -0.657400] |
| 29_direction1_neural_easy_selected_risk_grid | all | -0.814193 | [-1.384005, -0.368919] |
| 29_direction1_neural_easy_selected_risk_grid | hard | -0.605858 | [-1.146500, -0.191544] |
| 29_direction1_neural_easy_selected_risk_grid | easy | -1.957847 | [-3.360953, -0.933977] |
| 29_direction2_neural_all_none | all | -2.381133 | [-3.187457, -1.347843] |
| 29_direction2_neural_all_none | hard | -1.925239 | [-2.866784, -0.648942] |
| 29_direction2_neural_all_none | easy | -1.500718 | [-2.426373, -0.650640] |
| 29_direction2_neural_all_population_rescale | all | -1.589124 | [-2.494037, -0.588719] |
| 29_direction2_neural_all_population_rescale | hard | -1.224790 | [-2.326361, -0.036740] |
| 29_direction2_neural_all_population_rescale | easy | -0.742083 | [-1.591506, -0.063799] |
| 29_direction2_neural_all_selected_risk_grid | all | -1.677868 | [-2.402553, -0.861829] |
| 29_direction2_neural_all_selected_risk_grid | hard | -1.518956 | [-2.446053, -0.520830] |
| 29_direction2_neural_all_selected_risk_grid | easy | -0.567812 | [-1.086399, -0.101118] |
| 29_direction2_neural_easy_none | all | -0.811227 | [-1.495554, -0.209659] |
| 29_direction2_neural_easy_none | hard | -0.669834 | [-1.371925, -0.057492] |
| 29_direction2_neural_easy_none | easy | -1.408217 | [-2.058424, -0.782662] |
| 29_direction2_neural_easy_population_rescale | all | -0.754294 | [-1.469140, -0.124324] |
| 29_direction2_neural_easy_population_rescale | hard | -0.686001 | [-1.400222, -0.056844] |
| 29_direction2_neural_easy_population_rescale | easy | -0.881021 | [-1.520657, -0.317191] |
| 29_direction2_neural_easy_selected_risk_grid | all | -0.904666 | [-1.574873, -0.303725] |
| 29_direction2_neural_easy_selected_risk_grid | hard | -0.757212 | [-1.449156, -0.145570] |
| 29_direction2_neural_easy_selected_risk_grid | easy | -1.615444 | [-2.315369, -0.954375] |
| 43_direction1_neural_all_none | all | -1.403115 | [-2.292334, -0.519959] |
| 43_direction1_neural_all_none | hard | -0.942220 | [-2.243848, 0.527041] |
| 43_direction1_neural_all_none | easy | -0.842411 | [-1.615746, -0.179058] |
| 43_direction1_neural_all_population_rescale | all | -1.497049 | [-2.333641, -0.698713] |
| 43_direction1_neural_all_population_rescale | hard | -1.287402 | [-2.303844, -0.143533] |
| 43_direction1_neural_all_population_rescale | easy | -0.457884 | [-0.975990, 0.012050] |
| 43_direction1_neural_all_selected_risk_grid | all | -2.348153 | [-3.082289, -1.550023] |
| 43_direction1_neural_all_selected_risk_grid | hard | -2.717006 | [-3.490942, -1.939490] |
| 43_direction1_neural_all_selected_risk_grid | easy | -0.346200 | [-0.833793, 0.075939] |
| 43_direction1_neural_easy_none | all | -1.183493 | [-2.184773, -0.313173] |
| 43_direction1_neural_easy_none | hard | -1.080616 | [-2.029827, -0.267474] |
| 43_direction1_neural_easy_none | easy | -1.427108 | [-2.902439, -0.084734] |
| 43_direction1_neural_easy_population_rescale | all | -1.146424 | [-2.126061, -0.320392] |
| 43_direction1_neural_easy_population_rescale | hard | -1.022975 | [-1.943698, -0.243917] |
| 43_direction1_neural_easy_population_rescale | easy | -1.959636 | [-3.153116, -0.962800] |
| 43_direction1_neural_easy_selected_risk_grid | all | -1.110410 | [-1.813519, -0.487406] |
| 43_direction1_neural_easy_selected_risk_grid | hard | -0.838099 | [-1.493512, -0.275399] |
| 43_direction1_neural_easy_selected_risk_grid | easy | -2.126600 | [-3.367210, -0.952544] |
| 43_direction2_neural_all_none | all | -2.342278 | [-3.363519, -1.167487] |
| 43_direction2_neural_all_none | hard | -1.767676 | [-2.852312, -0.354394] |
| 43_direction2_neural_all_none | easy | -1.634934 | [-2.746374, -0.646307] |
| 43_direction2_neural_all_population_rescale | all | -1.845532 | [-2.714759, -0.888916] |
| 43_direction2_neural_all_population_rescale | hard | -1.392722 | [-2.315279, -0.195014] |
| 43_direction2_neural_all_population_rescale | easy | -0.898457 | [-1.845807, -0.110245] |
| 43_direction2_neural_all_selected_risk_grid | all | -2.091349 | [-2.687468, -1.486314] |
| 43_direction2_neural_all_selected_risk_grid | hard | -2.039661 | [-2.806117, -1.193230] |
| 43_direction2_neural_all_selected_risk_grid | easy | -1.218451 | [-2.115425, -0.432524] |
| 43_direction2_neural_easy_none | all | -1.067903 | [-2.049480, -0.235271] |
| 43_direction2_neural_easy_none | hard | -1.026698 | [-1.988266, -0.231552] |
| 43_direction2_neural_easy_none | easy | -0.789259 | [-1.913672, 0.348714] |
| 43_direction2_neural_easy_population_rescale | all | -1.033499 | [-1.981372, -0.251354] |
| 43_direction2_neural_easy_population_rescale | hard | -0.988036 | [-1.922082, -0.217716] |
| 43_direction2_neural_easy_population_rescale | easy | -0.839560 | [-1.842314, 0.031597] |
| 43_direction2_neural_easy_selected_risk_grid | all | -1.042396 | [-2.010818, -0.222038] |
| 43_direction2_neural_easy_selected_risk_grid | hard | -0.997922 | [-1.942393, -0.216643] |
| 43_direction2_neural_easy_selected_risk_grid | easy | -0.770885 | [-1.885243, 0.354018] |

## Calibration Versus Its Uncalibrated Control

| View | Subset | Paired gain (%) | Conditional CI |
|---|---|---:|---|
| 17_direction1_neural_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_neural_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_neural_all_population_rescale | all | -0.280978 | [-0.561806, -0.053384] |
| 17_direction1_neural_all_population_rescale | hard | -0.445543 | [-0.891087, -0.095008] |
| 17_direction1_neural_all_selected_risk_grid | all | -0.571018 | [-1.228791, 0.000000] |
| 17_direction1_neural_all_selected_risk_grid | hard | -0.950853 | [-1.981281, 0.000000] |
| 17_direction1_neural_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_neural_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_neural_easy_population_rescale | all | -0.415552 | [-0.769460, -0.124121] |
| 17_direction1_neural_easy_population_rescale | hard | -0.197009 | [-0.366398, -0.057218] |
| 17_direction1_neural_easy_selected_risk_grid | all | -0.243061 | [-0.474058, -0.044881] |
| 17_direction1_neural_easy_selected_risk_grid | hard | -0.122417 | [-0.234228, -0.026149] |
| 17_direction2_neural_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_neural_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_neural_all_population_rescale | all | 0.052531 | [-0.088967, 0.207810] |
| 17_direction2_neural_all_population_rescale | hard | -0.092352 | [-0.232246, 0.003619] |
| 17_direction2_neural_all_selected_risk_grid | all | 0.273819 | [-0.223998, 0.790375] |
| 17_direction2_neural_all_selected_risk_grid | hard | -0.037976 | [-0.580069, 0.573884] |
| 17_direction2_neural_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_neural_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_neural_easy_population_rescale | all | -0.143036 | [-0.225680, -0.074155] |
| 17_direction2_neural_easy_population_rescale | hard | -0.093919 | [-0.166719, -0.034262] |
| 17_direction2_neural_easy_selected_risk_grid | all | -0.034381 | [-0.056199, -0.014104] |
| 17_direction2_neural_easy_selected_risk_grid | hard | -0.028558 | [-0.050563, -0.008687] |
| 17_direction1_damping097_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_damping097_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_damping097_all_population_rescale | all | -0.573802 | [-0.936374, -0.246973] |
| 17_direction1_damping097_all_population_rescale | hard | -0.677411 | [-1.046941, -0.340117] |
| 17_direction1_damping097_all_selected_risk_grid | all | -0.251547 | [-0.429150, -0.092066] |
| 17_direction1_damping097_all_selected_risk_grid | hard | -0.260831 | [-0.435096, -0.107324] |
| 17_direction1_damping097_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_damping097_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction1_damping097_easy_population_rescale | all | -0.398309 | [-0.725706, -0.133535] |
| 17_direction1_damping097_easy_population_rescale | hard | -0.138066 | [-0.221836, -0.066939] |
| 17_direction1_damping097_easy_selected_risk_grid | all | -0.230641 | [-0.505622, -0.032629] |
| 17_direction1_damping097_easy_selected_risk_grid | hard | -0.209133 | [-0.453306, -0.029793] |
| 17_direction2_damping097_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_damping097_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_damping097_all_population_rescale | all | -0.775562 | [-1.349578, -0.258344] |
| 17_direction2_damping097_all_population_rescale | hard | -0.832119 | [-1.401437, -0.304658] |
| 17_direction2_damping097_all_selected_risk_grid | all | -0.593020 | [-1.109768, -0.136467] |
| 17_direction2_damping097_all_selected_risk_grid | hard | -0.606965 | [-1.125486, -0.148079] |
| 17_direction2_damping097_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_damping097_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 17_direction2_damping097_easy_population_rescale | all | -0.297018 | [-0.534045, -0.126393] |
| 17_direction2_damping097_easy_population_rescale | hard | -0.142552 | [-0.287661, -0.044912] |
| 17_direction2_damping097_easy_selected_risk_grid | all | -0.021267 | [-0.045007, -0.000449] |
| 17_direction2_damping097_easy_selected_risk_grid | hard | -0.023197 | [-0.048942, 0.000000] |
| 29_direction1_neural_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_neural_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_neural_all_population_rescale | all | -0.146022 | [-0.465452, 0.146407] |
| 29_direction1_neural_all_population_rescale | hard | -0.304182 | [-0.839052, 0.172734] |
| 29_direction1_neural_all_selected_risk_grid | all | -0.525858 | [-1.179159, 0.043638] |
| 29_direction1_neural_all_selected_risk_grid | hard | -0.988020 | [-2.129205, 0.000000] |
| 29_direction1_neural_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_neural_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_neural_easy_population_rescale | all | -0.189259 | [-0.293824, -0.101236] |
| 29_direction1_neural_easy_population_rescale | hard | -0.071743 | [-0.121398, -0.029801] |
| 29_direction1_neural_easy_selected_risk_grid | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_neural_easy_selected_risk_grid | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_neural_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_neural_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_neural_all_population_rescale | all | -0.107039 | [-0.284884, 0.077072] |
| 29_direction2_neural_all_population_rescale | hard | -0.231059 | [-0.345650, -0.125895] |
| 29_direction2_neural_all_selected_risk_grid | all | 0.039889 | [-0.270416, 0.378644] |
| 29_direction2_neural_all_selected_risk_grid | hard | -0.267244 | [-0.578957, -0.002322] |
| 29_direction2_neural_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_neural_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_neural_easy_population_rescale | all | -0.183938 | [-0.337638, -0.065633] |
| 29_direction2_neural_easy_population_rescale | hard | -0.129252 | [-0.230896, -0.046858] |
| 29_direction2_neural_easy_selected_risk_grid | all | -0.092846 | [-0.167753, -0.032181] |
| 29_direction2_neural_easy_selected_risk_grid | hard | -0.087006 | [-0.155560, -0.027148] |
| 29_direction1_damping097_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_damping097_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_damping097_all_population_rescale | all | -0.601416 | [-1.039875, -0.221237] |
| 29_direction1_damping097_all_population_rescale | hard | -0.654854 | [-1.107020, -0.258732] |
| 29_direction1_damping097_all_selected_risk_grid | all | -0.189474 | [-0.312357, -0.082385] |
| 29_direction1_damping097_all_selected_risk_grid | hard | -0.198214 | [-0.336327, -0.076106] |
| 29_direction1_damping097_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_damping097_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction1_damping097_easy_population_rescale | all | -0.376863 | [-0.725349, -0.093799] |
| 29_direction1_damping097_easy_population_rescale | hard | -0.137450 | [-0.254423, -0.043487] |
| 29_direction1_damping097_easy_selected_risk_grid | all | -0.181415 | [-0.348030, -0.044357] |
| 29_direction1_damping097_easy_selected_risk_grid | hard | -0.177449 | [-0.329080, -0.046551] |
| 29_direction2_damping097_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_damping097_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_damping097_all_population_rescale | all | -0.894943 | [-1.581833, -0.282782] |
| 29_direction2_damping097_all_population_rescale | hard | -0.931323 | [-1.620754, -0.291625] |
| 29_direction2_damping097_all_selected_risk_grid | all | -0.650565 | [-1.230431, -0.142855] |
| 29_direction2_damping097_all_selected_risk_grid | hard | -0.665687 | [-1.236122, -0.153429] |
| 29_direction2_damping097_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_damping097_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_damping097_easy_population_rescale | all | -0.241638 | [-0.482336, -0.053864] |
| 29_direction2_damping097_easy_population_rescale | hard | -0.113467 | [-0.244016, -0.026233] |
| 29_direction2_damping097_easy_selected_risk_grid | all | 0.000000 | [0.000000, 0.000000] |
| 29_direction2_damping097_easy_selected_risk_grid | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_neural_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_neural_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_neural_all_population_rescale | all | -0.515479 | [-0.998466, -0.102188] |
| 43_direction1_neural_all_population_rescale | hard | -0.771138 | [-1.529882, -0.133945] |
| 43_direction1_neural_all_selected_risk_grid | all | -1.321831 | [-2.723876, -0.211993] |
| 43_direction1_neural_all_selected_risk_grid | hard | -2.205693 | [-4.528640, -0.384893] |
| 43_direction1_neural_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_neural_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_neural_easy_population_rescale | all | -0.388361 | [-0.654259, -0.162864] |
| 43_direction1_neural_easy_population_rescale | hard | -0.149445 | [-0.240899, -0.066956] |
| 43_direction1_neural_easy_selected_risk_grid | all | -0.263855 | [-0.530892, -0.050093] |
| 43_direction1_neural_easy_selected_risk_grid | hard | -0.082705 | [-0.171632, -0.003202] |
| 43_direction2_neural_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_neural_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_neural_all_population_rescale | all | -0.130130 | [-0.351532, 0.090847] |
| 43_direction2_neural_all_population_rescale | hard | -0.271025 | [-0.498707, -0.091603] |
| 43_direction2_neural_all_selected_risk_grid | all | -0.003870 | [-0.706753, 0.604171] |
| 43_direction2_neural_all_selected_risk_grid | hard | -0.545248 | [-1.322310, -0.000102] |
| 43_direction2_neural_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_neural_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_neural_easy_population_rescale | all | -0.199210 | [-0.374057, -0.057328] |
| 43_direction2_neural_easy_population_rescale | hard | -0.103639 | [-0.210482, -0.021441] |
| 43_direction2_neural_easy_selected_risk_grid | all | -0.006298 | [-0.013264, -0.000912] |
| 43_direction2_neural_easy_selected_risk_grid | hard | -0.004725 | [-0.013185, 0.000000] |
| 43_direction1_damping097_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_damping097_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_damping097_all_population_rescale | all | -0.418949 | [-0.781603, -0.118366] |
| 43_direction1_damping097_all_population_rescale | hard | -0.410127 | [-0.756199, -0.116580] |
| 43_direction1_damping097_all_selected_risk_grid | all | -0.359720 | [-0.695303, -0.084937] |
| 43_direction1_damping097_all_selected_risk_grid | hard | -0.352359 | [-0.680242, -0.064688] |
| 43_direction1_damping097_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_damping097_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction1_damping097_easy_population_rescale | all | -0.425197 | [-0.801767, -0.122064] |
| 43_direction1_damping097_easy_population_rescale | hard | -0.205955 | [-0.368630, -0.074059] |
| 43_direction1_damping097_easy_selected_risk_grid | all | -0.330316 | [-0.719544, -0.055482] |
| 43_direction1_damping097_easy_selected_risk_grid | hard | -0.318349 | [-0.660098, -0.057623] |
| 43_direction2_damping097_all_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_damping097_all_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_damping097_all_population_rescale | all | -0.615688 | [-1.081516, -0.210358] |
| 43_direction2_damping097_all_population_rescale | hard | -0.636666 | [-1.086185, -0.235613] |
| 43_direction2_damping097_all_selected_risk_grid | all | -0.231587 | [-0.372677, -0.095283] |
| 43_direction2_damping097_all_selected_risk_grid | hard | -0.255713 | [-0.391305, -0.121883] |
| 43_direction2_damping097_easy_none | all | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_damping097_easy_none | hard | 0.000000 | [0.000000, 0.000000] |
| 43_direction2_damping097_easy_population_rescale | all | -0.232575 | [-0.408982, -0.094317] |
| 43_direction2_damping097_easy_population_rescale | hard | -0.141278 | [-0.248244, -0.059914] |
| 43_direction2_damping097_easy_selected_risk_grid | all | -0.031119 | [-0.060950, -0.006249] |
| 43_direction2_damping097_easy_selected_risk_grid | hard | -0.032761 | [-0.068116, -0.004907] |

No joint optimization or independent risk certificate is claimed. A zero-switch result is fallback, not positive transfer.
Bootstrap resamples localities 3,000 times conditional on shared source data/models. The two rotations are not independent replications.
Unknown, negative and undefined results remain visible. Pixel obs8/pred12 rawstride12; not t50, seconds, metric, physical safety or human gold.
No Stage5C, SMC, true3D, foundation, deployment promotion or submission-readiness claim.
