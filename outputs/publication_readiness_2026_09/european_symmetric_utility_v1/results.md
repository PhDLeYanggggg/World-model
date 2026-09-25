# Symmetric Utility Results

18 newly trained Torch utility heads / 36,000 updates. All 72 risk heads and all trajectories frozen.
All 48 registered views retained; no post-readout selection. Source development only, not reserved evaluation.

## Full Population

| Seed/candidate/event/risk/guard | ADE gain vs CV (%) | Conditional CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harmed | Switch (%) |
|---|---:|---|---:|---:|---:|---:|---:|
| 17_neural_all_ridge_no_guard | 0.614643 | [0.257613, 1.050919] | 0.799861 | 0.821433 | 0.706620 | 0/4 | 4.715819 |
| 17_neural_all_ridge_source_zero_guard | 0.284229 | [0.090529, 0.522413] | 0.338124 | 0.293842 | 0.364058 | 0/4 | 0.790359 |
| 17_neural_all_neural_underharm4_no_guard | 0.147822 | [0.040383, 0.284177] | 0.227622 | 0.148466 | 0.042697 | 0/4 | 0.293132 |
| 17_neural_all_neural_underharm4_source_zero_guard | 0.016004 | [0.000096, 0.034310] | 0.022773 | 0.016444 | -0.000000 | 0/4 | 0.081199 |
| 17_neural_easy_ridge_no_guard | 0.306160 | [0.022303, 0.531804] | 0.479299 | 0.356428 | 0.021378 | 1/4 | 5.556653 |
| 17_neural_easy_ridge_source_zero_guard | 0.317049 | [0.155231, 0.504554] | 0.418103 | 0.393313 | -0.000000 | 0/4 | 1.231468 |
| 17_neural_easy_neural_underharm4_no_guard | 0.271936 | [0.151281, 0.402917] | 0.394580 | 0.272401 | 0.049585 | 0/4 | 2.266051 |
| 17_neural_easy_neural_underharm4_source_zero_guard | 0.235744 | [0.103819, 0.378976] | 0.333123 | 0.267880 | 0.005623 | 0/4 | 0.646771 |
| 17_damping097_all_ridge_no_guard | 1.795341 | [0.921602, 2.731372] | 2.867230 | 1.910761 | 1.557245 | 0/4 | 34.385160 |
| 17_damping097_all_ridge_source_zero_guard | 0.745799 | [0.300983, 1.250658] | 1.153074 | 0.761188 | 0.669123 | 0/4 | 8.230894 |
| 17_damping097_all_neural_underharm4_no_guard | 0.667648 | [0.254163, 1.201257] | 1.033069 | 0.849438 | 0.624949 | 0/4 | 5.322147 |
| 17_damping097_all_neural_underharm4_source_zero_guard | 0.237544 | [0.100458, 0.394486] | 0.359914 | 0.275340 | 0.036771 | 0/4 | 0.820769 |
| 17_damping097_easy_ridge_no_guard | 1.679480 | [1.178135, 2.186394] | 2.608061 | 1.858853 | 2.490134 | 0/4 | 43.269722 |
| 17_damping097_easy_ridge_source_zero_guard | 1.184321 | [0.558671, 1.846962] | 1.874736 | 1.174761 | -0.000000 | 0/4 | 15.144105 |
| 17_damping097_easy_neural_underharm4_no_guard | 2.090108 | [1.072038, 3.099647] | 3.385622 | 2.013211 | -0.249111 | 0/4 | 28.611558 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | 2.061365 | [1.025286, 3.089612] | 3.336756 | 2.010295 | -0.000000 | 0/4 | 23.239562 |
| 29_neural_all_ridge_no_guard | 0.536012 | [0.239498, 0.906554] | 0.678542 | 0.702577 | 0.864991 | 0/4 | 3.168019 |
| 29_neural_all_ridge_source_zero_guard | 0.269827 | [0.091078, 0.511058] | 0.339164 | 0.285503 | 0.359525 | 0/4 | 0.616674 |
| 29_neural_all_neural_underharm4_no_guard | 0.046790 | [0.019851, 0.079773] | 0.072688 | 0.036070 | 0.001165 | 0/4 | 0.092485 |
| 29_neural_all_neural_underharm4_source_zero_guard | 0.017877 | [0.002758, 0.036321] | 0.028107 | 0.020089 | -0.000000 | 0/4 | 0.059880 |
| 29_neural_easy_ridge_no_guard | 0.354291 | [0.187834, 0.508931] | 0.500665 | 0.391013 | -0.000000 | 1/4 | 2.396471 |
| 29_neural_easy_ridge_source_zero_guard | 0.313034 | [0.158034, 0.471071] | 0.408246 | 0.383731 | -0.000000 | 0/4 | 0.908552 |
| 29_neural_easy_neural_underharm4_no_guard | 0.198183 | [0.086875, 0.323796] | 0.272648 | 0.215181 | -0.000000 | 0/4 | 0.441109 |
| 29_neural_easy_neural_underharm4_source_zero_guard | 0.197367 | [0.085767, 0.323301] | 0.271050 | 0.215183 | -0.000000 | 0/4 | 0.418536 |
| 29_damping097_all_ridge_no_guard | 1.787827 | [0.913550, 2.723597] | 2.857729 | 1.907625 | 1.557513 | 0/4 | 33.445256 |
| 29_damping097_all_ridge_source_zero_guard | 0.742949 | [0.298167, 1.246077] | 1.148721 | 0.762390 | 0.747042 | 0/4 | 7.739310 |
| 29_damping097_all_neural_underharm4_no_guard | 0.606929 | [0.252603, 1.021558] | 0.944601 | 0.761456 | 0.260306 | 0/4 | 3.803191 |
| 29_damping097_all_neural_underharm4_source_zero_guard | 0.258422 | [0.063620, 0.504337] | 0.406512 | 0.318303 | 0.051117 | 0/4 | 1.423022 |
| 29_damping097_easy_ridge_no_guard | 1.691829 | [1.203251, 2.180982] | 2.636548 | 1.773100 | 2.743480 | 0/4 | 40.047465 |
| 29_damping097_easy_ridge_source_zero_guard | 1.177475 | [0.560219, 1.833193] | 1.864692 | 1.191775 | -0.000000 | 0/4 | 13.567776 |
| 29_damping097_easy_neural_underharm4_no_guard | 2.197430 | [1.179905, 3.215424] | 3.571534 | 2.162102 | -0.238087 | 0/4 | 30.391355 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | 2.145306 | [1.094210, 3.202890] | 3.483817 | 2.152565 | -0.000000 | 0/4 | 21.265076 |
| 43_neural_all_ridge_no_guard | 0.612419 | [0.278765, 1.076380] | 0.826258 | 0.748689 | 0.645820 | 0/4 | 3.468049 |
| 43_neural_all_ridge_source_zero_guard | 0.231117 | [0.091109, 0.409638] | 0.309132 | 0.248583 | 0.096005 | 0/4 | 0.692230 |
| 43_neural_all_neural_underharm4_no_guard | 0.107540 | [0.032440, 0.195316] | 0.164979 | 0.094504 | 0.005303 | 0/4 | 0.199392 |
| 43_neural_all_neural_underharm4_source_zero_guard | 0.016701 | [0.004871, 0.031215] | 0.022755 | 0.011881 | -0.000000 | 0/4 | 0.063956 |
| 43_neural_easy_ridge_no_guard | 0.425078 | [0.222930, 0.633949] | 0.650229 | 0.472027 | -0.000000 | 0/4 | 3.629820 |
| 43_neural_easy_ridge_source_zero_guard | 0.385866 | [0.185142, 0.610147] | 0.560192 | 0.471993 | -0.000000 | 0/4 | 0.975643 |
| 43_neural_easy_neural_underharm4_no_guard | 0.418756 | [0.155332, 0.717351] | 0.564441 | 0.488659 | 0.688517 | 0/4 | 0.642696 |
| 43_neural_easy_neural_underharm4_source_zero_guard | 0.410981 | [0.149218, 0.713225] | 0.552063 | 0.488652 | 0.688517 | 0/4 | 0.599745 |
| 43_damping097_all_ridge_no_guard | 1.795469 | [0.920990, 2.727575] | 2.869113 | 1.910543 | 1.557885 | 0/4 | 34.421840 |
| 43_damping097_all_ridge_source_zero_guard | 0.746792 | [0.301050, 1.252742] | 1.154966 | 0.761211 | 0.705978 | 0/4 | 8.386708 |
| 43_damping097_all_neural_underharm4_no_guard | 0.608380 | [0.315441, 0.917433] | 0.973250 | 0.733477 | 0.269318 | 0/4 | 3.388104 |
| 43_damping097_all_neural_underharm4_source_zero_guard | 0.356359 | [0.131248, 0.631188] | 0.564121 | 0.411341 | 0.072316 | 0/4 | 1.242754 |
| 43_damping097_easy_ridge_no_guard | 1.733060 | [1.259870, 2.215188] | 2.686310 | 1.844276 | 2.787236 | 0/4 | 42.983801 |
| 43_damping097_easy_ridge_source_zero_guard | 1.202642 | [0.573103, 1.858810] | 1.893249 | 1.213341 | -0.000000 | 0/4 | 15.295217 |
| 43_damping097_easy_neural_underharm4_no_guard | 2.155857 | [1.170640, 3.135420] | 3.483287 | 2.103197 | -0.159926 | 0/4 | 27.360653 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | 2.117190 | [1.115331, 3.115337] | 3.416763 | 2.091541 | -0.000000 | 0/4 | 23.333302 |

## New Utility vs Frozen Old Utility

| View | Population/rule | Paired gain (%) | Conditional CI |
|---|---|---:|---|
| 17_neural_all_ridge_no_guard | pointwise | 0.008293 | [-0.018965, 0.033130] |
| 17_neural_all_ridge_no_guard | independent | 0.261263 | [-0.189810, 0.674281] |
| 17_neural_all_ridge_no_guard | scene_uniform | 0.071480 | [0.000664, 0.179766] |
| 17_neural_all_ridge_no_guard | joint | 0.241846 | [-0.225827, 0.676322] |
| 17_neural_all_ridge_no_guard | unary_exact | 0.256830 | [-0.196774, 0.677264] |
| 17_neural_all_ridge_no_guard | joint_exact | 0.245722 | [-0.208437, 0.676229] |
| 17_neural_all_ridge_no_guard | pointwise_hard | -0.004295 | [-0.048631, 0.029178] |
| 17_neural_all_ridge_no_guard | pointwise_easy | 0.057649 | [-0.036719, 0.175895] |
| 17_neural_all_ridge_source_zero_guard | pointwise | 0.012404 | [-0.011102, 0.034880] |
| 17_neural_all_ridge_source_zero_guard | independent | 0.392786 | [0.121032, 0.714232] |
| 17_neural_all_ridge_source_zero_guard | scene_uniform | 0.021378 | [0.000190, 0.051474] |
| 17_neural_all_ridge_source_zero_guard | joint | 0.397723 | [0.117131, 0.731313] |
| 17_neural_all_ridge_source_zero_guard | unary_exact | 0.386394 | [0.107067, 0.716357] |
| 17_neural_all_ridge_source_zero_guard | joint_exact | 0.386394 | [0.107067, 0.716357] |
| 17_neural_all_ridge_source_zero_guard | pointwise_hard | -0.001727 | [-0.044275, 0.029691] |
| 17_neural_all_ridge_source_zero_guard | pointwise_easy | 0.029149 | [-0.056230, 0.152086] |
| 17_neural_all_neural_underharm4_no_guard | pointwise | 0.000003 | [0.000000, 0.000008] |
| 17_neural_all_neural_underharm4_no_guard | independent | 0.010399 | [-0.229001, 0.192410] |
| 17_neural_all_neural_underharm4_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_neural_underharm4_no_guard | joint | 0.004397 | [-0.237545, 0.184618] |
| 17_neural_all_neural_underharm4_no_guard | unary_exact | 0.005610 | [-0.235047, 0.184912] |
| 17_neural_all_neural_underharm4_no_guard | joint_exact | 0.005110 | [-0.236561, 0.184912] |
| 17_neural_all_neural_underharm4_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_neural_underharm4_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_neural_underharm4_source_zero_guard | pointwise | 0.000003 | [0.000000, 0.000008] |
| 17_neural_all_neural_underharm4_source_zero_guard | independent | 0.107323 | [0.040642, 0.184883] |
| 17_neural_all_neural_underharm4_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_neural_underharm4_source_zero_guard | joint | 0.108472 | [0.040298, 0.187652] |
| 17_neural_all_neural_underharm4_source_zero_guard | unary_exact | 0.108472 | [0.040298, 0.187652] |
| 17_neural_all_neural_underharm4_source_zero_guard | joint_exact | 0.108472 | [0.040298, 0.187652] |
| 17_neural_all_neural_underharm4_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | pointwise | 0.028858 | [-0.101114, 0.118413] |
| 17_neural_easy_ridge_no_guard | independent | 0.218974 | [-0.225450, 0.653879] |
| 17_neural_easy_ridge_no_guard | scene_uniform | 0.006772 | [0.000000, 0.018581] |
| 17_neural_easy_ridge_no_guard | joint | 0.259162 | [-0.170747, 0.689926] |
| 17_neural_easy_ridge_no_guard | unary_exact | 0.209365 | [-0.238068, 0.644892] |
| 17_neural_easy_ridge_no_guard | joint_exact | 0.240070 | [-0.195417, 0.671598] |
| 17_neural_easy_ridge_no_guard | pointwise_hard | 0.044452 | [-0.089600, 0.146285] |
| 17_neural_easy_ridge_no_guard | pointwise_easy | 0.138594 | [0.006394, 0.343543] |
| 17_neural_easy_ridge_source_zero_guard | pointwise | 0.077231 | [0.033020, 0.122644] |
| 17_neural_easy_ridge_source_zero_guard | independent | 0.393148 | [0.102326, 0.732788] |
| 17_neural_easy_ridge_source_zero_guard | scene_uniform | 0.000868 | [0.000000, 0.002462] |
| 17_neural_easy_ridge_source_zero_guard | joint | 0.385337 | [0.095977, 0.726184] |
| 17_neural_easy_ridge_source_zero_guard | unary_exact | 0.385337 | [0.095977, 0.726184] |
| 17_neural_easy_ridge_source_zero_guard | joint_exact | 0.385337 | [0.095977, 0.726184] |
| 17_neural_easy_ridge_source_zero_guard | pointwise_hard | 0.097905 | [0.043052, 0.156188] |
| 17_neural_easy_ridge_source_zero_guard | pointwise_easy | 0.003001 | [-0.010366, 0.017122] |
| 17_neural_easy_neural_underharm4_no_guard | pointwise | 0.033856 | [0.018229, 0.051797] |
| 17_neural_easy_neural_underharm4_no_guard | independent | 0.111754 | [-0.152632, 0.411007] |
| 17_neural_easy_neural_underharm4_no_guard | scene_uniform | 0.006509 | [-0.006261, 0.020461] |
| 17_neural_easy_neural_underharm4_no_guard | joint | 0.110763 | [-0.154169, 0.410779] |
| 17_neural_easy_neural_underharm4_no_guard | unary_exact | 0.110763 | [-0.154169, 0.410779] |
| 17_neural_easy_neural_underharm4_no_guard | joint_exact | 0.110763 | [-0.154169, 0.410779] |
| 17_neural_easy_neural_underharm4_no_guard | pointwise_hard | 0.025120 | [0.007377, 0.048159] |
| 17_neural_easy_neural_underharm4_no_guard | pointwise_easy | 0.053278 | [-0.031950, 0.164966] |
| 17_neural_easy_neural_underharm4_source_zero_guard | pointwise | 0.021787 | [0.009557, 0.035292] |
| 17_neural_easy_neural_underharm4_source_zero_guard | independent | 0.186778 | [-0.005492, 0.459887] |
| 17_neural_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.006509 | [-0.006261, 0.020461] |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint | 0.186778 | [-0.005492, 0.459887] |
| 17_neural_easy_neural_underharm4_source_zero_guard | unary_exact | 0.186778 | [-0.005492, 0.459887] |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint_exact | 0.186778 | [-0.005492, 0.459887] |
| 17_neural_easy_neural_underharm4_source_zero_guard | pointwise_hard | 0.023490 | [0.005642, 0.046840] |
| 17_neural_easy_neural_underharm4_source_zero_guard | pointwise_easy | -0.004560 | [-0.040096, 0.029294] |
| 17_damping097_all_ridge_no_guard | pointwise | 0.026205 | [0.000679, 0.061263] |
| 17_damping097_all_ridge_no_guard | independent | 0.374531 | [-0.013591, 0.744494] |
| 17_damping097_all_ridge_no_guard | scene_uniform | 0.205213 | [0.000000, 0.430443] |
| 17_damping097_all_ridge_no_guard | joint | 0.336380 | [-0.041097, 0.695257] |
| 17_damping097_all_ridge_no_guard | unary_exact | 0.335439 | [-0.038182, 0.679769] |
| 17_damping097_all_ridge_no_guard | joint_exact | 0.353679 | [-0.019806, 0.709793] |
| 17_damping097_all_ridge_no_guard | pointwise_hard | 0.017582 | [-0.008287, 0.049673] |
| 17_damping097_all_ridge_no_guard | pointwise_easy | 0.092960 | [0.026920, 0.170198] |
| 17_damping097_all_ridge_source_zero_guard | pointwise | 0.011641 | [0.000711, 0.025567] |
| 17_damping097_all_ridge_source_zero_guard | independent | 0.424226 | [0.115402, 0.744894] |
| 17_damping097_all_ridge_source_zero_guard | scene_uniform | -0.010729 | [-0.032186, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | joint | 0.386807 | [0.078995, 0.687210] |
| 17_damping097_all_ridge_source_zero_guard | unary_exact | 0.387303 | [0.094899, 0.670528] |
| 17_damping097_all_ridge_source_zero_guard | joint_exact | 0.403421 | [0.100700, 0.706839] |
| 17_damping097_all_ridge_source_zero_guard | pointwise_hard | 0.008255 | [-0.000054, 0.018451] |
| 17_damping097_all_ridge_source_zero_guard | pointwise_easy | 0.030491 | [-0.002115, 0.077234] |
| 17_damping097_all_neural_underharm4_no_guard | pointwise | -0.047305 | [-0.133359, 0.001157] |
| 17_damping097_all_neural_underharm4_no_guard | independent | -0.053387 | [-0.283418, 0.137512] |
| 17_damping097_all_neural_underharm4_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_neural_underharm4_no_guard | joint | -0.076812 | [-0.298668, 0.112427] |
| 17_damping097_all_neural_underharm4_no_guard | unary_exact | -0.064755 | [-0.293269, 0.128414] |
| 17_damping097_all_neural_underharm4_no_guard | joint_exact | -0.066401 | [-0.292098, 0.122605] |
| 17_damping097_all_neural_underharm4_no_guard | pointwise_hard | -0.032443 | [-0.082228, 0.001062] |
| 17_damping097_all_neural_underharm4_no_guard | pointwise_easy | -0.014938 | [-0.047131, 0.002318] |
| 17_damping097_all_neural_underharm4_source_zero_guard | pointwise | 0.000219 | [0.000000, 0.000620] |
| 17_damping097_all_neural_underharm4_source_zero_guard | independent | 0.079523 | [0.014265, 0.151884] |
| 17_damping097_all_neural_underharm4_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint | 0.057214 | [-0.011504, 0.133639] |
| 17_damping097_all_neural_underharm4_source_zero_guard | unary_exact | 0.068443 | [0.004837, 0.143171] |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint_exact | 0.067838 | [0.005505, 0.140041] |
| 17_damping097_all_neural_underharm4_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | pointwise | 0.024081 | [-0.130903, 0.169720] |
| 17_damping097_easy_ridge_no_guard | independent | 0.210178 | [-0.328392, 0.667689] |
| 17_damping097_easy_ridge_no_guard | scene_uniform | 0.361936 | [0.210925, 0.538195] |
| 17_damping097_easy_ridge_no_guard | joint | 0.196268 | [-0.335411, 0.649293] |
| 17_damping097_easy_ridge_no_guard | unary_exact | 0.206597 | [-0.337131, 0.670124] |
| 17_damping097_easy_ridge_no_guard | joint_exact | 0.210705 | [-0.329316, 0.669434] |
| 17_damping097_easy_ridge_no_guard | pointwise_hard | 0.129727 | [-0.091622, 0.385598] |
| 17_damping097_easy_ridge_no_guard | pointwise_easy | 0.621104 | [0.442584, 0.803071] |
| 17_damping097_easy_ridge_source_zero_guard | pointwise | 0.120102 | [0.036769, 0.203023] |
| 17_damping097_easy_ridge_source_zero_guard | independent | 0.417383 | [0.128226, 0.691939] |
| 17_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.235180 | [0.075151, 0.437861] |
| 17_damping097_easy_ridge_source_zero_guard | joint | 0.402683 | [0.115357, 0.670921] |
| 17_damping097_easy_ridge_source_zero_guard | unary_exact | 0.419194 | [0.129508, 0.693764] |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact | 0.418330 | [0.128795, 0.693330] |
| 17_damping097_easy_ridge_source_zero_guard | pointwise_hard | 0.085920 | [0.016955, 0.153115] |
| 17_damping097_easy_ridge_source_zero_guard | pointwise_easy | 0.465731 | [0.222208, 0.722367] |
| 17_damping097_easy_neural_underharm4_no_guard | pointwise | 0.288494 | [0.045807, 0.527687] |
| 17_damping097_easy_neural_underharm4_no_guard | independent | 0.423111 | [0.024960, 0.816461] |
| 17_damping097_easy_neural_underharm4_no_guard | scene_uniform | 0.876636 | [0.408843, 1.428249] |
| 17_damping097_easy_neural_underharm4_no_guard | joint | 0.408962 | [0.011047, 0.805961] |
| 17_damping097_easy_neural_underharm4_no_guard | unary_exact | 0.429094 | [0.038764, 0.820889] |
| 17_damping097_easy_neural_underharm4_no_guard | joint_exact | 0.423592 | [0.029319, 0.819859] |
| 17_damping097_easy_neural_underharm4_no_guard | pointwise_hard | 0.249487 | [0.013050, 0.466578] |
| 17_damping097_easy_neural_underharm4_no_guard | pointwise_easy | 0.835808 | [0.369239, 1.359267] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | pointwise | 0.287796 | [0.043766, 0.527615] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | independent | 0.473817 | [0.119054, 0.838884] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.876636 | [0.408843, 1.428249] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint | 0.458716 | [0.099645, 0.828282] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | unary_exact | 0.473818 | [0.119055, 0.838884] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint_exact | 0.473346 | [0.118077, 0.838412] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | pointwise_hard | 0.249586 | [0.013147, 0.466702] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | pointwise_easy | 0.828122 | [0.357698, 1.350745] |
| 29_neural_all_ridge_no_guard | pointwise | 0.056346 | [0.019861, 0.114324] |
| 29_neural_all_ridge_no_guard | independent | 0.030971 | [-0.390862, 0.435930] |
| 29_neural_all_ridge_no_guard | scene_uniform | 0.095276 | [-0.048839, 0.282636] |
| 29_neural_all_ridge_no_guard | joint | 0.045473 | [-0.394424, 0.459576] |
| 29_neural_all_ridge_no_guard | unary_exact | 0.030413 | [-0.374280, 0.422186] |
| 29_neural_all_ridge_no_guard | joint_exact | 0.044639 | [-0.394424, 0.458509] |
| 29_neural_all_ridge_no_guard | pointwise_hard | 0.047769 | [0.013492, 0.104252] |
| 29_neural_all_ridge_no_guard | pointwise_easy | 0.105146 | [-0.006364, 0.238542] |
| 29_neural_all_ridge_source_zero_guard | pointwise | 0.052892 | [0.015762, 0.111861] |
| 29_neural_all_ridge_source_zero_guard | independent | 0.172206 | [-0.153503, 0.474747] |
| 29_neural_all_ridge_source_zero_guard | scene_uniform | 0.024046 | [-0.076974, 0.146190] |
| 29_neural_all_ridge_source_zero_guard | joint | 0.157813 | [-0.181164, 0.456731] |
| 29_neural_all_ridge_source_zero_guard | unary_exact | 0.157813 | [-0.181164, 0.456731] |
| 29_neural_all_ridge_source_zero_guard | joint_exact | 0.157813 | [-0.181164, 0.456731] |
| 29_neural_all_ridge_source_zero_guard | pointwise_hard | 0.043169 | [0.008152, 0.100869] |
| 29_neural_all_ridge_source_zero_guard | pointwise_easy | 0.078979 | [-0.033988, 0.214173] |
| 29_neural_all_neural_underharm4_no_guard | pointwise | -0.000043 | [-0.000167, 0.000038] |
| 29_neural_all_neural_underharm4_no_guard | independent | 0.083526 | [-0.116023, 0.314613] |
| 29_neural_all_neural_underharm4_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_underharm4_no_guard | joint | 0.065098 | [-0.134392, 0.296574] |
| 29_neural_all_neural_underharm4_no_guard | unary_exact | 0.076292 | [-0.122924, 0.305290] |
| 29_neural_all_neural_underharm4_no_guard | joint_exact | 0.064896 | [-0.134952, 0.296574] |
| 29_neural_all_neural_underharm4_no_guard | pointwise_hard | 0.000019 | [0.000000, 0.000058] |
| 29_neural_all_neural_underharm4_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_underharm4_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_underharm4_source_zero_guard | independent | -0.019844 | [-0.159454, 0.087058] |
| 29_neural_all_neural_underharm4_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_underharm4_source_zero_guard | joint | -0.019844 | [-0.159454, 0.087058] |
| 29_neural_all_neural_underharm4_source_zero_guard | unary_exact | -0.019844 | [-0.159454, 0.087058] |
| 29_neural_all_neural_underharm4_source_zero_guard | joint_exact | -0.019844 | [-0.159454, 0.087058] |
| 29_neural_all_neural_underharm4_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | pointwise | 0.035646 | [-0.043238, 0.091353] |
| 29_neural_easy_ridge_no_guard | independent | 0.177678 | [-0.241347, 0.591279] |
| 29_neural_easy_ridge_no_guard | scene_uniform | 0.014931 | [0.000000, 0.041611] |
| 29_neural_easy_ridge_no_guard | joint | 0.155030 | [-0.270997, 0.573751] |
| 29_neural_easy_ridge_no_guard | unary_exact | 0.174726 | [-0.231519, 0.585638] |
| 29_neural_easy_ridge_no_guard | joint_exact | 0.163744 | [-0.256345, 0.578646] |
| 29_neural_easy_ridge_no_guard | pointwise_hard | 0.033805 | [-0.068401, 0.111257] |
| 29_neural_easy_ridge_no_guard | pointwise_easy | 0.154805 | [0.053997, 0.279756] |
| 29_neural_easy_ridge_source_zero_guard | pointwise | 0.048030 | [0.014690, 0.082340] |
| 29_neural_easy_ridge_source_zero_guard | independent | 0.346939 | [0.086905, 0.651136] |
| 29_neural_easy_ridge_source_zero_guard | scene_uniform | 0.001591 | [0.000000, 0.003831] |
| 29_neural_easy_ridge_source_zero_guard | joint | 0.342434 | [0.077462, 0.651432] |
| 29_neural_easy_ridge_source_zero_guard | unary_exact | 0.339242 | [0.076953, 0.647621] |
| 29_neural_easy_ridge_source_zero_guard | joint_exact | 0.339242 | [0.076953, 0.647621] |
| 29_neural_easy_ridge_source_zero_guard | pointwise_hard | 0.064692 | [0.022283, 0.113783] |
| 29_neural_easy_ridge_source_zero_guard | pointwise_easy | 0.027667 | [0.000047, 0.057212] |
| 29_neural_easy_neural_underharm4_no_guard | pointwise | 0.031933 | [0.012508, 0.059400] |
| 29_neural_easy_neural_underharm4_no_guard | independent | 0.106340 | [-0.013700, 0.278806] |
| 29_neural_easy_neural_underharm4_no_guard | scene_uniform | 0.013814 | [0.000912, 0.036628] |
| 29_neural_easy_neural_underharm4_no_guard | joint | 0.103323 | [-0.019730, 0.277840] |
| 29_neural_easy_neural_underharm4_no_guard | unary_exact | 0.106340 | [-0.013700, 0.278806] |
| 29_neural_easy_neural_underharm4_no_guard | joint_exact | 0.106340 | [-0.013700, 0.278806] |
| 29_neural_easy_neural_underharm4_no_guard | pointwise_hard | 0.026659 | [0.002951, 0.061121] |
| 29_neural_easy_neural_underharm4_no_guard | pointwise_easy | 0.074945 | [0.011542, 0.158101] |
| 29_neural_easy_neural_underharm4_source_zero_guard | pointwise | 0.031331 | [0.011674, 0.058941] |
| 29_neural_easy_neural_underharm4_source_zero_guard | independent | 0.128844 | [0.020144, 0.293491] |
| 29_neural_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.013814 | [0.000912, 0.036628] |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint | 0.128844 | [0.020144, 0.293491] |
| 29_neural_easy_neural_underharm4_source_zero_guard | unary_exact | 0.128844 | [0.020144, 0.293491] |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint_exact | 0.128844 | [0.020144, 0.293491] |
| 29_neural_easy_neural_underharm4_source_zero_guard | pointwise_hard | 0.026661 | [0.002954, 0.061122] |
| 29_neural_easy_neural_underharm4_source_zero_guard | pointwise_easy | 0.069593 | [0.004871, 0.153789] |
| 29_damping097_all_ridge_no_guard | pointwise | 0.056353 | [0.006763, 0.141822] |
| 29_damping097_all_ridge_no_guard | independent | 0.231091 | [-0.086766, 0.516976] |
| 29_damping097_all_ridge_no_guard | scene_uniform | 0.168418 | [0.000000, 0.372181] |
| 29_damping097_all_ridge_no_guard | joint | 0.207875 | [-0.103935, 0.484364] |
| 29_damping097_all_ridge_no_guard | unary_exact | 0.204970 | [-0.100160, 0.481294] |
| 29_damping097_all_ridge_no_guard | joint_exact | 0.203895 | [-0.102386, 0.481511] |
| 29_damping097_all_ridge_no_guard | pointwise_hard | 0.033558 | [-0.001813, 0.088306] |
| 29_damping097_all_ridge_no_guard | pointwise_easy | 0.174751 | [0.031697, 0.409854] |
| 29_damping097_all_ridge_source_zero_guard | pointwise | 0.012767 | [0.001463, 0.028192] |
| 29_damping097_all_ridge_source_zero_guard | independent | 0.229933 | [-0.081427, 0.499227] |
| 29_damping097_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | joint | 0.208346 | [-0.095781, 0.467512] |
| 29_damping097_all_ridge_source_zero_guard | unary_exact | 0.203568 | [-0.099365, 0.460910] |
| 29_damping097_all_ridge_source_zero_guard | joint_exact | 0.202493 | [-0.100231, 0.460336] |
| 29_damping097_all_ridge_source_zero_guard | pointwise_hard | 0.009773 | [0.001465, 0.020050] |
| 29_damping097_all_ridge_source_zero_guard | pointwise_easy | 0.022854 | [0.001231, 0.051511] |
| 29_damping097_all_neural_underharm4_no_guard | pointwise | -0.006931 | [-0.020130, 0.000746] |
| 29_damping097_all_neural_underharm4_no_guard | independent | -0.036762 | [-0.184866, 0.097913] |
| 29_damping097_all_neural_underharm4_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_neural_underharm4_no_guard | joint | -0.042963 | [-0.193339, 0.094460] |
| 29_damping097_all_neural_underharm4_no_guard | unary_exact | -0.038532 | [-0.188916, 0.099227] |
| 29_damping097_all_neural_underharm4_no_guard | joint_exact | -0.041208 | [-0.190667, 0.095488] |
| 29_damping097_all_neural_underharm4_no_guard | pointwise_hard | -0.004225 | [-0.011556, 0.000992] |
| 29_damping097_all_neural_underharm4_no_guard | pointwise_easy | -0.000811 | [-0.002433, 0.000000] |
| 29_damping097_all_neural_underharm4_source_zero_guard | pointwise | 0.000373 | [0.000000, 0.001119] |
| 29_damping097_all_neural_underharm4_source_zero_guard | independent | 0.019338 | [-0.095706, 0.123137] |
| 29_damping097_all_neural_underharm4_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint | 0.016114 | [-0.104235, 0.123677] |
| 29_damping097_all_neural_underharm4_source_zero_guard | unary_exact | 0.017326 | [-0.103624, 0.124588] |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint_exact | 0.014649 | [-0.104257, 0.121103] |
| 29_damping097_all_neural_underharm4_source_zero_guard | pointwise_hard | 0.000496 | [0.000000, 0.001487] |
| 29_damping097_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | pointwise | 0.058342 | [-0.058115, 0.161758] |
| 29_damping097_easy_ridge_no_guard | independent | 0.175336 | [-0.261425, 0.525904] |
| 29_damping097_easy_ridge_no_guard | scene_uniform | 0.186887 | [0.083626, 0.306168] |
| 29_damping097_easy_ridge_no_guard | joint | 0.193767 | [-0.249766, 0.545742] |
| 29_damping097_easy_ridge_no_guard | unary_exact | 0.178450 | [-0.265818, 0.534625] |
| 29_damping097_easy_ridge_no_guard | joint_exact | 0.182712 | [-0.262675, 0.536474] |
| 29_damping097_easy_ridge_no_guard | pointwise_hard | 0.029190 | [-0.097711, 0.128963] |
| 29_damping097_easy_ridge_no_guard | pointwise_easy | 0.517650 | [0.309667, 0.740884] |
| 29_damping097_easy_ridge_source_zero_guard | pointwise | 0.093142 | [0.019036, 0.167587] |
| 29_damping097_easy_ridge_source_zero_guard | independent | 0.182589 | [-0.249377, 0.526157] |
| 29_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.135509 | [0.043712, 0.250991] |
| 29_damping097_easy_ridge_source_zero_guard | joint | 0.195523 | [-0.243548, 0.541067] |
| 29_damping097_easy_ridge_source_zero_guard | unary_exact | 0.189576 | [-0.247037, 0.536899] |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact | 0.192298 | [-0.245799, 0.538928] |
| 29_damping097_easy_ridge_source_zero_guard | pointwise_hard | 0.073111 | [0.009777, 0.133851] |
| 29_damping097_easy_ridge_source_zero_guard | pointwise_easy | 0.286258 | [0.109811, 0.501250] |
| 29_damping097_easy_neural_underharm4_no_guard | pointwise | 0.266854 | [0.034478, 0.503495] |
| 29_damping097_easy_neural_underharm4_no_guard | independent | 0.380214 | [0.018593, 0.747279] |
| 29_damping097_easy_neural_underharm4_no_guard | scene_uniform | 0.583711 | [0.257886, 0.944773] |
| 29_damping097_easy_neural_underharm4_no_guard | joint | 0.369208 | [0.003855, 0.739382] |
| 29_damping097_easy_neural_underharm4_no_guard | unary_exact | 0.383510 | [0.022709, 0.749838] |
| 29_damping097_easy_neural_underharm4_no_guard | joint_exact | 0.380571 | [0.019141, 0.747285] |
| 29_damping097_easy_neural_underharm4_no_guard | pointwise_hard | 0.245949 | [0.005950, 0.479614] |
| 29_damping097_easy_neural_underharm4_no_guard | pointwise_easy | 0.745893 | [0.356952, 1.220754] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | pointwise | 0.254360 | [0.016614, 0.494177] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | independent | 0.366649 | [-0.008553, 0.740650] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.583711 | [0.257886, 0.944773] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint | 0.367923 | [-0.007394, 0.741771] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | unary_exact | 0.366649 | [-0.008553, 0.740650] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint_exact | 0.366649 | [-0.008553, 0.740650] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | pointwise_hard | 0.243699 | [0.002757, 0.478291] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | pointwise_easy | 0.658832 | [0.228751, 1.166857] |
| 43_neural_all_ridge_no_guard | pointwise | 0.006054 | [-0.060303, 0.047513] |
| 43_neural_all_ridge_no_guard | independent | 0.248328 | [-0.154757, 0.637976] |
| 43_neural_all_ridge_no_guard | scene_uniform | 0.089525 | [0.002544, 0.238788] |
| 43_neural_all_ridge_no_guard | joint | 0.323916 | [-0.052269, 0.712137] |
| 43_neural_all_ridge_no_guard | unary_exact | 0.292108 | [-0.114838, 0.674578] |
| 43_neural_all_ridge_no_guard | joint_exact | 0.293783 | [-0.114062, 0.675043] |
| 43_neural_all_ridge_no_guard | pointwise_hard | -0.006692 | [-0.076271, 0.041400] |
| 43_neural_all_ridge_no_guard | pointwise_easy | 0.106925 | [0.031664, 0.189967] |
| 43_neural_all_ridge_source_zero_guard | pointwise | -0.011185 | [-0.073629, 0.029199] |
| 43_neural_all_ridge_source_zero_guard | independent | 0.371382 | [0.125365, 0.681666] |
| 43_neural_all_ridge_source_zero_guard | scene_uniform | 0.017818 | [-0.001712, 0.044662] |
| 43_neural_all_ridge_source_zero_guard | joint | 0.387240 | [0.129263, 0.714279] |
| 43_neural_all_ridge_source_zero_guard | unary_exact | 0.387240 | [0.129263, 0.714279] |
| 43_neural_all_ridge_source_zero_guard | joint_exact | 0.387240 | [0.129263, 0.714279] |
| 43_neural_all_ridge_source_zero_guard | pointwise_hard | -0.021472 | [-0.082279, 0.014880] |
| 43_neural_all_ridge_source_zero_guard | pointwise_easy | 0.061043 | [-0.008302, 0.146147] |
| 43_neural_all_neural_underharm4_no_guard | pointwise | 0.004275 | [0.000000, 0.010542] |
| 43_neural_all_neural_underharm4_no_guard | independent | 0.013448 | [-0.143478, 0.146191] |
| 43_neural_all_neural_underharm4_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_neural_underharm4_no_guard | joint | 0.027753 | [-0.130346, 0.157780] |
| 43_neural_all_neural_underharm4_no_guard | unary_exact | 0.028072 | [-0.129797, 0.157923] |
| 43_neural_all_neural_underharm4_no_guard | joint_exact | 0.028072 | [-0.129797, 0.157923] |
| 43_neural_all_neural_underharm4_no_guard | pointwise_hard | 0.001744 | [0.000000, 0.005232] |
| 43_neural_all_neural_underharm4_no_guard | pointwise_easy | 0.010132 | [0.000000, 0.024781] |
| 43_neural_all_neural_underharm4_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_neural_underharm4_source_zero_guard | independent | 0.070272 | [-0.000579, 0.155235] |
| 43_neural_all_neural_underharm4_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_neural_underharm4_source_zero_guard | joint | 0.078511 | [0.004373, 0.165259] |
| 43_neural_all_neural_underharm4_source_zero_guard | unary_exact | 0.078511 | [0.004373, 0.165259] |
| 43_neural_all_neural_underharm4_source_zero_guard | joint_exact | 0.078511 | [0.004373, 0.165259] |
| 43_neural_all_neural_underharm4_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | pointwise | 0.059250 | [0.009366, 0.101971] |
| 43_neural_easy_ridge_no_guard | independent | -0.156905 | [-0.678797, 0.349684] |
| 43_neural_easy_ridge_no_guard | scene_uniform | 0.013353 | [0.000000, 0.040059] |
| 43_neural_easy_ridge_no_guard | joint | -0.162937 | [-0.687552, 0.346374] |
| 43_neural_easy_ridge_no_guard | unary_exact | -0.146402 | [-0.652365, 0.351917] |
| 43_neural_easy_ridge_no_guard | joint_exact | -0.155878 | [-0.679093, 0.347484] |
| 43_neural_easy_ridge_no_guard | pointwise_hard | 0.044053 | [-0.015302, 0.098377] |
| 43_neural_easy_ridge_no_guard | pointwise_easy | 0.207272 | [0.004901, 0.427119] |
| 43_neural_easy_ridge_source_zero_guard | pointwise | 0.048457 | [0.014798, 0.084551] |
| 43_neural_easy_ridge_source_zero_guard | independent | 0.162419 | [-0.194200, 0.480393] |
| 43_neural_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | joint | 0.154483 | [-0.217270, 0.480152] |
| 43_neural_easy_ridge_source_zero_guard | unary_exact | 0.154483 | [-0.217270, 0.480152] |
| 43_neural_easy_ridge_source_zero_guard | joint_exact | 0.154483 | [-0.217270, 0.480152] |
| 43_neural_easy_ridge_source_zero_guard | pointwise_hard | 0.058885 | [0.017446, 0.102222] |
| 43_neural_easy_ridge_source_zero_guard | pointwise_easy | -0.004763 | [-0.018847, 0.003821] |
| 43_neural_easy_neural_underharm4_no_guard | pointwise | -0.012703 | [-0.078348, 0.035763] |
| 43_neural_easy_neural_underharm4_no_guard | independent | 0.149574 | [-0.001379, 0.325040] |
| 43_neural_easy_neural_underharm4_no_guard | scene_uniform | 0.000027 | [-0.020285, 0.021499] |
| 43_neural_easy_neural_underharm4_no_guard | joint | 0.148828 | [-0.001379, 0.323547] |
| 43_neural_easy_neural_underharm4_no_guard | unary_exact | 0.148828 | [-0.001379, 0.323547] |
| 43_neural_easy_neural_underharm4_no_guard | joint_exact | 0.148828 | [-0.001379, 0.323547] |
| 43_neural_easy_neural_underharm4_no_guard | pointwise_hard | -0.014463 | [-0.063120, 0.027158] |
| 43_neural_easy_neural_underharm4_no_guard | pointwise_easy | -0.007092 | [-0.082729, 0.058111] |
| 43_neural_easy_neural_underharm4_source_zero_guard | pointwise | -0.016974 | [-0.081700, 0.031919] |
| 43_neural_easy_neural_underharm4_source_zero_guard | independent | 0.070861 | [-0.050382, 0.243337] |
| 43_neural_easy_neural_underharm4_source_zero_guard | scene_uniform | -0.007405 | [-0.023356, 0.005053] |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint | 0.070861 | [-0.050382, 0.243337] |
| 43_neural_easy_neural_underharm4_source_zero_guard | unary_exact | 0.070861 | [-0.050382, 0.243337] |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint_exact | 0.070861 | [-0.050382, 0.243337] |
| 43_neural_easy_neural_underharm4_source_zero_guard | pointwise_hard | -0.014470 | [-0.063131, 0.027151] |
| 43_neural_easy_neural_underharm4_source_zero_guard | pointwise_easy | -0.030960 | [-0.094693, 0.018989] |
| 43_damping097_all_ridge_no_guard | pointwise | 0.053540 | [0.004044, 0.135062] |
| 43_damping097_all_ridge_no_guard | independent | 0.401078 | [0.074857, 0.721103] |
| 43_damping097_all_ridge_no_guard | scene_uniform | 0.262043 | [0.032065, 0.519700] |
| 43_damping097_all_ridge_no_guard | joint | 0.388522 | [0.073818, 0.694572] |
| 43_damping097_all_ridge_no_guard | unary_exact | 0.397673 | [0.068611, 0.722244] |
| 43_damping097_all_ridge_no_guard | joint_exact | 0.399152 | [0.075231, 0.722178] |
| 43_damping097_all_ridge_no_guard | pointwise_hard | 0.029466 | [-0.010935, 0.086352] |
| 43_damping097_all_ridge_no_guard | pointwise_easy | 0.195931 | [0.052233, 0.417442] |
| 43_damping097_all_ridge_source_zero_guard | pointwise | 0.012649 | [0.000804, 0.027794] |
| 43_damping097_all_ridge_source_zero_guard | independent | 0.367189 | [0.138251, 0.634070] |
| 43_damping097_all_ridge_source_zero_guard | scene_uniform | 0.015766 | [-0.000138, 0.041841] |
| 43_damping097_all_ridge_source_zero_guard | joint | 0.356866 | [0.136331, 0.611074] |
| 43_damping097_all_ridge_source_zero_guard | unary_exact | 0.365126 | [0.134905, 0.632787] |
| 43_damping097_all_ridge_source_zero_guard | joint_exact | 0.365065 | [0.134842, 0.632727] |
| 43_damping097_all_ridge_source_zero_guard | pointwise_hard | 0.006396 | [-0.001707, 0.016371] |
| 43_damping097_all_ridge_source_zero_guard | pointwise_easy | 0.043650 | [0.010630, 0.095692] |
| 43_damping097_all_neural_underharm4_no_guard | pointwise | -0.012588 | [-0.035637, 0.000921] |
| 43_damping097_all_neural_underharm4_no_guard | independent | 0.182556 | [0.044020, 0.315795] |
| 43_damping097_all_neural_underharm4_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_neural_underharm4_no_guard | joint | 0.185662 | [0.044067, 0.318218] |
| 43_damping097_all_neural_underharm4_no_guard | unary_exact | 0.183466 | [0.043172, 0.315902] |
| 43_damping097_all_neural_underharm4_no_guard | joint_exact | 0.184355 | [0.043797, 0.316502] |
| 43_damping097_all_neural_underharm4_no_guard | pointwise_hard | -0.007514 | [-0.021692, 0.001081] |
| 43_damping097_all_neural_underharm4_no_guard | pointwise_easy | -0.000547 | [-0.001642, 0.000000] |
| 43_damping097_all_neural_underharm4_source_zero_guard | pointwise | 0.000098 | [0.000000, 0.000256] |
| 43_damping097_all_neural_underharm4_source_zero_guard | independent | 0.153455 | [0.052931, 0.271120] |
| 43_damping097_all_neural_underharm4_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint | 0.154547 | [0.051521, 0.272872] |
| 43_damping097_all_neural_underharm4_source_zero_guard | unary_exact | 0.153900 | [0.051292, 0.271568] |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint_exact | 0.153900 | [0.051292, 0.271568] |
| 43_damping097_all_neural_underharm4_source_zero_guard | pointwise_hard | 0.000056 | [0.000000, 0.000169] |
| 43_damping097_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | pointwise | 0.186757 | [-0.004780, 0.396021] |
| 43_damping097_easy_ridge_no_guard | independent | 0.563623 | [0.220014, 0.864830] |
| 43_damping097_easy_ridge_no_guard | scene_uniform | 0.377470 | [0.224513, 0.550646] |
| 43_damping097_easy_ridge_no_guard | joint | 0.557681 | [0.211270, 0.859577] |
| 43_damping097_easy_ridge_no_guard | unary_exact | 0.547847 | [0.200975, 0.855032] |
| 43_damping097_easy_ridge_no_guard | joint_exact | 0.555479 | [0.209411, 0.862093] |
| 43_damping097_easy_ridge_no_guard | pointwise_hard | 0.113803 | [-0.071050, 0.280203] |
| 43_damping097_easy_ridge_no_guard | pointwise_easy | 0.666197 | [0.484549, 0.866433] |
| 43_damping097_easy_ridge_source_zero_guard | pointwise | 0.119228 | [0.045206, 0.197662] |
| 43_damping097_easy_ridge_source_zero_guard | independent | 0.486883 | [0.233685, 0.761407] |
| 43_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.164138 | [0.056697, 0.303735] |
| 43_damping097_easy_ridge_source_zero_guard | joint | 0.484425 | [0.231387, 0.755731] |
| 43_damping097_easy_ridge_source_zero_guard | unary_exact | 0.478374 | [0.223232, 0.752393] |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact | 0.481677 | [0.225544, 0.755668] |
| 43_damping097_easy_ridge_source_zero_guard | pointwise_hard | 0.097648 | [0.040036, 0.160936] |
| 43_damping097_easy_ridge_source_zero_guard | pointwise_easy | 0.343581 | [0.175883, 0.513851] |
| 43_damping097_easy_neural_underharm4_no_guard | pointwise | 0.266576 | [0.067954, 0.470060] |
| 43_damping097_easy_neural_underharm4_no_guard | independent | 0.626048 | [0.295830, 0.974720] |
| 43_damping097_easy_neural_underharm4_no_guard | scene_uniform | 0.635095 | [0.210288, 1.192050] |
| 43_damping097_easy_neural_underharm4_no_guard | joint | 0.633693 | [0.310084, 0.976324] |
| 43_damping097_easy_neural_underharm4_no_guard | unary_exact | 0.618491 | [0.288834, 0.969576] |
| 43_damping097_easy_neural_underharm4_no_guard | joint_exact | 0.618475 | [0.288802, 0.969638] |
| 43_damping097_easy_neural_underharm4_no_guard | pointwise_hard | 0.232154 | [0.041106, 0.418292] |
| 43_damping097_easy_neural_underharm4_no_guard | pointwise_easy | 0.827777 | [0.474971, 1.203884] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | pointwise | 0.259267 | [0.055745, 0.468282] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | independent | 0.604634 | [0.279721, 0.956493] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.635095 | [0.210288, 1.192050] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint | 0.599053 | [0.276039, 0.953723] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | unary_exact | 0.597112 | [0.274312, 0.952379] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint_exact | 0.597177 | [0.274319, 0.952507] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | pointwise_hard | 0.230698 | [0.038665, 0.416253] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | pointwise_easy | 0.765027 | [0.377135, 1.178700] |

## Neural vs Protected Damping

| View | Population/rule | Paired gain (%) | Conditional CI |
|---|---|---:|---|
| 17_all_ridge_no_guard | pointwise | -1.221564 | [-1.990686, -0.506932] |
| 17_all_ridge_no_guard | independent | -1.918509 | [-3.325502, -0.323073] |
| 17_all_ridge_no_guard | scene_uniform | -0.288270 | [-0.553590, -0.031120] |
| 17_all_ridge_no_guard | joint | -1.940498 | [-3.352804, -0.340167] |
| 17_all_ridge_no_guard | unary_exact | -1.913405 | [-3.338694, -0.300503] |
| 17_all_ridge_no_guard | joint_exact | -1.940085 | [-3.360148, -0.325470] |
| 17_all_ridge_no_guard | pointwise_hard | -1.127314 | [-1.732457, -0.540256] |
| 17_all_ridge_no_guard | pointwise_easy | -0.275029 | [-1.084224, 0.327195] |
| 17_all_ridge_source_zero_guard | pointwise | -0.471018 | [-0.918056, -0.073998] |
| 17_all_ridge_source_zero_guard | independent | -1.535000 | [-2.684393, -0.399906] |
| 17_all_ridge_source_zero_guard | scene_uniform | -0.055834 | [-0.203007, 0.079710] |
| 17_all_ridge_source_zero_guard | joint | -1.525817 | [-2.658010, -0.392263] |
| 17_all_ridge_source_zero_guard | unary_exact | -1.524068 | [-2.659884, -0.393138] |
| 17_all_ridge_source_zero_guard | joint_exact | -1.538467 | [-2.680604, -0.398149] |
| 17_all_ridge_source_zero_guard | pointwise_hard | -0.476548 | [-0.917795, -0.080517] |
| 17_all_ridge_source_zero_guard | pointwise_easy | 0.001312 | [-0.161943, 0.171939] |
| 17_all_neural_underharm4_no_guard | pointwise | -0.529276 | [-0.999279, -0.173580] |
| 17_all_neural_underharm4_no_guard | independent | -1.890453 | [-2.792837, -1.057603] |
| 17_all_neural_underharm4_no_guard | scene_uniform | -0.005777 | [-0.052764, 0.044022] |
| 17_all_neural_underharm4_no_guard | joint | -1.891132 | [-2.791807, -1.063687] |
| 17_all_neural_underharm4_no_guard | unary_exact | -1.891408 | [-2.794310, -1.063202] |
| 17_all_neural_underharm4_no_guard | joint_exact | -1.891578 | [-2.792702, -1.063687] |
| 17_all_neural_underharm4_no_guard | pointwise_hard | -0.715034 | [-1.233687, -0.321806] |
| 17_all_neural_underharm4_no_guard | pointwise_easy | 0.090966 | [0.012513, 0.196280] |
| 17_all_neural_underharm4_source_zero_guard | pointwise | -0.222783 | [-0.380136, -0.088964] |
| 17_all_neural_underharm4_source_zero_guard | independent | -1.312254 | [-2.044368, -0.676903] |
| 17_all_neural_underharm4_source_zero_guard | scene_uniform | -0.009185 | [-0.041696, 0.017365] |
| 17_all_neural_underharm4_source_zero_guard | joint | -1.308304 | [-2.035516, -0.674309] |
| 17_all_neural_underharm4_source_zero_guard | unary_exact | -1.308304 | [-2.035516, -0.674309] |
| 17_all_neural_underharm4_source_zero_guard | joint_exact | -1.308304 | [-2.035516, -0.674309] |
| 17_all_neural_underharm4_source_zero_guard | pointwise_hard | -0.260638 | [-0.448291, -0.101210] |
| 17_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.008024 | [-0.017020, 0.032477] |
| 17_easy_ridge_no_guard | pointwise | -1.403671 | [-1.887959, -0.876877] |
| 17_easy_ridge_no_guard | independent | -2.677765 | [-3.528584, -1.718147] |
| 17_easy_ridge_no_guard | scene_uniform | -0.432095 | [-0.620330, -0.254968] |
| 17_easy_ridge_no_guard | joint | -2.627322 | [-3.430835, -1.710766] |
| 17_easy_ridge_no_guard | unary_exact | -2.672167 | [-3.505614, -1.713973] |
| 17_easy_ridge_no_guard | joint_exact | -2.645238 | [-3.452770, -1.715007] |
| 17_easy_ridge_no_guard | pointwise_hard | -1.538765 | [-2.124792, -0.980318] |
| 17_easy_ridge_no_guard | pointwise_easy | -1.062882 | [-1.769928, -0.146382] |
| 17_easy_ridge_source_zero_guard | pointwise | -0.888981 | [-1.481645, -0.353400] |
| 17_easy_ridge_source_zero_guard | independent | -1.699013 | [-2.764581, -0.699224] |
| 17_easy_ridge_source_zero_guard | scene_uniform | -0.252508 | [-0.457712, -0.083663] |
| 17_easy_ridge_source_zero_guard | joint | -1.699228 | [-2.756218, -0.700474] |
| 17_easy_ridge_source_zero_guard | unary_exact | -1.699228 | [-2.756218, -0.700474] |
| 17_easy_ridge_source_zero_guard | joint_exact | -1.699228 | [-2.756218, -0.700474] |
| 17_easy_ridge_source_zero_guard | pointwise_hard | -0.801829 | [-1.403691, -0.295978] |
| 17_easy_ridge_source_zero_guard | pointwise_easy | -0.976280 | [-1.516055, -0.478533] |
| 17_easy_neural_underharm4_no_guard | pointwise | -1.888934 | [-2.889298, -0.894940] |
| 17_easy_neural_underharm4_no_guard | independent | -2.455653 | [-3.305705, -1.638974] |
| 17_easy_neural_underharm4_no_guard | scene_uniform | -1.001927 | [-1.665329, -0.425756] |
| 17_easy_neural_underharm4_no_guard | joint | -2.456823 | [-3.304597, -1.643019] |
| 17_easy_neural_underharm4_no_guard | unary_exact | -2.457307 | [-3.305419, -1.643008] |
| 17_easy_neural_underharm4_no_guard | joint_exact | -2.456823 | [-3.304597, -1.643019] |
| 17_easy_neural_underharm4_no_guard | pointwise_hard | -1.807260 | [-2.765200, -0.850620] |
| 17_easy_neural_underharm4_no_guard | pointwise_easy | -1.898773 | [-3.008765, -0.887410] |
| 17_easy_neural_underharm4_source_zero_guard | pointwise | -1.896385 | [-2.892810, -0.904418] |
| 17_easy_neural_underharm4_source_zero_guard | independent | -1.836796 | [-2.939699, -0.852070] |
| 17_easy_neural_underharm4_source_zero_guard | scene_uniform | -1.006658 | [-1.667729, -0.431884] |
| 17_easy_neural_underharm4_source_zero_guard | joint | -1.836012 | [-2.938111, -0.851305] |
| 17_easy_neural_underharm4_source_zero_guard | unary_exact | -1.836506 | [-2.939118, -0.851787] |
| 17_easy_neural_underharm4_source_zero_guard | joint_exact | -1.836012 | [-2.938111, -0.851305] |
| 17_easy_neural_underharm4_source_zero_guard | pointwise_hard | -1.808866 | [-2.766065, -0.852994] |
| 17_easy_neural_underharm4_source_zero_guard | pointwise_easy | -1.851045 | [-2.990512, -0.814770] |
| 29_all_ridge_no_guard | pointwise | -1.295186 | [-2.087761, -0.556155] |
| 29_all_ridge_no_guard | independent | -2.211748 | [-3.693101, -0.626090] |
| 29_all_ridge_no_guard | scene_uniform | -0.234262 | [-0.452195, -0.026860] |
| 29_all_ridge_no_guard | joint | -2.217702 | [-3.707560, -0.626516] |
| 29_all_ridge_no_guard | unary_exact | -2.209634 | [-3.682372, -0.628238] |
| 29_all_ridge_no_guard | joint_exact | -2.220326 | [-3.709429, -0.631318] |
| 29_all_ridge_no_guard | pointwise_hard | -1.247715 | [-1.919641, -0.616374] |
| 29_all_ridge_no_guard | pointwise_easy | -0.262430 | [-1.101275, 0.377335] |
| 29_all_ridge_source_zero_guard | pointwise | -0.482291 | [-0.911507, -0.125360] |
| 29_all_ridge_source_zero_guard | independent | -1.589199 | [-2.842588, -0.409554] |
| 29_all_ridge_source_zero_guard | scene_uniform | -0.084420 | [-0.270934, 0.065517] |
| 29_all_ridge_source_zero_guard | joint | -1.597572 | [-2.854705, -0.413942] |
| 29_all_ridge_source_zero_guard | unary_exact | -1.598772 | [-2.857200, -0.414477] |
| 29_all_ridge_source_zero_guard | joint_exact | -1.597654 | [-2.854707, -0.414185] |
| 29_all_ridge_source_zero_guard | pointwise_hard | -0.485773 | [-0.903595, -0.137811] |
| 29_all_ridge_source_zero_guard | pointwise_easy | 0.013821 | [-0.134500, 0.185387] |
| 29_all_neural_underharm4_no_guard | pointwise | -0.568447 | [-0.990003, -0.211690] |
| 29_all_neural_underharm4_no_guard | independent | -1.723019 | [-2.310612, -1.170517] |
| 29_all_neural_underharm4_no_guard | scene_uniform | 0.002646 | [-0.029453, 0.032969] |
| 29_all_neural_underharm4_no_guard | joint | -1.735677 | [-2.338347, -1.179911] |
| 29_all_neural_underharm4_no_guard | unary_exact | -1.724027 | [-2.312177, -1.179406] |
| 29_all_neural_underharm4_no_guard | joint_exact | -1.737367 | [-2.341193, -1.179915] |
| 29_all_neural_underharm4_no_guard | pointwise_hard | -0.737722 | [-1.224791, -0.322718] |
| 29_all_neural_underharm4_no_guard | pointwise_easy | 0.030721 | [-0.004845, 0.080336] |
| 29_all_neural_underharm4_source_zero_guard | pointwise | -0.242659 | [-0.477033, -0.057839] |
| 29_all_neural_underharm4_source_zero_guard | independent | -1.147729 | [-1.828630, -0.568432] |
| 29_all_neural_underharm4_source_zero_guard | scene_uniform | 0.007864 | [-0.003197, 0.027752] |
| 29_all_neural_underharm4_source_zero_guard | joint | -1.143766 | [-1.824501, -0.567468] |
| 29_all_neural_underharm4_source_zero_guard | unary_exact | -1.143766 | [-1.824501, -0.567468] |
| 29_all_neural_underharm4_source_zero_guard | joint_exact | -1.143766 | [-1.824501, -0.567468] |
| 29_all_neural_underharm4_source_zero_guard | pointwise_hard | -0.301523 | [-0.596746, -0.067385] |
| 29_all_neural_underharm4_source_zero_guard | pointwise_easy | 0.005705 | [-0.007676, 0.022627] |
| 29_easy_ridge_no_guard | pointwise | -1.367563 | [-1.837565, -0.872361] |
| 29_easy_ridge_no_guard | independent | -2.517588 | [-3.585080, -1.314505] |
| 29_easy_ridge_no_guard | scene_uniform | -0.210636 | [-0.351551, -0.092103] |
| 29_easy_ridge_no_guard | joint | -2.553006 | [-3.614946, -1.354562] |
| 29_easy_ridge_no_guard | unary_exact | -2.516747 | [-3.566792, -1.327314] |
| 29_easy_ridge_no_guard | joint_exact | -2.530647 | [-3.586175, -1.330395] |
| 29_easy_ridge_no_guard | pointwise_hard | -1.413743 | [-1.885183, -0.930905] |
| 29_easy_ridge_no_guard | pointwise_easy | -0.851600 | [-1.590254, 0.081153] |
| 29_easy_ridge_source_zero_guard | pointwise | -0.886148 | [-1.486305, -0.354391] |
| 29_easy_ridge_source_zero_guard | independent | -1.434178 | [-2.629705, -0.311327] |
| 29_easy_ridge_source_zero_guard | scene_uniform | -0.133842 | [-0.250625, -0.040876] |
| 29_easy_ridge_source_zero_guard | joint | -1.445183 | [-2.639526, -0.317537] |
| 29_easy_ridge_source_zero_guard | unary_exact | -1.442365 | [-2.634882, -0.315041] |
| 29_easy_ridge_source_zero_guard | joint_exact | -1.443356 | [-2.635873, -0.316008] |
| 29_easy_ridge_source_zero_guard | pointwise_hard | -0.829323 | [-1.435900, -0.305100] |
| 29_easy_ridge_source_zero_guard | pointwise_easy | -0.746645 | [-1.176776, -0.357542] |
| 29_easy_neural_underharm4_no_guard | pointwise | -2.076768 | [-3.065869, -1.073727] |
| 29_easy_neural_underharm4_no_guard | independent | -3.130330 | [-4.105283, -2.195863] |
| 29_easy_neural_underharm4_no_guard | scene_uniform | -0.765800 | [-1.269593, -0.340001] |
| 29_easy_neural_underharm4_no_guard | joint | -3.116301 | [-4.094375, -2.180107] |
| 29_easy_neural_underharm4_no_guard | unary_exact | -3.129437 | [-4.111356, -2.191707] |
| 29_easy_neural_underharm4_no_guard | joint_exact | -3.126338 | [-4.104894, -2.191670] |
| 29_easy_neural_underharm4_no_guard | pointwise_hard | -2.023274 | [-3.030036, -1.015791] |
| 29_easy_neural_underharm4_no_guard | pointwise_easy | -1.846315 | [-2.815765, -0.998173] |
| 29_easy_neural_underharm4_source_zero_guard | pointwise | -2.025307 | [-3.047916, -1.005193] |
| 29_easy_neural_underharm4_source_zero_guard | independent | -2.013678 | [-3.033008, -1.023960] |
| 29_easy_neural_underharm4_source_zero_guard | scene_uniform | -0.763089 | [-1.268133, -0.334758] |
| 29_easy_neural_underharm4_source_zero_guard | joint | -2.013469 | [-3.032762, -1.023958] |
| 29_easy_neural_underharm4_source_zero_guard | unary_exact | -2.013678 | [-3.033008, -1.023960] |
| 29_easy_neural_underharm4_source_zero_guard | joint_exact | -2.013678 | [-3.033008, -1.023960] |
| 29_easy_neural_underharm4_source_zero_guard | pointwise_hard | -2.013731 | [-3.028309, -1.005728] |
| 29_easy_neural_underharm4_source_zero_guard | pointwise_easy | -1.564249 | [-2.652121, -0.614967] |
| 43_all_ridge_no_guard | pointwise | -1.223506 | [-1.982285, -0.518133] |
| 43_all_ridge_no_guard | independent | -1.731896 | [-3.038473, -0.227488] |
| 43_all_ridge_no_guard | scene_uniform | -0.243391 | [-0.587230, 0.068713] |
| 43_all_ridge_no_guard | joint | -1.680365 | [-3.025716, -0.132137] |
| 43_all_ridge_no_guard | unary_exact | -1.716925 | [-3.014253, -0.214138] |
| 43_all_ridge_no_guard | joint_exact | -1.703878 | [-2.987079, -0.211852] |
| 43_all_ridge_no_guard | pointwise_hard | -1.200866 | [-1.812167, -0.631335] |
| 43_all_ridge_no_guard | pointwise_easy | -0.255358 | [-1.086585, 0.413122] |
| 43_all_ridge_source_zero_guard | pointwise | -0.525734 | [-0.964241, -0.150185] |
| 43_all_ridge_source_zero_guard | independent | -1.436740 | [-2.569594, -0.235801] |
| 43_all_ridge_source_zero_guard | scene_uniform | -0.095111 | [-0.289827, 0.067089] |
| 43_all_ridge_source_zero_guard | joint | -1.417560 | [-2.537954, -0.226567] |
| 43_all_ridge_source_zero_guard | unary_exact | -1.423120 | [-2.544987, -0.226779] |
| 43_all_ridge_source_zero_guard | joint_exact | -1.417560 | [-2.537954, -0.226567] |
| 43_all_ridge_source_zero_guard | pointwise_hard | -0.522097 | [-0.946409, -0.172933] |
| 43_all_ridge_source_zero_guard | pointwise_easy | 0.039789 | [-0.164376, 0.261227] |
| 43_all_neural_underharm4_no_guard | pointwise | -0.506473 | [-0.801841, -0.225727] |
| 43_all_neural_underharm4_no_guard | independent | -2.000812 | [-2.768884, -1.221815] |
| 43_all_neural_underharm4_no_guard | scene_uniform | -0.034180 | [-0.084245, 0.014195] |
| 43_all_neural_underharm4_no_guard | joint | -1.995145 | [-2.760032, -1.221037] |
| 43_all_neural_underharm4_no_guard | unary_exact | -1.991120 | [-2.754067, -1.220335] |
| 43_all_neural_underharm4_no_guard | joint_exact | -1.993798 | [-2.759207, -1.220991] |
| 43_all_neural_underharm4_no_guard | pointwise_hard | -0.646920 | [-0.959521, -0.344660] |
| 43_all_neural_underharm4_no_guard | pointwise_easy | 0.029801 | [-0.026193, 0.086527] |
| 43_all_neural_underharm4_source_zero_guard | pointwise | -0.342925 | [-0.621486, -0.116286] |
| 43_all_neural_underharm4_source_zero_guard | independent | -1.419541 | [-2.260454, -0.719239] |
| 43_all_neural_underharm4_source_zero_guard | scene_uniform | -0.028924 | [-0.075415, 0.010436] |
| 43_all_neural_underharm4_source_zero_guard | joint | -1.411826 | [-2.250151, -0.716116] |
| 43_all_neural_underharm4_source_zero_guard | unary_exact | -1.411396 | [-2.249699, -0.715696] |
| 43_all_neural_underharm4_source_zero_guard | joint_exact | -1.411396 | [-2.249699, -0.715696] |
| 43_all_neural_underharm4_source_zero_guard | pointwise_hard | -0.403918 | [-0.729372, -0.140868] |
| 43_all_neural_underharm4_source_zero_guard | pointwise_easy | -0.004484 | [-0.043200, 0.025884] |
| 43_easy_ridge_no_guard | pointwise | -1.336630 | [-1.717713, -0.899289] |
| 43_easy_ridge_no_guard | independent | -3.207030 | [-4.125616, -2.190146] |
| 43_easy_ridge_no_guard | scene_uniform | -0.381919 | [-0.571109, -0.209971] |
| 43_easy_ridge_no_guard | joint | -3.202081 | [-4.105999, -2.190118] |
| 43_easy_ridge_no_guard | unary_exact | -3.184607 | [-4.086976, -2.169190] |
| 43_easy_ridge_no_guard | joint_exact | -3.199831 | [-4.115444, -2.189407] |
| 43_easy_ridge_no_guard | pointwise_hard | -1.403648 | [-1.800378, -0.943683] |
| 43_easy_ridge_no_guard | pointwise_easy | -0.927321 | [-1.652067, -0.001425] |
| 43_easy_ridge_source_zero_guard | pointwise | -0.837003 | [-1.357769, -0.351356] |
| 43_easy_ridge_source_zero_guard | independent | -1.909433 | [-3.046070, -0.866547] |
| 43_easy_ridge_source_zero_guard | scene_uniform | -0.200096 | [-0.346874, -0.077798] |
| 43_easy_ridge_source_zero_guard | joint | -1.905236 | [-3.023497, -0.863859] |
| 43_easy_ridge_source_zero_guard | unary_exact | -1.907964 | [-3.027570, -0.864211] |
| 43_easy_ridge_source_zero_guard | joint_exact | -1.908804 | [-3.030633, -0.865031] |
| 43_easy_ridge_source_zero_guard | pointwise_hard | -0.760056 | [-1.233794, -0.305270] |
| 43_easy_ridge_source_zero_guard | pointwise_easy | -0.862398 | [-1.323319, -0.427143] |
| 43_easy_neural_underharm4_no_guard | pointwise | -1.800993 | [-2.630776, -0.961184] |
| 43_easy_neural_underharm4_no_guard | independent | -3.099961 | [-4.081062, -2.124281] |
| 43_easy_neural_underharm4_no_guard | scene_uniform | -0.833851 | [-1.434802, -0.355980] |
| 43_easy_neural_underharm4_no_guard | joint | -3.085078 | [-4.048937, -2.116699] |
| 43_easy_neural_underharm4_no_guard | unary_exact | -3.083811 | [-4.046955, -2.113924] |
| 43_easy_neural_underharm4_no_guard | joint_exact | -3.083793 | [-4.047010, -2.113908] |
| 43_easy_neural_underharm4_no_guard | pointwise_hard | -1.673591 | [-2.492290, -0.869486] |
| 43_easy_neural_underharm4_no_guard | pointwise_easy | -2.051341 | [-3.049743, -1.163218] |
| 43_easy_neural_underharm4_source_zero_guard | pointwise | -1.770057 | [-2.615811, -0.908983] |
| 43_easy_neural_underharm4_source_zero_guard | independent | -2.055330 | [-3.279937, -0.969840] |
| 43_easy_neural_underharm4_source_zero_guard | scene_uniform | -0.790220 | [-1.414507, -0.305614] |
| 43_easy_neural_underharm4_source_zero_guard | joint | -2.044992 | [-3.263148, -0.969737] |
| 43_easy_neural_underharm4_source_zero_guard | unary_exact | -2.046430 | [-3.269130, -0.969669] |
| 43_easy_neural_underharm4_source_zero_guard | joint_exact | -2.046498 | [-3.269132, -0.969737] |
| 43_easy_neural_underharm4_source_zero_guard | pointwise_hard | -1.661930 | [-2.489560, -0.849336] |
| 43_easy_neural_underharm4_source_zero_guard | pointwise_easy | -1.895222 | [-2.969189, -0.935290] |

## Joint Pilot

| View | Rule | ADE gain (%) | CI | Worst easy degradation (%) | Switch (%) |
|---|---|---:|---|---:|---:|
| 17_neural_all_ridge_no_guard | independent | 2.817424 | [1.431816, 4.504215] | 17.406240 | 19.898627 |
| 17_neural_all_ridge_no_guard | scene_uniform | 0.249824 | [-0.007639, 0.660697] | -0.000000 | 0.686723 |
| 17_neural_all_ridge_no_guard | joint | 2.791805 | [1.411804, 4.468741] | 17.406240 | 19.833224 |
| 17_neural_all_ridge_no_guard | unary_exact | 2.804471 | [1.409042, 4.498476] | 17.406240 | 19.898627 |
| 17_neural_all_ridge_no_guard | joint_exact | 2.794793 | [1.400948, 4.495864] | 17.406240 | 19.898627 |
| 17_neural_all_ridge_source_zero_guard | independent | 1.425691 | [0.563519, 2.567926] | 4.927829 | 7.128842 |
| 17_neural_all_ridge_source_zero_guard | scene_uniform | 0.080564 | [0.006832, 0.191792] | -0.000000 | 0.408764 |
| 17_neural_all_ridge_source_zero_guard | joint | 1.431322 | [0.564539, 2.579246] | 4.927829 | 7.112492 |
| 17_neural_all_ridge_source_zero_guard | unary_exact | 1.420198 | [0.558086, 2.568123] | 4.927829 | 7.128842 |
| 17_neural_all_ridge_source_zero_guard | joint_exact | 1.420198 | [0.558086, 2.568123] | 4.927829 | 7.128842 |
| 17_neural_all_neural_underharm4_no_guard | independent | 0.735904 | [0.103273, 1.511682] | 6.773534 | 8.420536 |
| 17_neural_all_neural_underharm4_no_guard | scene_uniform | 0.013743 | [-0.034107, 0.067268] | -0.000000 | 0.147155 |
| 17_neural_all_neural_underharm4_no_guard | joint | 0.731504 | [0.104118, 1.505169] | 5.892847 | 8.420536 |
| 17_neural_all_neural_underharm4_no_guard | unary_exact | 0.732005 | [0.105596, 1.505182] | 5.892847 | 8.420536 |
| 17_neural_all_neural_underharm4_no_guard | joint_exact | 0.731504 | [0.104118, 1.505169] | 5.892847 | 8.420536 |
| 17_neural_all_neural_underharm4_source_zero_guard | independent | 0.272575 | [0.100970, 0.481015] | 6.773534 | 3.172008 |
| 17_neural_all_neural_underharm4_source_zero_guard | scene_uniform | 0.010054 | [-0.003901, 0.034062] | -0.000000 | 0.049052 |
| 17_neural_all_neural_underharm4_source_zero_guard | joint | 0.274416 | [0.103096, 0.483394] | 5.892847 | 3.172008 |
| 17_neural_all_neural_underharm4_source_zero_guard | unary_exact | 0.274416 | [0.103096, 0.483394] | 5.892847 | 3.172008 |
| 17_neural_all_neural_underharm4_source_zero_guard | joint_exact | 0.274416 | [0.103096, 0.483394] | 5.892847 | 3.172008 |
| 17_neural_easy_ridge_no_guard | independent | 1.528725 | [0.884814, 2.374001] | 1.191222 | 12.459124 |
| 17_neural_easy_ridge_no_guard | scene_uniform | 0.032757 | [0.008321, 0.062655] | -0.000000 | 0.212557 |
| 17_neural_easy_ridge_no_guard | joint | 1.566478 | [0.917859, 2.395546] | 1.191222 | 12.491825 |
| 17_neural_easy_ridge_no_guard | unary_exact | 1.519150 | [0.865390, 2.359556] | 1.191222 | 12.459124 |
| 17_neural_easy_ridge_no_guard | joint_exact | 1.549560 | [0.900736, 2.388369] | 1.191222 | 12.459124 |
| 17_neural_easy_ridge_source_zero_guard | independent | 0.927600 | [0.427780, 1.452004] | 1.191222 | 2.844997 |
| 17_neural_easy_ridge_source_zero_guard | scene_uniform | 0.009799 | [0.002239, 0.018544] | -0.000000 | 0.114454 |
| 17_neural_easy_ridge_source_zero_guard | joint | 0.919778 | [0.414114, 1.449243] | 1.191222 | 2.844997 |
| 17_neural_easy_ridge_source_zero_guard | unary_exact | 0.919778 | [0.414114, 1.449243] | 1.191222 | 2.844997 |
| 17_neural_easy_ridge_source_zero_guard | joint_exact | 0.919778 | [0.414114, 1.449243] | 1.191222 | 2.844997 |
| 17_neural_easy_neural_underharm4_no_guard | independent | 1.340699 | [0.614833, 2.230665] | 0.825823 | 7.226946 |
| 17_neural_easy_neural_underharm4_no_guard | scene_uniform | 0.132295 | [0.033937, 0.253199] | -0.000000 | 0.457816 |
| 17_neural_easy_neural_underharm4_no_guard | joint | 1.339709 | [0.612803, 2.228710] | 0.825823 | 7.226946 |
| 17_neural_easy_neural_underharm4_no_guard | unary_exact | 1.339709 | [0.612803, 2.228710] | 0.825823 | 7.226946 |
| 17_neural_easy_neural_underharm4_no_guard | joint_exact | 1.339709 | [0.612803, 2.228710] | 0.825823 | 7.226946 |
| 17_neural_easy_neural_underharm4_source_zero_guard | independent | 1.022644 | [0.314071, 1.925392] | 0.825823 | 2.092871 |
| 17_neural_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.124856 | [0.024931, 0.247144] | -0.000000 | 0.441465 |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint | 1.022644 | [0.314071, 1.925392] | 0.825823 | 2.092871 |
| 17_neural_easy_neural_underharm4_source_zero_guard | unary_exact | 1.022644 | [0.314071, 1.925392] | 0.825823 | 2.092871 |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint_exact | 1.022644 | [0.314071, 1.925392] | 0.825823 | 2.092871 |
| 17_damping097_all_ridge_no_guard | independent | 4.641204 | [3.697470, 5.409073] | 5.399579 | 57.717462 |
| 17_damping097_all_ridge_no_guard | scene_uniform | 0.534048 | [0.157946, 1.084559] | 0.553836 | 5.183126 |
| 17_damping097_all_ridge_no_guard | joint | 4.636439 | [3.684751, 5.407916] | 5.399579 | 57.635710 |
| 17_damping097_all_ridge_no_guard | unary_exact | 4.623595 | [3.677165, 5.393682] | 5.399579 | 57.717462 |
| 17_damping097_all_ridge_no_guard | joint_exact | 4.639008 | [3.687322, 5.409985] | 5.399579 | 57.717462 |
| 17_damping097_all_ridge_source_zero_guard | independent | 2.881938 | [1.515727, 4.282104] | 0.626173 | 29.529104 |
| 17_damping097_all_ridge_source_zero_guard | scene_uniform | 0.135899 | [0.025444, 0.266771] | 0.553836 | 0.817528 |
| 17_damping097_all_ridge_source_zero_guard | joint | 2.878971 | [1.511266, 4.283202] | 0.626173 | 29.512753 |
| 17_damping097_all_ridge_source_zero_guard | unary_exact | 2.866304 | [1.506211, 4.269646] | 0.626173 | 29.529104 |
| 17_damping097_all_ridge_source_zero_guard | joint_exact | 2.879742 | [1.511266, 4.284463] | 0.626173 | 29.529104 |
| 17_damping097_all_neural_underharm4_no_guard | independent | 2.563857 | [1.770956, 3.380167] | 1.989607 | 29.725311 |
| 17_damping097_all_neural_underharm4_no_guard | scene_uniform | 0.019514 | [-0.000983, 0.048181] | -0.000000 | 0.179856 |
| 17_damping097_all_neural_underharm4_no_guard | joint | 2.560169 | [1.767860, 3.374688] | 2.000919 | 29.692610 |
| 17_damping097_all_neural_underharm4_no_guard | unary_exact | 2.560876 | [1.770005, 3.374197] | 1.471508 | 29.725311 |
| 17_damping097_all_neural_underharm4_no_guard | joint_exact | 2.560576 | [1.768268, 3.375502] | 1.929954 | 29.725311 |
| 17_damping097_all_neural_underharm4_source_zero_guard | independent | 1.547823 | [0.796055, 2.362672] | 0.043847 | 12.655330 |
| 17_damping097_all_neural_underharm4_source_zero_guard | scene_uniform | 0.019222 | [0.001561, 0.045665] | -0.000000 | 0.130804 |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint | 1.545828 | [0.795737, 2.358641] | 0.043847 | 12.655330 |
| 17_damping097_all_neural_underharm4_source_zero_guard | unary_exact | 1.545828 | [0.795737, 2.358641] | 0.043847 | 12.655330 |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint_exact | 1.545828 | [0.795737, 2.358641] | 0.043847 | 12.655330 |
| 17_damping097_easy_ridge_no_guard | independent | 4.079762 | [3.069645, 4.941360] | 0.457574 | 59.009156 |
| 17_damping097_easy_ridge_no_guard | scene_uniform | 0.461797 | [0.280715, 0.649883] | 2.982040 | 7.210595 |
| 17_damping097_easy_ridge_no_guard | joint | 4.070641 | [3.068928, 4.927391] | 0.647216 | 58.943754 |
| 17_damping097_easy_ridge_no_guard | unary_exact | 4.065853 | [3.063779, 4.915629] | 0.647216 | 59.009156 |
| 17_damping097_easy_ridge_no_guard | joint_exact | 4.070681 | [3.067760, 4.922596] | 0.647216 | 59.009156 |
| 17_damping097_easy_ridge_source_zero_guard | independent | 2.544087 | [1.260074, 3.867677] | 0.457574 | 31.556573 |
| 17_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.260556 | [0.091042, 0.466595] | -0.000000 | 3.580772 |
| 17_damping097_easy_ridge_source_zero_guard | joint | 2.537106 | [1.258436, 3.855656] | 0.647216 | 31.556573 |
| 17_damping097_easy_ridge_source_zero_guard | unary_exact | 2.537106 | [1.258436, 3.855656] | 0.647216 | 31.556573 |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact | 2.537106 | [1.258436, 3.855656] | 0.647216 | 31.556573 |
| 17_damping097_easy_neural_underharm4_no_guard | independent | 3.689411 | [2.623033, 4.656518] | 0.504167 | 52.779595 |
| 17_damping097_easy_neural_underharm4_no_guard | scene_uniform | 1.111408 | [0.536041, 1.749617] | 0.134152 | 14.306736 |
| 17_damping097_easy_neural_underharm4_no_guard | joint | 3.689556 | [2.620485, 4.655981] | 0.504167 | 52.779595 |
| 17_damping097_easy_neural_underharm4_no_guard | unary_exact | 3.689996 | [2.620907, 4.656400] | 0.455434 | 52.779595 |
| 17_damping097_easy_neural_underharm4_no_guard | joint_exact | 3.689556 | [2.620485, 4.655981] | 0.504167 | 52.779595 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | independent | 2.772613 | [1.436696, 4.119753] | 0.455434 | 39.862655 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | scene_uniform | 1.108698 | [0.531740, 1.749617] | 0.134152 | 14.290386 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint | 2.771899 | [1.435303, 4.116234] | 0.455434 | 39.862655 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | unary_exact | 2.772349 | [1.436181, 4.118451] | 0.455434 | 39.862655 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint_exact | 2.771899 | [1.435303, 4.116234] | 0.455434 | 39.862655 |
| 29_neural_all_ridge_no_guard | independent | 2.541504 | [1.112926, 4.204099] | 17.670016 | 16.121648 |
| 29_neural_all_ridge_no_guard | scene_uniform | 0.224936 | [-0.072704, 0.690500] | 0.302621 | 0.604971 |
| 29_neural_all_ridge_no_guard | joint | 2.528276 | [1.098921, 4.184321] | 17.670016 | 16.088947 |
| 29_neural_all_ridge_no_guard | unary_exact | 2.538684 | [1.106957, 4.182480] | 17.670016 | 16.121648 |
| 29_neural_all_ridge_no_guard | joint_exact | 2.527498 | [1.098921, 4.182200] | 17.670016 | 16.121648 |
| 29_neural_all_ridge_source_zero_guard | independent | 1.344878 | [0.491048, 2.489584] | 4.677195 | 6.066056 |
| 29_neural_all_ridge_source_zero_guard | scene_uniform | 0.047010 | [-0.052271, 0.165968] | 0.302621 | 0.277959 |
| 29_neural_all_ridge_source_zero_guard | joint | 1.330760 | [0.485517, 2.475417] | 4.677195 | 6.066056 |
| 29_neural_all_ridge_source_zero_guard | unary_exact | 1.330760 | [0.485517, 2.475417] | 4.677195 | 6.066056 |
| 29_neural_all_ridge_source_zero_guard | joint_exact | 1.330760 | [0.485517, 2.475417] | 4.677195 | 6.066056 |
| 29_neural_all_neural_underharm4_no_guard | independent | 0.704444 | [0.139534, 1.397876] | 3.932105 | 5.869850 |
| 29_neural_all_neural_underharm4_no_guard | scene_uniform | 0.007112 | [-0.034766, 0.048586] | -0.000000 | 0.098103 |
| 29_neural_all_neural_underharm4_no_guard | joint | 0.687174 | [0.129236, 1.380579] | 3.932105 | 5.853499 |
| 29_neural_all_neural_underharm4_no_guard | unary_exact | 0.698308 | [0.136554, 1.388809] | 3.932105 | 5.869850 |
| 29_neural_all_neural_underharm4_no_guard | joint_exact | 0.686973 | [0.128645, 1.380230] | 3.932105 | 5.869850 |
| 29_neural_all_neural_underharm4_source_zero_guard | independent | 0.192433 | [-0.116020, 0.496406] | 1.676569 | 2.321779 |
| 29_neural_all_neural_underharm4_source_zero_guard | scene_uniform | 0.012045 | [-0.004348, 0.040483] | -0.000000 | 0.049052 |
| 29_neural_all_neural_underharm4_source_zero_guard | joint | 0.192433 | [-0.116020, 0.496406] | 1.676569 | 2.321779 |
| 29_neural_all_neural_underharm4_source_zero_guard | unary_exact | 0.192433 | [-0.116020, 0.496406] | 1.676569 | 2.321779 |
| 29_neural_all_neural_underharm4_source_zero_guard | joint_exact | 0.192433 | [-0.116020, 0.496406] | 1.676569 | 2.321779 |
| 29_neural_easy_ridge_no_guard | independent | 1.702860 | [0.951537, 2.606899] | 2.895802 | 9.565075 |
| 29_neural_easy_ridge_no_guard | scene_uniform | 0.029486 | [0.004646, 0.060518] | -0.000000 | 0.163506 |
| 29_neural_easy_ridge_no_guard | joint | 1.677303 | [0.923028, 2.587999] | 2.895802 | 9.532374 |
| 29_neural_easy_ridge_no_guard | unary_exact | 1.699756 | [0.945599, 2.613975] | 0.735997 | 9.565075 |
| 29_neural_easy_ridge_no_guard | joint_exact | 1.689006 | [0.935579, 2.592744] | 2.895802 | 9.565075 |
| 29_neural_easy_ridge_source_zero_guard | independent | 1.053535 | [0.372143, 1.889734] | 0.735997 | 2.468934 |
| 29_neural_easy_ridge_source_zero_guard | scene_uniform | 0.006292 | [0.000000, 0.014476] | -0.000000 | 0.065402 |
| 29_neural_easy_ridge_source_zero_guard | joint | 1.045817 | [0.357402, 1.884782] | 0.735997 | 2.468934 |
| 29_neural_easy_ridge_source_zero_guard | unary_exact | 1.045817 | [0.357402, 1.884782] | 0.735997 | 2.468934 |
| 29_neural_easy_ridge_source_zero_guard | joint_exact | 1.045817 | [0.357402, 1.884782] | 0.735997 | 2.468934 |
| 29_neural_easy_neural_underharm4_no_guard | independent | 0.862322 | [0.325693, 1.505437] | 1.255801 | 3.875082 |
| 29_neural_easy_neural_underharm4_no_guard | scene_uniform | 0.042618 | [0.011436, 0.079153] | 0.302621 | 0.310661 |
| 29_neural_easy_neural_underharm4_no_guard | joint | 0.859308 | [0.320601, 1.505437] | 1.255801 | 3.891432 |
| 29_neural_easy_neural_underharm4_no_guard | unary_exact | 0.862322 | [0.325693, 1.505437] | 1.255801 | 3.875082 |
| 29_neural_easy_neural_underharm4_no_guard | joint_exact | 0.862322 | [0.325693, 1.505437] | 1.255801 | 3.875082 |
| 29_neural_easy_neural_underharm4_source_zero_guard | independent | 0.837036 | [0.293884, 1.486560] | 0.302621 | 1.226292 |
| 29_neural_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.042618 | [0.011436, 0.079153] | 0.302621 | 0.310661 |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint | 0.837036 | [0.293884, 1.486560] | 0.302621 | 1.226292 |
| 29_neural_easy_neural_underharm4_source_zero_guard | unary_exact | 0.837036 | [0.293884, 1.486560] | 0.302621 | 1.226292 |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint_exact | 0.837036 | [0.293884, 1.486560] | 0.302621 | 1.226292 |
| 29_damping097_all_ridge_no_guard | independent | 4.640381 | [3.582963, 5.489804] | 5.399579 | 54.087639 |
| 29_damping097_all_ridge_no_guard | scene_uniform | 0.456882 | [0.117707, 0.964240] | 0.553836 | 3.564421 |
| 29_damping097_all_ridge_no_guard | joint | 4.632901 | [3.570036, 5.485649] | 5.399579 | 53.989536 |
| 29_damping097_all_ridge_no_guard | unary_exact | 4.635540 | [3.570664, 5.488102] | 5.399579 | 54.087639 |
| 29_damping097_all_ridge_no_guard | joint_exact | 4.634548 | [3.570933, 5.488052] | 5.399579 | 54.087639 |
| 29_damping097_all_ridge_source_zero_guard | independent | 2.850229 | [1.420219, 4.290009] | 0.234504 | 27.501635 |
| 29_damping097_all_ridge_source_zero_guard | scene_uniform | 0.130781 | [0.020325, 0.262710] | 0.553836 | 0.784827 |
| 29_damping097_all_ridge_source_zero_guard | joint | 2.844090 | [1.413725, 4.286319] | 0.234504 | 27.468934 |
| 29_damping097_all_ridge_source_zero_guard | unary_exact | 2.845161 | [1.413517, 4.290927] | 0.234504 | 27.501635 |
| 29_damping097_all_ridge_source_zero_guard | joint_exact | 2.844169 | [1.413725, 4.286398] | 0.234504 | 27.501635 |
| 29_damping097_all_neural_underharm4_no_guard | independent | 2.376770 | [1.588533, 3.222075] | 0.361621 | 25.768476 |
| 29_damping097_all_neural_underharm4_no_guard | scene_uniform | 0.004477 | [-0.006730, 0.016319] | -0.000000 | 0.114454 |
| 29_damping097_all_neural_underharm4_no_guard | joint | 2.371744 | [1.582086, 3.215340] | 0.361621 | 25.735775 |
| 29_damping097_all_neural_underharm4_no_guard | unary_exact | 2.371679 | [1.582190, 3.215583] | 0.361621 | 25.768476 |
| 29_damping097_all_neural_underharm4_no_guard | joint_exact | 2.373117 | [1.582190, 3.218829] | 0.361621 | 25.768476 |
| 29_damping097_all_neural_underharm4_source_zero_guard | independent | 1.311709 | [0.615796, 2.078555] | -0.000000 | 11.281884 |
| 29_damping097_all_neural_underharm4_source_zero_guard | scene_uniform | 0.004186 | [-0.002379, 0.013448] | -0.000000 | 0.065402 |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint | 1.307820 | [0.612495, 2.077807] | -0.000000 | 11.281884 |
| 29_damping097_all_neural_underharm4_source_zero_guard | unary_exact | 1.307820 | [0.612495, 2.077807] | -0.000000 | 11.281884 |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint_exact | 1.307820 | [0.612495, 2.077807] | -0.000000 | 11.281884 |
| 29_damping097_easy_ridge_no_guard | independent | 4.088323 | [2.858382, 5.048765] | 0.622123 | 55.166776 |
| 29_damping097_easy_ridge_no_guard | scene_uniform | 0.239078 | [0.117168, 0.383072] | 3.255178 | 3.744277 |
| 29_damping097_easy_ridge_no_guard | joint | 4.096769 | [2.863126, 5.057119] | 0.622123 | 55.035971 |
| 29_damping097_easy_ridge_no_guard | unary_exact | 4.085136 | [2.853967, 5.040589] | 0.622123 | 55.166776 |
| 29_damping097_easy_ridge_no_guard | joint_exact | 4.087487 | [2.854857, 5.043895] | 0.622123 | 55.166776 |
| 29_damping097_easy_ridge_source_zero_guard | independent | 2.412426 | [1.060670, 3.778402] | 0.207270 | 29.447351 |
| 29_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.139615 | [0.046551, 0.255096] | -0.000000 | 1.945716 |
| 29_damping097_easy_ridge_source_zero_guard | joint | 2.415452 | [1.059311, 3.775297] | -0.000000 | 29.414650 |
| 29_damping097_easy_ridge_source_zero_guard | unary_exact | 2.412896 | [1.057687, 3.771764] | 0.207270 | 29.447351 |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact | 2.413809 | [1.057709, 3.772007] | 0.207270 | 29.447351 |
| 29_damping097_easy_neural_underharm4_no_guard | independent | 3.847244 | [2.838046, 4.787885] | -0.176256 | 52.861347 |
| 29_damping097_easy_neural_underharm4_no_guard | scene_uniform | 0.795827 | [0.365945, 1.282096] | -0.000000 | 10.595160 |
| 29_damping097_easy_neural_underharm4_no_guard | joint | 3.831195 | [2.812717, 4.769949] | -0.176256 | 52.877698 |
| 29_damping097_easy_neural_underharm4_no_guard | unary_exact | 3.846187 | [2.835046, 4.788635] | -0.176256 | 52.861347 |
| 29_damping097_easy_neural_underharm4_no_guard | joint_exact | 3.843384 | [2.831744, 4.787367] | -0.176256 | 52.861347 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | independent | 2.757132 | [1.458737, 4.087054] | -0.000000 | 37.393721 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.793117 | [0.362381, 1.279454] | -0.000000 | 10.578810 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint | 2.756939 | [1.458657, 4.086779] | -0.000000 | 37.344670 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | unary_exact | 2.757132 | [1.458737, 4.087054] | -0.000000 | 37.393721 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint_exact | 2.757132 | [1.458737, 4.087054] | -0.000000 | 37.393721 |
| 43_neural_all_ridge_no_guard | independent | 3.002063 | [1.641123, 4.640644] | 5.285604 | 17.887508 |
| 43_neural_all_ridge_no_guard | scene_uniform | 0.309040 | [-0.055388, 0.907452] | 0.302724 | 0.670373 |
| 43_neural_all_ridge_no_guard | joint | 3.045336 | [1.619872, 4.738283] | 5.285604 | 17.838457 |
| 43_neural_all_ridge_no_guard | unary_exact | 3.017806 | [1.647506, 4.651170] | 5.285604 | 17.887508 |
| 43_neural_all_ridge_no_guard | joint_exact | 3.026315 | [1.673033, 4.661886] | 5.285604 | 17.724003 |
| 43_neural_all_ridge_source_zero_guard | independent | 1.521702 | [0.572736, 2.829874] | 1.227280 | 6.687377 |
| 43_neural_all_ridge_source_zero_guard | scene_uniform | 0.052121 | [-0.056810, 0.181677] | 0.302724 | 0.228908 |
| 43_neural_all_ridge_source_zero_guard | joint | 1.537363 | [0.580273, 2.843123] | 1.227280 | 6.687377 |
| 43_neural_all_ridge_source_zero_guard | unary_exact | 1.537363 | [0.580273, 2.843123] | 1.227280 | 6.687377 |
| 43_neural_all_ridge_source_zero_guard | joint_exact | 1.537363 | [0.580273, 2.843123] | 1.227280 | 6.687377 |
| 43_neural_all_neural_underharm4_no_guard | independent | 0.847252 | [0.302438, 1.538562] | 5.093873 | 8.109876 |
| 43_neural_all_neural_underharm4_no_guard | scene_uniform | 0.007064 | [-0.035944, 0.047648] | -0.000000 | 0.147155 |
| 43_neural_all_neural_underharm4_no_guard | joint | 0.855240 | [0.313980, 1.542044] | 5.093873 | 8.093525 |
| 43_neural_all_neural_underharm4_no_guard | unary_exact | 0.855560 | [0.314287, 1.542356] | 5.093873 | 8.109876 |
| 43_neural_all_neural_underharm4_no_guard | joint_exact | 0.855560 | [0.314287, 1.542356] | 5.093873 | 8.109876 |
| 43_neural_all_neural_underharm4_source_zero_guard | independent | 0.304364 | [0.079579, 0.558399] | 5.093873 | 3.335513 |
| 43_neural_all_neural_underharm4_source_zero_guard | scene_uniform | 0.010755 | [-0.004538, 0.036803] | -0.000000 | 0.049052 |
| 43_neural_all_neural_underharm4_source_zero_guard | joint | 0.312590 | [0.088035, 0.567217] | 5.093873 | 3.335513 |
| 43_neural_all_neural_underharm4_source_zero_guard | unary_exact | 0.312590 | [0.088035, 0.567217] | 5.093873 | 3.335513 |
| 43_neural_all_neural_underharm4_source_zero_guard | joint_exact | 0.312590 | [0.088035, 0.567217] | 5.093873 | 3.335513 |
| 43_neural_easy_ridge_no_guard | independent | 1.254345 | [0.517700, 2.081014] | 3.126514 | 11.347286 |
| 43_neural_easy_ridge_no_guard | scene_uniform | 0.019925 | [0.001655, 0.047770] | -0.000000 | 0.147155 |
| 43_neural_easy_ridge_no_guard | joint | 1.242183 | [0.485512, 2.081627] | 3.126514 | 11.298234 |
| 43_neural_easy_ridge_no_guard | unary_exact | 1.258545 | [0.510957, 2.087698] | 1.444859 | 11.347286 |
| 43_neural_easy_ridge_no_guard | joint_exact | 1.249156 | [0.501568, 2.083043] | 3.126514 | 11.347286 |
| 43_neural_easy_ridge_source_zero_guard | independent | 0.831934 | [0.251599, 1.373860] | 2.089947 | 2.697842 |
| 43_neural_easy_ridge_source_zero_guard | scene_uniform | 0.006271 | [0.000000, 0.014383] | -0.000000 | 0.065402 |
| 43_neural_easy_ridge_source_zero_guard | joint | 0.825620 | [0.229611, 1.374332] | 1.444859 | 2.697842 |
| 43_neural_easy_ridge_source_zero_guard | unary_exact | 0.825620 | [0.229611, 1.374332] | 1.444859 | 2.697842 |
| 43_neural_easy_ridge_source_zero_guard | joint_exact | 0.825620 | [0.229611, 1.374332] | 1.444859 | 2.697842 |
| 43_neural_easy_neural_underharm4_no_guard | independent | 1.293155 | [0.581805, 2.173782] | 1.611507 | 5.559189 |
| 43_neural_easy_neural_underharm4_no_guard | scene_uniform | 0.123567 | [0.030560, 0.241617] | 0.302724 | 0.425114 |
| 43_neural_easy_neural_underharm4_no_guard | joint | 1.292406 | [0.581771, 2.173763] | 1.611507 | 5.559189 |
| 43_neural_easy_neural_underharm4_no_guard | unary_exact | 1.292406 | [0.581771, 2.173763] | 1.611507 | 5.559189 |
| 43_neural_easy_neural_underharm4_no_guard | joint_exact | 1.292406 | [0.581771, 2.173763] | 1.611507 | 5.559189 |
| 43_neural_easy_neural_underharm4_source_zero_guard | independent | 1.148516 | [0.405850, 2.065761] | 1.611507 | 2.027469 |
| 43_neural_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.116135 | [0.022317, 0.236406] | 0.302724 | 0.408764 |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint | 1.148516 | [0.405850, 2.065761] | 1.611507 | 2.027469 |
| 43_neural_easy_neural_underharm4_source_zero_guard | unary_exact | 1.148516 | [0.405850, 2.065761] | 1.611507 | 2.027469 |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint_exact | 1.148516 | [0.405850, 2.065761] | 1.611507 | 2.027469 |
| 43_damping097_all_ridge_no_guard | independent | 4.649927 | [3.792660, 5.384078] | 5.399579 | 57.079791 |
| 43_damping097_all_ridge_no_guard | scene_uniform | 0.550947 | [0.181022, 1.038125] | 0.553836 | 5.395683 |
| 43_damping097_all_ridge_no_guard | joint | 4.644825 | [3.782813, 5.383875] | 5.399579 | 56.981687 |
| 43_damping097_all_ridge_no_guard | unary_exact | 4.651310 | [3.785994, 5.393753] | 5.399579 | 57.079791 |
| 43_damping097_all_ridge_no_guard | joint_exact | 4.647728 | [3.783495, 5.386554] | 5.399579 | 57.079791 |
| 43_damping097_all_ridge_source_zero_guard | independent | 2.887541 | [1.558678, 4.245345] | 0.821228 | 28.907783 |
| 43_damping097_all_ridge_source_zero_guard | scene_uniform | 0.146545 | [0.037141, 0.275580] | 0.553836 | 0.882930 |
| 43_damping097_all_ridge_source_zero_guard | joint | 2.885116 | [1.554953, 4.248503] | 0.821228 | 28.907783 |
| 43_damping097_all_ridge_source_zero_guard | unary_exact | 2.890135 | [1.556459, 4.253522] | 0.821228 | 28.907783 |
| 43_damping097_all_ridge_source_zero_guard | joint_exact | 2.885116 | [1.554953, 4.248503] | 0.821228 | 28.907783 |
| 43_damping097_all_neural_underharm4_no_guard | independent | 2.778277 | [1.992016, 3.615145] | 1.531394 | 29.103990 |
| 43_damping097_all_neural_underharm4_no_guard | scene_uniform | 0.041199 | [0.006817, 0.082476] | -0.000000 | 0.425114 |
| 43_damping097_all_neural_underharm4_no_guard | joint | 2.780660 | [1.995992, 3.615455] | 1.518745 | 29.136691 |
| 43_damping097_all_neural_underharm4_no_guard | unary_exact | 2.777207 | [1.991091, 3.612658] | 1.518745 | 29.103990 |
| 43_damping097_all_neural_underharm4_no_guard | joint_exact | 2.779704 | [1.994491, 3.614680] | 1.542329 | 29.103990 |
| 43_damping097_all_neural_underharm4_source_zero_guard | independent | 1.679040 | [0.851545, 2.593129] | 0.003521 | 15.434925 |
| 43_damping097_all_neural_underharm4_source_zero_guard | scene_uniform | 0.039625 | [0.005927, 0.080495] | -0.000000 | 0.327011 |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint | 1.679729 | [0.851761, 2.595071] | 0.003521 | 15.451275 |
| 43_damping097_all_neural_underharm4_source_zero_guard | unary_exact | 1.679319 | [0.851624, 2.594260] | 0.003521 | 15.434925 |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint_exact | 1.679319 | [0.851624, 2.594260] | 0.003521 | 15.434925 |
| 43_damping097_easy_ridge_no_guard | independent | 4.305755 | [3.356735, 5.114134] | 0.721627 | 58.191629 |
| 43_damping097_easy_ridge_no_guard | scene_uniform | 0.399366 | [0.241395, 0.576546] | 2.845076 | 6.115108 |
| 43_damping097_easy_ridge_no_guard | joint | 4.289955 | [3.343697, 5.096568] | 0.721627 | 58.093525 |
| 43_damping097_easy_ridge_no_guard | unary_exact | 4.289635 | [3.347199, 5.085931] | 0.721627 | 58.191629 |
| 43_damping097_easy_ridge_no_guard | joint_exact | 4.294345 | [3.345837, 5.098196] | 0.721627 | 58.191629 |
| 43_damping097_easy_ridge_source_zero_guard | independent | 2.650166 | [1.357046, 3.980883] | 0.231816 | 30.886200 |
| 43_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.205379 | [0.084328, 0.352195] | 0.371067 | 3.073905 |
| 43_damping097_easy_ridge_source_zero_guard | joint | 2.640555 | [1.353690, 3.962433] | 0.372641 | 30.886200 |
| 43_damping097_easy_ridge_source_zero_guard | unary_exact | 2.643047 | [1.356853, 3.960606] | 0.372641 | 30.886200 |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact | 2.643799 | [1.356853, 3.962843] | 0.372641 | 30.886200 |
| 43_damping097_easy_neural_underharm4_no_guard | independent | 4.244311 | [3.337067, 5.112402] | 0.693432 | 55.608241 |
| 43_damping097_easy_neural_underharm4_no_guard | scene_uniform | 0.940783 | [0.458704, 1.532250] | 0.729897 | 11.903205 |
| 43_damping097_easy_neural_underharm4_no_guard | joint | 4.230237 | [3.336528, 5.095627] | 0.693432 | 55.608241 |
| 43_damping097_easy_neural_underharm4_no_guard | unary_exact | 4.229068 | [3.334497, 5.091944] | 0.693432 | 55.608241 |
| 43_damping097_easy_neural_underharm4_no_guard | joint_exact | 4.229052 | [3.334480, 5.092002] | 0.693432 | 55.608241 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | independent | 3.099769 | [1.752101, 4.482383] | 0.693432 | 40.467626 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.889973 | [0.394236, 1.499908] | 0.729897 | 11.723349 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint | 3.090501 | [1.741389, 4.463999] | 0.693432 | 40.434925 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | unary_exact | 3.091773 | [1.742723, 4.465396] | 0.693432 | 40.467626 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint_exact | 3.091835 | [1.742723, 4.465399] | 0.693432 | 40.467626 |

## Within-Candidate Joint Contrasts

| View | Contrast | Gain (%) | CI |
|---|---|---:|---|
| 17_neural_all_ridge_no_guard | joint_vs_independent | -0.026720 | [-0.080566, 0.020143] |
| 17_neural_all_ridge_no_guard | joint_exact_vs_independent | -0.025982 | [-0.075574, 0.011257] |
| 17_neural_all_ridge_no_guard | joint_exact_vs_unary | -0.010240 | [-0.030719, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | joint_vs_independent | 0.005795 | [-0.020099, 0.033404] |
| 17_neural_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_neural_all_neural_underharm4_no_guard | joint_vs_independent | -0.004613 | [-0.020365, 0.007052] |
| 17_neural_all_neural_underharm4_no_guard | joint_exact_vs_independent | -0.004247 | [-0.030610, 0.019123] |
| 17_neural_all_neural_underharm4_no_guard | joint_exact_vs_unary | -0.000539 | [-0.001617, 0.000000] |
| 17_neural_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.001845 | [-0.004353, 0.009196] |
| 17_neural_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_neural_easy_ridge_no_guard | joint_vs_independent | 0.038073 | [-0.029743, 0.156534] |
| 17_neural_easy_ridge_no_guard | joint_exact_vs_independent | -0.031757 | [-0.186162, 0.102397] |
| 17_neural_easy_ridge_no_guard | joint_exact_vs_unary | 0.033104 | [0.000000, 0.099313] |
| 17_neural_easy_ridge_source_zero_guard | joint_vs_independent | -0.007819 | [-0.023458, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_neural_easy_neural_underharm4_no_guard | joint_vs_independent | -0.000985 | [-0.002954, 0.000000] |
| 17_neural_easy_neural_underharm4_no_guard | joint_exact_vs_independent | -0.003399 | [-0.010198, 0.000000] |
| 17_neural_easy_neural_underharm4_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_all_ridge_no_guard | joint_vs_independent | -0.004875 | [-0.012317, 0.000674] |
| 17_damping097_all_ridge_no_guard | joint_exact_vs_independent | -0.007899 | [-0.026613, 0.002702] |
| 17_damping097_all_ridge_no_guard | joint_exact_vs_unary | 0.020390 | [0.000000, 0.056821] |
| 17_damping097_all_ridge_source_zero_guard | joint_vs_independent | -0.002976 | [-0.010182, 0.002155] |
| 17_damping097_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_all_neural_underharm4_no_guard | joint_vs_independent | -0.003806 | [-0.008498, 0.000954] |
| 17_damping097_all_neural_underharm4_no_guard | joint_exact_vs_independent | -0.004014 | [-0.009419, 0.001372] |
| 17_damping097_all_neural_underharm4_no_guard | joint_exact_vs_unary | -0.000357 | [-0.005724, 0.004652] |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint_vs_independent | -0.002054 | [-0.006660, 0.002233] |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_easy_ridge_no_guard | joint_vs_independent | -0.009693 | [-0.028249, 0.002179] |
| 17_damping097_easy_ridge_no_guard | joint_exact_vs_independent | -0.010347 | [-0.030373, 0.001733] |
| 17_damping097_easy_ridge_no_guard | joint_exact_vs_unary | 0.005164 | [-0.001880, 0.017371] |
| 17_damping097_easy_ridge_source_zero_guard | joint_vs_independent | -0.007455 | [-0.025134, 0.002346] |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_easy_neural_underharm4_no_guard | joint_vs_independent | 0.000184 | [-0.005284, 0.007785] |
| 17_damping097_easy_neural_underharm4_no_guard | joint_exact_vs_independent | -0.000159 | [-0.006121, 0.008182] |
| 17_damping097_easy_neural_underharm4_no_guard | joint_exact_vs_unary | -0.000523 | [-0.001601, 0.000032] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint_vs_independent | -0.000748 | [-0.002244, 0.000000] |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_all_ridge_no_guard | joint_vs_independent | -0.013443 | [-0.036961, 0.007007] |
| 29_neural_all_ridge_no_guard | joint_exact_vs_independent | -0.020242 | [-0.055009, 0.009271] |
| 29_neural_all_ridge_no_guard | joint_exact_vs_unary | -0.011671 | [-0.035014, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | joint_vs_independent | -0.014402 | [-0.037506, 0.005591] |
| 29_neural_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_all_neural_underharm4_no_guard | joint_vs_independent | -0.017460 | [-0.044780, 0.000000] |
| 29_neural_all_neural_underharm4_no_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_all_neural_underharm4_no_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_easy_ridge_no_guard | joint_vs_independent | -0.025788 | [-0.063109, 0.000000] |
| 29_neural_easy_ridge_no_guard | joint_exact_vs_independent | -0.067262 | [-0.190669, 0.000000] |
| 29_neural_easy_ridge_no_guard | joint_exact_vs_unary | -0.011839 | [-0.035518, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | joint_vs_independent | -0.007686 | [-0.023059, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_easy_neural_underharm4_no_guard | joint_vs_independent | -0.003013 | [-0.009038, 0.000000] |
| 29_neural_easy_neural_underharm4_no_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_easy_neural_underharm4_no_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_all_ridge_no_guard | joint_vs_independent | -0.007693 | [-0.017229, -0.000478] |
| 29_damping097_all_ridge_no_guard | joint_exact_vs_independent | -0.011976 | [-0.033300, 0.001042] |
| 29_damping097_all_ridge_no_guard | joint_exact_vs_unary | -0.001174 | [-0.007282, 0.003441] |
| 29_damping097_all_ridge_source_zero_guard | joint_vs_independent | -0.006270 | [-0.016136, 0.000665] |
| 29_damping097_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_all_neural_underharm4_no_guard | joint_vs_independent | -0.005133 | [-0.012030, -0.000202] |
| 29_damping097_all_neural_underharm4_no_guard | joint_exact_vs_independent | -0.010873 | [-0.032335, 0.001114] |
| 29_damping097_all_neural_underharm4_no_guard | joint_exact_vs_unary | 0.001532 | [0.000000, 0.004596] |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint_vs_independent | -0.003933 | [-0.010643, 0.000491] |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_easy_ridge_no_guard | joint_vs_independent | 0.008799 | [-0.012885, 0.029873] |
| 29_damping097_easy_ridge_no_guard | joint_exact_vs_independent | -0.000303 | [-0.025994, 0.022752] |
| 29_damping097_easy_ridge_no_guard | joint_exact_vs_unary | 0.002622 | [0.000000, 0.006327] |
| 29_damping097_easy_ridge_source_zero_guard | joint_vs_independent | 0.003072 | [-0.016637, 0.022437] |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_easy_neural_underharm4_no_guard | joint_vs_independent | -0.016634 | [-0.045591, 0.005031] |
| 29_damping097_easy_neural_underharm4_no_guard | joint_exact_vs_independent | -0.005825 | [-0.017185, 0.000000] |
| 29_damping097_easy_neural_underharm4_no_guard | joint_exact_vs_unary | -0.003155 | [-0.008756, 0.000000] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint_vs_independent | -0.000202 | [-0.000488, 0.000000] |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_all_ridge_no_guard | joint_vs_independent | 0.047288 | [-0.030076, 0.138548] |
| 43_neural_all_ridge_no_guard | joint_exact_vs_independent | 0.022872 | [-0.005844, 0.064772] |
| 43_neural_all_ridge_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | joint_vs_independent | 0.016031 | [-0.004465, 0.045891] |
| 43_neural_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_all_neural_underharm4_no_guard | joint_vs_independent | 0.008010 | [-0.000236, 0.022001] |
| 43_neural_all_neural_underharm4_no_guard | joint_exact_vs_independent | 0.021739 | [0.000182, 0.059718] |
| 43_neural_all_neural_underharm4_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.008246 | [0.000000, 0.022023] |
| 43_neural_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_easy_ridge_no_guard | joint_vs_independent | -0.011978 | [-0.038859, 0.008987] |
| 43_neural_easy_ridge_no_guard | joint_exact_vs_independent | -0.035236 | [-0.115233, 0.008103] |
| 43_neural_easy_ridge_no_guard | joint_exact_vs_unary | -0.010186 | [-0.033976, 0.003417] |
| 43_neural_easy_ridge_source_zero_guard | joint_vs_independent | -0.006208 | [-0.022673, 0.004050] |
| 43_neural_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_easy_neural_underharm4_no_guard | joint_vs_independent | -0.000750 | [-0.002251, 0.000000] |
| 43_neural_easy_neural_underharm4_no_guard | joint_exact_vs_independent | -0.000827 | [-0.002480, 0.000000] |
| 43_neural_easy_neural_underharm4_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_all_ridge_no_guard | joint_vs_independent | -0.005246 | [-0.013797, 0.001998] |
| 43_damping097_all_ridge_no_guard | joint_exact_vs_independent | -0.007916 | [-0.027228, 0.004037] |
| 43_damping097_all_ridge_no_guard | joint_exact_vs_unary | -0.004239 | [-0.017326, 0.004610] |
| 43_damping097_all_ridge_source_zero_guard | joint_vs_independent | -0.002421 | [-0.009926, 0.003485] |
| 43_damping097_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_all_neural_underharm4_no_guard | joint_vs_independent | 0.002446 | [-0.002242, 0.008443] |
| 43_damping097_all_neural_underharm4_no_guard | joint_exact_vs_independent | 0.001681 | [-0.003350, 0.007460] |
| 43_damping097_all_neural_underharm4_no_guard | joint_exact_vs_unary | 0.002714 | [-0.000214, 0.008354] |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000707 | [-0.001683, 0.003588] |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_easy_ridge_no_guard | joint_vs_independent | -0.016608 | [-0.032737, -0.002594] |
| 43_damping097_easy_ridge_no_guard | joint_exact_vs_independent | -0.014032 | [-0.032137, -0.000673] |
| 43_damping097_easy_ridge_no_guard | joint_exact_vs_unary | 0.003988 | [-0.014000, 0.020553] |
| 43_damping097_easy_ridge_source_zero_guard | joint_vs_independent | -0.010208 | [-0.026536, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_easy_neural_underharm4_no_guard | joint_vs_independent | -0.014898 | [-0.034505, 0.000000] |
| 43_damping097_easy_neural_underharm4_no_guard | joint_exact_vs_independent | -0.016955 | [-0.041863, 0.000000] |
| 43_damping097_easy_neural_underharm4_no_guard | joint_exact_vs_unary | -0.000016 | [-0.000266, 0.000217] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint_vs_independent | -0.009798 | [-0.027963, 0.000000] |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |

Undefined localities remain undefined; none are dropped to manufacture a CI.
Matched counts apply within each candidate. The two candidates share predicted-risk budgets, not necessarily intervention counts.
318,969 full targets; 6,116 joint targets / 1,152 queries. Joint contains zero zero-CV examples.
3,000 locality resamples, conditional on shared development data/models; no independent safety certification.
Image-pixel obs8/pred12 raw stride12. No t50, seconds, metric, true3D, foundation or deployment claim. Stage5C/SMC off.
