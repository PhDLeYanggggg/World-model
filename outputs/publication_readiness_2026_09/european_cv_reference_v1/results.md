# CV-Reference Repair Results

Fresh cost fitting and fixed control readout on cached_verified neural forecasts. All source-development results; independent reserved data remain closed.
The original neural and previous-policy per-locality readouts reproduce exactly. Actual matched intervention counts are checked from decision arrays.

## Full Registered Cohort

| Seed/head | Rule | ADE gain vs CV (%) | Conditional CI | ADE vs selected baseline (%) | ADE vs fixed damping097 (%) | Easy gain vs CV (%) | Worst easy gain (%) | Zero-CV harmed | Switch rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 17_ridge | CV | 0.000000 | [0.000000, 0.000000] | 0.051508 | -4.379645 | 0.000000 | 0.0 | 0 | 0.000000 |
| 17_ridge | training_selected_baseline | -2.659469 | [-15.435549, 5.062794] | 0.000000 | -6.217447 | -15.475906 | -111.6572648817784 | 0 | 0.000000 |
| 17_ridge | fixed_damping097 | 3.975483 | [0.903779, 5.803674] | 4.645128 | 0.000000 | 1.320277 | -26.575301140144838 | 0 | 0.000000 |
| 17_ridge | neural | 2.449067 | [-6.794709, 8.709852] | 4.338503 | -1.143540 | -13.731925 | -75.95218486074367 | 4 | 1.000000 |
| 17_ridge | pointwise | 5.668090 | [1.165528, 9.639586] | 6.536893 | 1.837827 | -9.499327 | -36.50071964705845 | 4 | 0.854315 |
| 17_ridge | previous_pointwise | 1.954673 | [-8.103772, 8.726326] | 4.013100 | -1.599118 | -13.757989 | -81.40896241420923 | 2 | 0.543037 |
| 17_neural_underharm4 | CV | 0.000000 | [0.000000, 0.000000] | 0.051508 | -4.379645 | 0.000000 | 0.0 | 0 | 0.000000 |
| 17_neural_underharm4 | training_selected_baseline | -2.659469 | [-15.435549, 5.062794] | 0.000000 | -6.217447 | -15.475906 | -111.6572648817784 | 0 | 0.000000 |
| 17_neural_underharm4 | fixed_damping097 | 3.975483 | [0.903779, 5.803674] | 4.645128 | 0.000000 | 1.320277 | -26.575301140144838 | 0 | 0.000000 |
| 17_neural_underharm4 | neural | 2.449067 | [-6.794709, 8.709852] | 4.338503 | -1.143540 | -13.731925 | -75.95218486074367 | 4 | 1.000000 |
| 17_neural_underharm4 | pointwise | 4.301472 | [2.471736, 6.150904] | 4.614751 | 0.197968 | -1.904997 | -8.710791831840181 | 1 | 0.357069 |
| 17_neural_underharm4 | previous_pointwise | 1.159889 | [-9.081521, 7.803053] | 3.285094 | -2.412435 | -13.383862 | -87.2886954766248 | 2 | 0.254297 |
| 29_ridge | CV | 0.000000 | [0.000000, 0.000000] | 0.051508 | -4.379645 | 0.000000 | 0.0 | 0 | 0.000000 |
| 29_ridge | training_selected_baseline | -2.659469 | [-15.435549, 5.062794] | 0.000000 | -6.217447 | -15.475906 | -111.6572648817784 | 0 | 0.000000 |
| 29_ridge | fixed_damping097 | 3.975483 | [0.903779, 5.803674] | 4.645128 | 0.000000 | 1.320277 | -26.575301140144838 | 0 | 0.000000 |
| 29_ridge | neural | 1.870192 | [-7.754228, 8.577163] | 3.830943 | -1.721814 | -13.647865 | -75.60372772159994 | 4 | 1.000000 |
| 29_ridge | pointwise | 5.202805 | [0.317951, 9.584418] | 6.137252 | 1.375384 | -9.655775 | -39.573259166837985 | 3 | 0.847559 |
| 29_ridge | previous_pointwise | 1.851374 | [-8.211587, 8.687410] | 3.905445 | -1.707403 | -13.554667 | -80.34906499379039 | 3 | 0.551840 |
| 29_neural_underharm4 | CV | 0.000000 | [0.000000, 0.000000] | 0.051508 | -4.379645 | 0.000000 | 0.0 | 0 | 0.000000 |
| 29_neural_underharm4 | training_selected_baseline | -2.659469 | [-15.435549, 5.062794] | 0.000000 | -6.217447 | -15.475906 | -111.6572648817784 | 0 | 0.000000 |
| 29_neural_underharm4 | fixed_damping097 | 3.975483 | [0.903779, 5.803674] | 4.645128 | 0.000000 | 1.320277 | -26.575301140144838 | 0 | 0.000000 |
| 29_neural_underharm4 | neural | 1.870192 | [-7.754228, 8.577163] | 3.830943 | -1.721814 | -13.647865 | -75.60372772159994 | 4 | 1.000000 |
| 29_neural_underharm4 | pointwise | 4.184817 | [2.189534, 6.266913] | 4.532360 | 0.090299 | -2.329701 | -12.53882175239398 | 1 | 0.202277 |
| 29_neural_underharm4 | previous_pointwise | 1.300299 | [-8.579932, 7.732449] | 3.365695 | -2.285760 | -12.806931 | -80.75703369649962 | 0 | 0.071609 |
| 43_ridge | CV | 0.000000 | [0.000000, 0.000000] | 0.051508 | -4.379645 | 0.000000 | 0.0 | 0 | 0.000000 |
| 43_ridge | training_selected_baseline | -2.659469 | [-15.435549, 5.062794] | 0.000000 | -6.217447 | -15.475906 | -111.6572648817784 | 0 | 0.000000 |
| 43_ridge | fixed_damping097 | 3.975483 | [0.903779, 5.803674] | 4.645128 | 0.000000 | 1.320277 | -26.575301140144838 | 0 | 0.000000 |
| 43_ridge | neural | 2.153530 | [-7.700888, 8.653290] | 4.171053 | -1.405713 | -14.393556 | -81.03588500734608 | 4 | 1.000000 |
| 43_ridge | pointwise | 5.400333 | [0.627994, 9.489104] | 6.355909 | 1.590594 | -9.771339 | -41.193654379383624 | 3 | 0.849092 |
| 43_ridge | previous_pointwise | 1.311490 | [-9.678842, 8.503426] | 3.563791 | -2.198792 | -14.382528 | -88.47003898259736 | 3 | 0.509473 |
| 43_neural_underharm4 | CV | 0.000000 | [0.000000, 0.000000] | 0.051508 | -4.379645 | 0.000000 | 0.0 | 0 | 0.000000 |
| 43_neural_underharm4 | training_selected_baseline | -2.659469 | [-15.435549, 5.062794] | 0.000000 | -6.217447 | -15.475906 | -111.6572648817784 | 0 | 0.000000 |
| 43_neural_underharm4 | fixed_damping097 | 3.975483 | [0.903779, 5.803674] | 4.645128 | 0.000000 | 1.320277 | -26.575301140144838 | 0 | 0.000000 |
| 43_neural_underharm4 | neural | 2.153530 | [-7.700888, 8.653290] | 4.171053 | -1.405713 | -14.393556 | -81.03588500734608 | 4 | 1.000000 |
| 43_neural_underharm4 | pointwise | 4.430765 | [2.376067, 6.480490] | 4.801785 | 0.356767 | -2.375716 | -11.28692902055084 | 1 | 0.213745 |
| 43_neural_underharm4 | previous_pointwise | 1.499779 | [-8.581633, 8.039586] | 3.587313 | -2.067572 | -13.258345 | -86.89796748932443 | 0 | 0.096853 |

## Matched Joint Population

6116 targets, not all 318969 forecast targets. Negative easy gain denotes degradation; observed risk is not certified risk.

| Seed/head | Rule | ADE vs CV (%) | Conditional CI | Hard vs CV (%) | Easy vs CV (%) | Worst easy gain (%) | Zero-CV harmed | Switch rate | Observed safety |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 17_ridge | CV | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.0 | 0 | 0.000000 | zero-event support absent; not certified |
| 17_ridge | training_selected_baseline | -0.838189 | [-11.702714, 5.712282] | 10.064890 | -11.269240 | -65.68402436297703 | 0 | 0.000000 | zero-event support absent; not certified |
| 17_ridge | fixed_damping097 | 4.173131 | [1.699226, 6.035544] | 7.782670 | 1.974279 | -18.747991702826393 | 0 | 0.000000 | zero-event support absent; not certified |
| 17_ridge | neural | 4.631448 | [-3.394464, 9.950885] | 14.227349 | -9.745961 | -42.94237427323322 | 0 | 1.000000 | zero-event support absent; not certified |
| 17_ridge | pointwise | 7.940205 | [5.025566, 11.042428] | 14.386880 | -6.803069 | -30.979852028891173 | 0 | 0.855625 | zero-event support absent; not certified |
| 17_ridge | independent | 2.721123 | [1.155727, 4.563501] | 4.931543 | -0.537602 | -16.323914433408437 | 0 | 0.205363 | zero-event support absent; not certified |
| 17_ridge | scene_uniform | 0.308689 | [0.013128, 0.782196] | 0.376409 | 0.534638 | 0.0 | 0 | 0.010955 | zero-event support absent; not certified |
| 17_ridge | joint | 2.719285 | [1.159437, 4.546555] | 4.934954 | -0.541830 | -16.323914433408437 | 0 | 0.205036 | zero-event support absent; not certified |
| 17_ridge | unary_exact | 2.721737 | [1.160757, 4.551205] | 4.934954 | -0.541830 | -16.323914433408437 | 0 | 0.205363 | zero-event support absent; not certified |
| 17_ridge | joint_exact | 2.721737 | [1.160757, 4.551205] | 4.934954 | -0.541830 | -16.323914433408437 | 0 | 0.205363 | zero-event support absent; not certified |
| 17_ridge | previous_pointwise | 3.635853 | [-4.535510, 9.078881] | 13.203554 | -10.264079 | -47.49680260892746 | 0 | 0.567201 | zero-event support absent; not certified |
| 17_ridge | previous_independent | -0.412809 | [-11.124319, 6.156619] | 10.365616 | -11.291487 | -64.75319450317527 | 0 | 0.275180 | zero-event support absent; not certified |
| 17_ridge | previous_scene_uniform | -0.588135 | [-11.211184, 5.896372] | 10.136455 | -11.273968 | -65.37759109111572 | 0 | 0.024035 | zero-event support absent; not certified |
| 17_ridge | previous_joint | -0.411957 | [-11.120649, 6.156520] | 10.368502 | -11.302329 | -64.75319450317527 | 0 | 0.275180 | zero-event support absent; not certified |
| 17_ridge | previous_unary_exact | -0.411800 | [-11.120496, 6.156527] | 10.368502 | -11.303878 | -64.75319450317527 | 0 | 0.275180 | zero-event support absent; not certified |
| 17_ridge | previous_joint_exact | -0.411660 | [-11.120360, 6.156534] | 10.368502 | -11.302329 | -64.75319450317527 | 0 | 0.275180 | zero-event support absent; not certified |
| 17_neural_underharm4 | CV | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.0 | 0 | 0.000000 | zero-event support absent; not certified |
| 17_neural_underharm4 | training_selected_baseline | -0.838189 | [-11.702714, 5.712282] | 10.064890 | -11.269240 | -65.68402436297703 | 0 | 0.000000 | zero-event support absent; not certified |
| 17_neural_underharm4 | fixed_damping097 | 4.173131 | [1.699226, 6.035544] | 7.782670 | 1.974279 | -18.747991702826393 | 0 | 0.000000 | zero-event support absent; not certified |
| 17_neural_underharm4 | neural | 4.631448 | [-3.394464, 9.950885] | 14.227349 | -9.745961 | -42.94237427323322 | 0 | 1.000000 | zero-event support absent; not certified |
| 17_neural_underharm4 | pointwise | 4.523357 | [2.725180, 6.563311] | 9.693537 | -2.253258 | -13.831078352859416 | 0 | 0.282701 | zero-event support absent; not certified |
| 17_neural_underharm4 | independent | 1.050906 | [0.573523, 1.601443] | 1.461560 | -1.095811 | -5.168955356175942 | 0 | 0.080445 | zero-event support absent; not certified |
| 17_neural_underharm4 | scene_uniform | 0.067765 | [-0.000441, 0.138184] | 0.078317 | 0.128634 | -0.3026758149820008 | 0 | 0.005069 | zero-event support absent; not certified |
| 17_neural_underharm4 | joint | 1.056708 | [0.574183, 1.610700] | 1.463165 | -1.095811 | -5.168955356175942 | 0 | 0.080445 | zero-event support absent; not certified |
| 17_neural_underharm4 | unary_exact | 1.052208 | [0.574183, 1.601574] | 1.463165 | -1.095811 | -5.168955356175942 | 0 | 0.080445 | zero-event support absent; not certified |
| 17_neural_underharm4 | joint_exact | 1.056708 | [0.574183, 1.610700] | 1.463165 | -1.095811 | -5.168955356175942 | 0 | 0.080445 | zero-event support absent; not certified |
| 17_neural_underharm4 | previous_pointwise | 2.666831 | [-5.714764, 8.083903] | 12.513664 | -9.387955 | -45.045407445512 | 0 | 0.177894 | zero-event support absent; not certified |
| 17_neural_underharm4 | previous_independent | -0.511647 | [-11.273971, 5.979236] | 10.217516 | -10.903333 | -63.32244856687035 | 0 | 0.118705 | zero-event support absent; not certified |
| 17_neural_underharm4 | previous_scene_uniform | -0.818107 | [-11.674007, 5.728594] | 10.085632 | -11.246661 | -65.4310135025843 | 0 | 0.013898 | zero-event support absent; not certified |
| 17_neural_underharm4 | previous_joint | -0.511388 | [-11.273640, 5.979416] | 10.217516 | -10.903333 | -63.32244856687035 | 0 | 0.118215 | zero-event support absent; not certified |
| 17_neural_underharm4 | previous_unary_exact | -0.511553 | [-11.273877, 5.979331] | 10.216909 | -10.903333 | -63.32244856687035 | 0 | 0.118705 | zero-event support absent; not certified |
| 17_neural_underharm4 | previous_joint_exact | -0.511647 | [-11.273971, 5.979236] | 10.217516 | -10.903333 | -63.32244856687035 | 0 | 0.118705 | zero-event support absent; not certified |
| 29_ridge | CV | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.0 | 0 | 0.000000 | zero-event support absent; not certified |
| 29_ridge | training_selected_baseline | -0.838189 | [-11.702714, 5.712282] | 10.064890 | -11.269240 | -65.68402436297703 | 0 | 0.000000 | zero-event support absent; not certified |
| 29_ridge | fixed_damping097 | 4.173131 | [1.699226, 6.035544] | 7.782670 | 1.974279 | -18.747991702826393 | 0 | 0.000000 | zero-event support absent; not certified |
| 29_ridge | neural | 4.397232 | [-3.484813, 9.731192] | 14.117737 | -9.417432 | -40.930485788545504 | 0 | 1.000000 | zero-event support absent; not certified |
| 29_ridge | pointwise | 7.701344 | [4.599720, 10.900904] | 14.521226 | -6.828022 | -31.841953325021176 | 0 | 0.849902 | zero-event support absent; not certified |
| 29_ridge | independent | 2.436928 | [1.070107, 4.038525] | 4.433606 | -1.506164 | -16.586446623531238 | 0 | 0.189339 | zero-event support absent; not certified |
| 29_ridge | scene_uniform | 0.276441 | [-0.022968, 0.753079] | 0.336367 | 0.496757 | 0.0 | 0 | 0.010791 | zero-event support absent; not certified |
| 29_ridge | joint | 2.427843 | [1.070227, 4.023028] | 4.421938 | -1.591105 | -16.586446623531238 | 0 | 0.188849 | zero-event support absent; not certified |
| 29_ridge | unary_exact | 2.439071 | [1.076173, 4.037253] | 4.432691 | -1.587206 | -16.586446623531238 | 0 | 0.189339 | zero-event support absent; not certified |
| 29_ridge | joint_exact | 2.439071 | [1.076173, 4.037253] | 4.432691 | -1.587206 | -16.586446623531238 | 0 | 0.189339 | zero-event support absent; not certified |
| 29_ridge | previous_pointwise | 3.564606 | [-4.343109, 8.916706] | 13.141185 | -9.970227 | -46.02183871767531 | 0 | 0.574559 | zero-event support absent; not certified |
| 29_ridge | previous_independent | -0.447359 | [-11.105124, 6.031447] | 10.296692 | -11.468326 | -64.67992356868042 | 0 | 0.278940 | zero-event support absent; not certified |
| 29_ridge | previous_scene_uniform | -0.597060 | [-11.198779, 5.871832] | 10.088617 | -11.254809 | -65.26953240542872 | 0 | 0.020438 | zero-event support absent; not certified |
| 29_ridge | previous_joint | -0.485738 | [-11.125523, 5.981740] | 10.238855 | -11.490925 | -64.67992356868042 | 0 | 0.278286 | zero-event support absent; not certified |
| 29_ridge | previous_unary_exact | -0.493362 | [-11.130841, 5.965187] | 10.236410 | -11.490925 | -64.67992356868042 | 0 | 0.278940 | zero-event support absent; not certified |
| 29_ridge | previous_joint_exact | -0.491922 | [-11.129401, 5.965223] | 10.238855 | -11.490925 | -64.67992356868042 | 0 | 0.278940 | zero-event support absent; not certified |
| 29_neural_underharm4 | CV | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.0 | 0 | 0.000000 | zero-event support absent; not certified |
| 29_neural_underharm4 | training_selected_baseline | -0.838189 | [-11.702714, 5.712282] | 10.064890 | -11.269240 | -65.68402436297703 | 0 | 0.000000 | zero-event support absent; not certified |
| 29_neural_underharm4 | fixed_damping097 | 4.173131 | [1.699226, 6.035544] | 7.782670 | 1.974279 | -18.747991702826393 | 0 | 0.000000 | zero-event support absent; not certified |
| 29_neural_underharm4 | neural | 4.397232 | [-3.484813, 9.731192] | 14.117737 | -9.417432 | -40.930485788545504 | 0 | 1.000000 | zero-event support absent; not certified |
| 29_neural_underharm4 | pointwise | 4.442355 | [2.513705, 6.595407] | 10.169730 | -1.960606 | -15.452817653974504 | 0 | 0.163342 | zero-event support absent; not certified |
| 29_neural_underharm4 | independent | 0.932395 | [0.311780, 1.730165] | 1.299264 | -0.761503 | -6.012406123656744 | 0 | 0.052812 | zero-event support absent; not certified |
| 29_neural_underharm4 | scene_uniform | 0.066472 | [-0.017769, 0.185960] | 0.079609 | 0.051279 | 0.0 | 0 | 0.003761 | zero-event support absent; not certified |
| 29_neural_underharm4 | joint | 0.932340 | [0.310811, 1.732016] | 1.299264 | -0.759127 | -6.012406123656744 | 0 | 0.052812 | zero-event support absent; not certified |
| 29_neural_underharm4 | unary_exact | 0.933359 | [0.311804, 1.733034] | 1.299264 | -0.761503 | -6.012406123656744 | 0 | 0.052812 | zero-event support absent; not certified |
| 29_neural_underharm4 | joint_exact | 0.932340 | [0.310811, 1.732016] | 1.299264 | -0.759127 | -6.012406123656744 | 0 | 0.052812 | zero-event support absent; not certified |
| 29_neural_underharm4 | previous_pointwise | 2.800045 | [-5.076113, 7.942230] | 12.421977 | -9.381553 | -42.84964521232722 | 0 | 0.078646 | zero-event support absent; not certified |
| 29_neural_underharm4 | previous_independent | -0.293910 | [-10.786134, 6.084421] | 10.413502 | -10.808390 | -60.15727600921193 | 0 | 0.038751 | zero-event support absent; not certified |
| 29_neural_underharm4 | previous_scene_uniform | -0.739565 | [-11.466641, 5.741019] | 10.102666 | -11.236781 | -65.33094279694028 | 0 | 0.013571 | zero-event support absent; not certified |
| 29_neural_underharm4 | previous_joint | -0.293910 | [-10.786134, 6.084421] | 10.413502 | -10.808390 | -60.15727600921193 | 0 | 0.038751 | zero-event support absent; not certified |
| 29_neural_underharm4 | previous_unary_exact | -0.293910 | [-10.786134, 6.084421] | 10.413502 | -10.808390 | -60.15727600921193 | 0 | 0.038751 | zero-event support absent; not certified |
| 29_neural_underharm4 | previous_joint_exact | -0.293910 | [-10.786134, 6.084421] | 10.413502 | -10.808390 | -60.15727600921193 | 0 | 0.038751 | zero-event support absent; not certified |
| 43_ridge | CV | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.0 | 0 | 0.000000 | zero-event support absent; not certified |
| 43_ridge | training_selected_baseline | -0.838189 | [-11.702714, 5.712282] | 10.064890 | -11.269240 | -65.68402436297703 | 0 | 0.000000 | zero-event support absent; not certified |
| 43_ridge | fixed_damping097 | 4.173131 | [1.699226, 6.035544] | 7.782670 | 1.974279 | -18.747991702826393 | 0 | 0.000000 | zero-event support absent; not certified |
| 43_ridge | neural | 4.398923 | [-3.973090, 9.869706] | 14.239154 | -10.251424 | -46.557902236303384 | 0 | 1.000000 | zero-event support absent; not certified |
| 43_ridge | pointwise | 7.654364 | [4.788685, 10.587706] | 14.201465 | -5.821945 | -29.088456846053013 | 0 | 0.852027 | zero-event support absent; not certified |
| 43_ridge | independent | 2.766982 | [1.421707, 4.317564] | 4.744560 | -0.104728 | -5.117382124785408 | 0 | 0.189666 | zero-event support absent; not certified |
| 43_ridge | scene_uniform | 0.241220 | [-0.034160, 0.663358] | 0.310189 | 0.471834 | 0.0 | 0 | 0.010301 | zero-event support absent; not certified |
| 43_ridge | joint | 2.760702 | [1.417063, 4.317489] | 4.733475 | -0.120570 | -5.117382124785408 | 0 | 0.189012 | zero-event support absent; not certified |
| 43_ridge | unary_exact | 2.768602 | [1.424185, 4.316559] | 4.747844 | -0.098242 | -5.117382124785408 | 0 | 0.189666 | zero-event support absent; not certified |
| 43_ridge | joint_exact | 2.770515 | [1.424667, 4.317734] | 4.747844 | -0.102162 | -5.117382124785408 | 0 | 0.189666 | zero-event support absent; not certified |
| 43_ridge | previous_pointwise | 3.246483 | [-5.508045, 8.987363] | 13.225357 | -9.891523 | -46.364108273848736 | 0 | 0.536952 | zero-event support absent; not certified |
| 43_ridge | previous_independent | -0.445238 | [-11.256610, 6.116909] | 10.368157 | -11.210681 | -63.45676178179309 | 0 | 0.250818 | zero-event support absent; not certified |
| 43_ridge | previous_scene_uniform | -0.717044 | [-11.588549, 5.868844] | 10.142137 | -11.246473 | -65.49701794052109 | 0 | 0.015697 | zero-event support absent; not certified |
| 43_ridge | previous_joint | -0.493840 | [-11.292284, 6.029302] | 10.302590 | -11.216254 | -63.45676178179309 | 0 | 0.250491 | zero-event support absent; not certified |
| 43_ridge | previous_unary_exact | -0.497796 | [-11.292454, 6.022362] | 10.302537 | -11.216232 | -63.45676178179309 | 0 | 0.250818 | zero-event support absent; not certified |
| 43_ridge | previous_joint_exact | -0.494796 | [-11.292369, 6.025993] | 10.306326 | -11.216254 | -63.45676178179309 | 0 | 0.250818 | zero-event support absent; not certified |
| 43_neural_underharm4 | CV | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000000 | 0.0 | 0 | 0.000000 | zero-event support absent; not certified |
| 43_neural_underharm4 | training_selected_baseline | -0.838189 | [-11.702714, 5.712282] | 10.064890 | -11.269240 | -65.68402436297703 | 0 | 0.000000 | zero-event support absent; not certified |
| 43_neural_underharm4 | fixed_damping097 | 4.173131 | [1.699226, 6.035544] | 7.782670 | 1.974279 | -18.747991702826393 | 0 | 0.000000 | zero-event support absent; not certified |
| 43_neural_underharm4 | neural | 4.398923 | [-3.973090, 9.869706] | 14.239154 | -10.251424 | -46.557902236303384 | 0 | 1.000000 | zero-event support absent; not certified |
| 43_neural_underharm4 | pointwise | 4.217150 | [1.973090, 6.475980] | 8.647163 | -1.434678 | -14.532147888318846 | 0 | 0.187868 | zero-event support absent; not certified |
| 43_neural_underharm4 | independent | 0.599705 | [0.292701, 0.926339] | 0.930303 | -0.174272 | -1.647274911769947 | 0 | 0.042511 | zero-event support absent; not certified |
| 43_neural_underharm4 | scene_uniform | 0.026612 | [0.003339, 0.062699] | 0.070635 | 0.050451 | -0.30272383439131634 | 0 | 0.002453 | zero-event support absent; not certified |
| 43_neural_underharm4 | joint | 0.593634 | [0.286773, 0.919505] | 0.922830 | -0.174272 | -1.647274911769947 | 0 | 0.042511 | zero-event support absent; not certified |
| 43_neural_underharm4 | unary_exact | 0.593634 | [0.286773, 0.919505] | 0.922830 | -0.174272 | -1.647274911769947 | 0 | 0.042511 | zero-event support absent; not certified |
| 43_neural_underharm4 | joint_exact | 0.593634 | [0.286773, 0.919505] | 0.922830 | -0.174272 | -1.647274911769947 | 0 | 0.042511 | zero-event support absent; not certified |
| 43_neural_underharm4 | previous_pointwise | 2.983650 | [-5.539663, 8.536094] | 12.829508 | -10.171589 | -52.82188267843291 | 0 | 0.116089 | zero-event support absent; not certified |
| 43_neural_underharm4 | previous_independent | -0.202714 | [-10.426054, 6.054777] | 10.316102 | -11.103532 | -63.321398377396896 | 0 | 0.054120 | zero-event support absent; not certified |
| 43_neural_underharm4 | previous_scene_uniform | -0.491120 | [-10.732590, 5.742376] | 10.109487 | -11.068111 | -63.321398377396896 | 0 | 0.020111 | zero-event support absent; not certified |
| 43_neural_underharm4 | previous_joint | -0.202655 | [-10.426054, 6.054951] | 10.316102 | -11.103532 | -63.321398377396896 | 0 | 0.053957 | zero-event support absent; not certified |
| 43_neural_underharm4 | previous_unary_exact | -0.202714 | [-10.426054, 6.054777] | 10.316102 | -11.103532 | -63.321398377396896 | 0 | 0.054120 | zero-event support absent; not certified |
| 43_neural_underharm4 | previous_joint_exact | -0.202714 | [-10.426054, 6.054777] | 10.316102 | -11.103532 | -63.321398377396896 | 0 | 0.054120 | zero-event support absent; not certified |

## Direct Paired Contrasts

| Seed/head | Contrast | ADE gain (%) | Conditional CI |
|---|---|---:|---|
| 17_ridge | pointwise_vs_previous | 2.800407 | [0.448607, 6.844699] |
| 17_ridge | joint_vs_previous | 1.297630 | [-3.476634, 8.224451] |
| 17_ridge | joint_vs_independent | -0.002266 | [-0.024955, 0.016049] |
| 17_ridge | joint_exact_vs_independent | 0.003600 | [-0.018721, 0.024976] |
| 17_ridge | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 17_neural_underharm4 | pointwise_vs_previous | 1.533021 | [-3.230973, 8.035868] |
| 17_neural_underharm4 | joint_vs_previous | -0.418975 | [-5.252295, 6.815635] |
| 17_neural_underharm4 | joint_vs_independent | 0.005966 | [0.000000, 0.016570] |
| 17_neural_underharm4 | joint_exact_vs_independent | 0.011440 | [0.000000, 0.032252] |
| 17_neural_underharm4 | joint_exact_vs_unary | 0.009373 | [0.000000, 0.028118] |
| 29_ridge | pointwise_vs_previous | 2.495813 | [0.198026, 6.326876] |
| 29_ridge | joint_vs_previous | 1.093713 | [-3.556366, 7.952742] |
| 29_ridge | joint_vs_independent | -0.009705 | [-0.028616, 0.006809] |
| 29_ridge | joint_exact_vs_independent | 0.003705 | [-0.004532, 0.015647] |
| 29_ridge | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 29_neural_underharm4 | pointwise_vs_previous | 1.391177 | [-3.175742, 7.627105] |
| 29_neural_underharm4 | joint_vs_previous | -0.699342 | [-5.534639, 6.472074] |
| 29_neural_underharm4 | joint_vs_independent | -0.000044 | [-0.003106, 0.002975] |
| 29_neural_underharm4 | joint_exact_vs_independent | undefined | undefined |
| 29_neural_underharm4 | joint_exact_vs_unary | undefined | undefined |
| 43_ridge | pointwise_vs_previous | 3.004155 | [0.582289, 7.178618] |
| 43_ridge | joint_vs_previous | 1.284243 | [-3.650434, 8.621910] |
| 43_ridge | joint_vs_independent | -0.006632 | [-0.030916, 0.010121] |
| 43_ridge | joint_exact_vs_independent | 0.005101 | [-0.000696, 0.015367] |
| 43_ridge | joint_exact_vs_unary | 0.002414 | [0.000000, 0.007243] |
| 43_neural_underharm4 | pointwise_vs_previous | 1.428912 | [-2.965286, 7.624402] |
| 43_neural_underharm4 | joint_vs_previous | -1.053736 | [-5.699911, 5.946520] |
| 43_neural_underharm4 | joint_vs_independent | -0.006125 | [-0.018374, 0.000000] |
| 43_neural_underharm4 | joint_exact_vs_independent | undefined | undefined |
| 43_neural_underharm4 | joint_exact_vs_unary | undefined | undefined |

A change of fallback also changes the cost labels, causal disagreement features and fitting cost scale; these necessary derived changes are part of one reference-contract repair.
Matching is within the new policy family. Old/new reference comparisons are on the same population but need not share intervention rates.
No post-readout winner, threshold or checkpoint selection. No change to deployment, risk limits, held roles, metric/time claims, Stage5C or SMC.
