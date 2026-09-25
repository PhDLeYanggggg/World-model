# Symmetric Risk Results

36 newly trained Torch risk heads / 72,000 updates. All 18 utility heads and all trajectories frozen.
Ridge risk and decision banks are cached_verified; metrics are recomputed, not called fresh optimization.
All 48 registered views retained; no post-readout selection. Source development only, not reserved evaluation.

## Full Population

| Seed/candidate/event/risk/guard | ADE gain vs CV (%) | Conditional CI | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harmed | Switch (%) |
|---|---:|---|---:|---:|---:|---:|---:|
| 17_neural_all_ridge_no_guard | 0.614643 | [0.257613, 1.050919] | 0.799861 | 0.821433 | 0.706620 | 0/4 | 4.715819 |
| 17_neural_all_ridge_source_zero_guard | 0.284229 | [0.090529, 0.522413] | 0.338124 | 0.293842 | 0.364058 | 0/4 | 0.790359 |
| 17_neural_all_neural_mse_no_guard | 0.272965 | [0.090958, 0.504689] | 0.360183 | 0.331099 | 0.031556 | 0/4 | 0.337337 |
| 17_neural_all_neural_mse_source_zero_guard | 0.074696 | [0.011934, 0.150599] | 0.112277 | 0.088270 | 0.014070 | 0/4 | 0.141080 |
| 17_neural_easy_ridge_no_guard | 0.306160 | [0.022303, 0.531804] | 0.479299 | 0.356428 | 0.021378 | 1/4 | 5.556653 |
| 17_neural_easy_ridge_source_zero_guard | 0.317049 | [0.155231, 0.504554] | 0.418103 | 0.393313 | -0.000000 | 0/4 | 1.231468 |
| 17_neural_easy_neural_mse_no_guard | 1.134124 | [0.646143, 1.646515] | 1.582187 | 1.208197 | 2.464437 | 0/4 | 12.885265 |
| 17_neural_easy_neural_mse_source_zero_guard | 1.046956 | [0.509299, 1.602645] | 1.444029 | 1.185508 | 2.464437 | 0/4 | 7.321088 |
| 17_damping097_all_ridge_no_guard | 1.795341 | [0.921602, 2.731372] | 2.867230 | 1.910761 | 1.557245 | 0/4 | 34.385160 |
| 17_damping097_all_ridge_source_zero_guard | 0.745799 | [0.300983, 1.250658] | 1.153074 | 0.761188 | 0.669123 | 0/4 | 8.230894 |
| 17_damping097_all_neural_mse_no_guard | 2.079589 | [1.280344, 2.913720] | 3.355021 | 2.526276 | 0.363286 | 0/4 | 35.751123 |
| 17_damping097_all_neural_mse_source_zero_guard | 1.037333 | [0.489104, 1.672107] | 1.655556 | 1.219764 | 0.345673 | 0/4 | 4.437422 |
| 17_damping097_easy_ridge_no_guard | 1.679480 | [1.178135, 2.186394] | 2.608061 | 1.858853 | 2.490134 | 0/4 | 43.269722 |
| 17_damping097_easy_ridge_source_zero_guard | 1.184321 | [0.558671, 1.846962] | 1.874736 | 1.174761 | -0.000000 | 0/4 | 15.144105 |
| 17_damping097_easy_neural_mse_no_guard | 1.671067 | [1.042272, 2.342275] | 2.675813 | 1.291957 | -0.581822 | 0/4 | 50.515567 |
| 17_damping097_easy_neural_mse_source_zero_guard | 1.437949 | [0.689822, 2.209592] | 2.304340 | 1.221830 | -0.000000 | 0/4 | 21.068192 |
| 29_neural_all_ridge_no_guard | 0.536012 | [0.239498, 0.906554] | 0.678542 | 0.702577 | 0.864991 | 0/4 | 3.168019 |
| 29_neural_all_ridge_source_zero_guard | 0.269827 | [0.091078, 0.511058] | 0.339164 | 0.285503 | 0.359525 | 0/4 | 0.616674 |
| 29_neural_all_neural_mse_no_guard | 0.254410 | [0.088194, 0.471819] | 0.340661 | 0.323501 | 0.097637 | 0/4 | 0.704771 |
| 29_neural_all_neural_mse_source_zero_guard | 0.066977 | [0.016024, 0.127882] | 0.089302 | 0.076632 | 0.039666 | 0/4 | 0.190928 |
| 29_neural_easy_ridge_no_guard | 0.354291 | [0.187834, 0.508931] | 0.500665 | 0.391013 | -0.000000 | 1/4 | 2.396471 |
| 29_neural_easy_ridge_source_zero_guard | 0.313034 | [0.158034, 0.471071] | 0.408246 | 0.383731 | -0.000000 | 0/4 | 0.908552 |
| 29_neural_easy_neural_mse_no_guard | 1.581042 | [0.694957, 2.806370] | 2.248684 | 1.792688 | 7.289078 | 1/4 | 5.736294 |
| 29_neural_easy_neural_mse_source_zero_guard | 1.538415 | [0.619488, 2.766722] | 2.177973 | 1.782651 | 7.289078 | 0/4 | 3.717603 |
| 29_damping097_all_ridge_no_guard | 1.787827 | [0.913550, 2.723597] | 2.857729 | 1.907625 | 1.557513 | 0/4 | 33.445256 |
| 29_damping097_all_ridge_source_zero_guard | 0.742949 | [0.298167, 1.246077] | 1.148721 | 0.762390 | 0.747042 | 0/4 | 7.739310 |
| 29_damping097_all_neural_mse_no_guard | 2.052666 | [1.204481, 2.954766] | 3.314798 | 2.462924 | 1.764583 | 0/4 | 33.701081 |
| 29_damping097_all_neural_mse_source_zero_guard | 1.091229 | [0.488336, 1.816315] | 1.736842 | 1.267852 | 0.208399 | 0/4 | 6.189003 |
| 29_damping097_easy_ridge_no_guard | 1.691829 | [1.203251, 2.180982] | 2.636548 | 1.773100 | 2.743480 | 0/4 | 40.047465 |
| 29_damping097_easy_ridge_source_zero_guard | 1.177475 | [0.560219, 1.833193] | 1.864692 | 1.191775 | -0.000000 | 0/4 | 13.567776 |
| 29_damping097_easy_neural_mse_no_guard | 1.150926 | [0.773275, 1.558488] | 1.826306 | 0.771757 | -0.405580 | 0/4 | 45.581546 |
| 29_damping097_easy_neural_mse_source_zero_guard | 0.898015 | [0.418708, 1.403232] | 1.423866 | 0.687937 | -0.000000 | 0/4 | 16.638294 |
| 43_neural_all_ridge_no_guard | 0.612419 | [0.278765, 1.076380] | 0.826258 | 0.748689 | 0.645820 | 0/4 | 3.468049 |
| 43_neural_all_ridge_source_zero_guard | 0.231117 | [0.091109, 0.409638] | 0.309132 | 0.248583 | 0.096005 | 0/4 | 0.692230 |
| 43_neural_all_neural_mse_no_guard | 0.648975 | [0.153614, 1.305400] | 0.911776 | 0.867514 | 0.240295 | 0/4 | 1.673517 |
| 43_neural_all_neural_mse_source_zero_guard | 0.129547 | [0.032258, 0.275802] | 0.189503 | 0.160553 | 0.114920 | 0/4 | 0.283727 |
| 43_neural_easy_ridge_no_guard | 0.425078 | [0.222930, 0.633949] | 0.650229 | 0.472027 | -0.000000 | 0/4 | 3.629820 |
| 43_neural_easy_ridge_source_zero_guard | 0.385866 | [0.185142, 0.610147] | 0.560192 | 0.471993 | -0.000000 | 0/4 | 0.975643 |
| 43_neural_easy_neural_mse_no_guard | 2.090101 | [0.602831, 4.024609] | 3.013263 | 2.357885 | 16.713521 | 0/4 | 7.838379 |
| 43_neural_easy_neural_mse_source_zero_guard | 2.048412 | [0.544242, 3.996223] | 2.943527 | 2.351350 | 16.713521 | 0/4 | 4.965686 |
| 43_damping097_all_ridge_no_guard | 1.795469 | [0.920990, 2.727575] | 2.869113 | 1.910543 | 1.557885 | 0/4 | 34.421840 |
| 43_damping097_all_ridge_source_zero_guard | 0.746792 | [0.301050, 1.252742] | 1.154966 | 0.761211 | 0.705978 | 0/4 | 8.386708 |
| 43_damping097_all_neural_mse_no_guard | 2.414229 | [1.546270, 3.282675] | 3.825677 | 2.934448 | 1.803671 | 0/4 | 29.270556 |
| 43_damping097_all_neural_mse_source_zero_guard | 1.385134 | [0.631490, 2.233934] | 2.185562 | 1.517973 | 0.460068 | 0/4 | 8.414924 |
| 43_damping097_easy_ridge_no_guard | 1.733060 | [1.259870, 2.215188] | 2.686310 | 1.844276 | 2.787236 | 0/4 | 42.983801 |
| 43_damping097_easy_ridge_source_zero_guard | 1.202642 | [0.573103, 1.858810] | 1.893249 | 1.213341 | -0.000000 | 0/4 | 15.295217 |
| 43_damping097_easy_neural_mse_no_guard | 1.872286 | [1.173363, 2.622665] | 3.014657 | 1.474072 | -0.640980 | 0/4 | 51.063896 |
| 43_damping097_easy_neural_mse_source_zero_guard | 1.636275 | [0.809311, 2.513824] | 2.637777 | 1.404455 | -0.000000 | 0/4 | 21.931598 |

## New Risk vs Frozen Old Risk

| View | Population/rule | Paired gain (%) | Conditional CI |
|---|---|---:|---|
| 17_neural_all_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_neural_all_neural_mse_no_guard | pointwise | 0.125517 | [0.010897, 0.259988] |
| 17_neural_all_neural_mse_no_guard | independent | 1.911163 | [1.030865, 2.974724] |
| 17_neural_all_neural_mse_no_guard | scene_uniform | -0.006447 | [-0.028652, 0.009611] |
| 17_neural_all_neural_mse_no_guard | joint | 1.894389 | [1.031611, 2.938538] |
| 17_neural_all_neural_mse_no_guard | unary_exact | 1.910076 | [1.036653, 2.978634] |
| 17_neural_all_neural_mse_no_guard | joint_exact | 1.894389 | [1.031611, 2.938538] |
| 17_neural_all_neural_mse_no_guard | pointwise_hard | 0.183349 | [0.053078, 0.347958] |
| 17_neural_all_neural_mse_no_guard | pointwise_easy | -0.030852 | [-0.084218, 0.017859] |
| 17_neural_all_neural_mse_source_zero_guard | pointwise | 0.058710 | [0.006986, 0.132432] |
| 17_neural_all_neural_mse_source_zero_guard | independent | 0.785599 | [0.318755, 1.300597] |
| 17_neural_all_neural_mse_source_zero_guard | scene_uniform | 0.003194 | [-0.000993, 0.010574] |
| 17_neural_all_neural_mse_source_zero_guard | joint | 0.782222 | [0.313912, 1.301415] |
| 17_neural_all_neural_mse_source_zero_guard | unary_exact | 0.784689 | [0.317052, 1.301415] |
| 17_neural_all_neural_mse_source_zero_guard | joint_exact | 0.782222 | [0.313912, 1.301415] |
| 17_neural_all_neural_mse_source_zero_guard | pointwise_hard | 0.071853 | [0.001494, 0.161981] |
| 17_neural_all_neural_mse_source_zero_guard | pointwise_easy | -0.000194 | [-0.027323, 0.028053] |
| 17_neural_easy_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_neural_easy_neural_mse_no_guard | pointwise | 0.865464 | [0.451508, 1.315839] |
| 17_neural_easy_neural_mse_no_guard | independent | 2.926386 | [1.532896, 4.527947] |
| 17_neural_easy_neural_mse_no_guard | scene_uniform | 0.024285 | [-0.004858, 0.060662] |
| 17_neural_easy_neural_mse_no_guard | joint | 2.945726 | [1.562356, 4.519954] |
| 17_neural_easy_neural_mse_no_guard | unary_exact | 2.954943 | [1.574889, 4.531408] |
| 17_neural_easy_neural_mse_no_guard | joint_exact | 2.946139 | [1.562366, 4.521599] |
| 17_neural_easy_neural_mse_no_guard | pointwise_hard | 0.940271 | [0.435709, 1.501755] |
| 17_neural_easy_neural_mse_no_guard | pointwise_easy | 0.237607 | [-0.556534, 0.984626] |
| 17_neural_easy_neural_mse_source_zero_guard | pointwise | 0.814418 | [0.377917, 1.290196] |
| 17_neural_easy_neural_mse_source_zero_guard | independent | 2.585976 | [1.078600, 4.297657] |
| 17_neural_easy_neural_mse_source_zero_guard | scene_uniform | 0.024285 | [-0.004858, 0.060662] |
| 17_neural_easy_neural_mse_source_zero_guard | joint | 2.577436 | [1.077515, 4.286623] |
| 17_neural_easy_neural_mse_source_zero_guard | unary_exact | 2.578725 | [1.077623, 4.288805] |
| 17_neural_easy_neural_mse_source_zero_guard | joint_exact | 2.577850 | [1.077623, 4.287055] |
| 17_neural_easy_neural_mse_source_zero_guard | pointwise_hard | 0.922099 | [0.406613, 1.488243] |
| 17_neural_easy_neural_mse_source_zero_guard | pointwise_easy | -0.035606 | [-0.699567, 0.616446] |
| 17_damping097_all_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_all_neural_mse_no_guard | pointwise | 1.424217 | [0.916746, 2.004775] |
| 17_damping097_all_neural_mse_no_guard | independent | 2.003123 | [1.541572, 2.461797] |
| 17_damping097_all_neural_mse_no_guard | scene_uniform | 0.633734 | [0.198633, 1.186230] |
| 17_damping097_all_neural_mse_no_guard | joint | 1.998609 | [1.544084, 2.446121] |
| 17_damping097_all_neural_mse_no_guard | unary_exact | 1.996106 | [1.539549, 2.445066] |
| 17_damping097_all_neural_mse_no_guard | joint_exact | 2.002339 | [1.543367, 2.450570] |
| 17_damping097_all_neural_mse_no_guard | pointwise_hard | 1.695921 | [1.100445, 2.424755] |
| 17_damping097_all_neural_mse_no_guard | pointwise_easy | 0.225063 | [-0.043321, 0.572030] |
| 17_damping097_all_neural_mse_source_zero_guard | pointwise | 0.803558 | [0.373420, 1.305336] |
| 17_damping097_all_neural_mse_source_zero_guard | independent | 1.439840 | [0.792403, 2.086790] |
| 17_damping097_all_neural_mse_source_zero_guard | scene_uniform | 0.135358 | [0.025185, 0.272366] |
| 17_damping097_all_neural_mse_source_zero_guard | joint | 1.434178 | [0.789796, 2.072464] |
| 17_damping097_all_neural_mse_source_zero_guard | unary_exact | 1.432352 | [0.788186, 2.071714] |
| 17_damping097_all_neural_mse_source_zero_guard | joint_exact | 1.437745 | [0.792396, 2.077310] |
| 17_damping097_all_neural_mse_source_zero_guard | pointwise_hard | 0.949727 | [0.442516, 1.541903] |
| 17_damping097_all_neural_mse_source_zero_guard | pointwise_easy | 0.011886 | [-0.058505, 0.070846] |
| 17_damping097_easy_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_neural_mse_no_guard | pointwise | -0.441931 | [-0.943968, 0.062317] |
| 17_damping097_easy_neural_mse_no_guard | independent | 0.353379 | [-0.052597, 0.772335] |
| 17_damping097_easy_neural_mse_no_guard | scene_uniform | 0.082634 | [-0.110947, 0.309923] |
| 17_damping097_easy_neural_mse_no_guard | joint | 0.350922 | [-0.052128, 0.767840] |
| 17_damping097_easy_neural_mse_no_guard | unary_exact | 0.351455 | [-0.051655, 0.772026] |
| 17_damping097_easy_neural_mse_no_guard | joint_exact | 0.352127 | [-0.051212, 0.771672] |
| 17_damping097_easy_neural_mse_no_guard | pointwise_hard | -0.750271 | [-1.269080, -0.259841] |
| 17_damping097_easy_neural_mse_no_guard | pointwise_easy | 1.080667 | [0.285323, 1.994076] |
| 17_damping097_easy_neural_mse_source_zero_guard | pointwise | -0.646578 | [-1.015037, -0.307972] |
| 17_damping097_easy_neural_mse_source_zero_guard | independent | -0.070669 | [-0.256986, 0.078479] |
| 17_damping097_easy_neural_mse_source_zero_guard | scene_uniform | -0.084475 | [-0.170808, -0.013264] |
| 17_damping097_easy_neural_mse_source_zero_guard | joint | -0.069922 | [-0.255738, 0.079420] |
| 17_damping097_easy_neural_mse_source_zero_guard | unary_exact | -0.070393 | [-0.256676, 0.078755] |
| 17_damping097_easy_neural_mse_source_zero_guard | joint_exact | -0.069922 | [-0.255738, 0.079420] |
| 17_damping097_easy_neural_mse_source_zero_guard | pointwise_hard | -0.817494 | [-1.302583, -0.382212] |
| 17_damping097_easy_neural_mse_source_zero_guard | pointwise_easy | 0.124115 | [0.037650, 0.211844] |
| 29_neural_all_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_neural_all_neural_mse_no_guard | pointwise | 0.207742 | [0.047076, 0.420856] |
| 29_neural_all_neural_mse_no_guard | independent | 1.736319 | [1.083375, 2.404017] |
| 29_neural_all_neural_mse_no_guard | scene_uniform | -0.010669 | [-0.030900, 0.000000] |
| 29_neural_all_neural_mse_no_guard | joint | 1.750599 | [1.091588, 2.415851] |
| 29_neural_all_neural_mse_no_guard | unary_exact | 1.751068 | [1.088288, 2.420053] |
| 29_neural_all_neural_mse_no_guard | joint_exact | 1.754013 | [1.091579, 2.424532] |
| 29_neural_all_neural_mse_no_guard | pointwise_hard | 0.287637 | [0.076424, 0.546111] |
| 29_neural_all_neural_mse_no_guard | pointwise_easy | -0.017648 | [-0.036791, -0.002173] |
| 29_neural_all_neural_mse_source_zero_guard | pointwise | 0.049121 | [0.004040, 0.099843] |
| 29_neural_all_neural_mse_source_zero_guard | independent | 0.824710 | [0.295484, 1.389169] |
| 29_neural_all_neural_mse_source_zero_guard | scene_uniform | -0.001106 | [-0.003318, 0.000000] |
| 29_neural_all_neural_mse_source_zero_guard | joint | 0.822525 | [0.292308, 1.393057] |
| 29_neural_all_neural_mse_source_zero_guard | unary_exact | 0.822525 | [0.292308, 1.393057] |
| 29_neural_all_neural_mse_source_zero_guard | joint_exact | 0.822525 | [0.292308, 1.393057] |
| 29_neural_all_neural_mse_source_zero_guard | pointwise_hard | 0.056578 | [0.001377, 0.118713] |
| 29_neural_all_neural_mse_source_zero_guard | pointwise_easy | -0.004598 | [-0.011209, 0.000000] |
| 29_neural_easy_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_neural_easy_neural_mse_no_guard | pointwise | 1.388420 | [0.583081, 2.537445] |
| 29_neural_easy_neural_mse_no_guard | independent | 3.200068 | [1.673761, 5.221504] |
| 29_neural_easy_neural_mse_no_guard | scene_uniform | 0.126824 | [0.025919, 0.254274] |
| 29_neural_easy_neural_mse_no_guard | joint | 3.177537 | [1.670535, 5.195055] |
| 29_neural_easy_neural_mse_no_guard | unary_exact | 3.181029 | [1.674986, 5.202130] |
| 29_neural_easy_neural_mse_no_guard | joint_exact | 3.181743 | [1.678066, 5.199220] |
| 29_neural_easy_neural_mse_no_guard | pointwise_hard | 1.584732 | [0.608012, 2.905705] |
| 29_neural_easy_neural_mse_no_guard | pointwise_easy | -0.467766 | [-1.960330, 0.665161] |
| 29_neural_easy_neural_mse_source_zero_guard | pointwise | 1.346608 | [0.515260, 2.503383] |
| 29_neural_easy_neural_mse_source_zero_guard | independent | 2.834389 | [1.191617, 4.969368] |
| 29_neural_easy_neural_mse_source_zero_guard | scene_uniform | 0.119387 | [0.023201, 0.247327] |
| 29_neural_easy_neural_mse_source_zero_guard | joint | 2.811432 | [1.183776, 4.933965] |
| 29_neural_easy_neural_mse_source_zero_guard | unary_exact | 2.814352 | [1.183776, 4.934184] |
| 29_neural_easy_neural_mse_source_zero_guard | joint_exact | 2.811432 | [1.183776, 4.933965] |
| 29_neural_easy_neural_mse_source_zero_guard | pointwise_hard | 1.574693 | [0.592762, 2.898694] |
| 29_neural_easy_neural_mse_source_zero_guard | pointwise_easy | -0.662360 | [-2.102117, 0.417861] |
| 29_damping097_all_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_all_neural_mse_no_guard | pointwise | 1.459401 | [0.895813, 2.046933] |
| 29_damping097_all_neural_mse_no_guard | independent | 1.895336 | [1.499723, 2.342384] |
| 29_damping097_all_neural_mse_no_guard | scene_uniform | 0.428612 | [0.131045, 0.803691] |
| 29_damping097_all_neural_mse_no_guard | joint | 1.898524 | [1.505074, 2.344581] |
| 29_damping097_all_neural_mse_no_guard | unary_exact | 1.899952 | [1.506981, 2.348840] |
| 29_damping097_all_neural_mse_no_guard | joint_exact | 1.897456 | [1.503276, 2.344549] |
| 29_damping097_all_neural_mse_no_guard | pointwise_hard | 1.721211 | [1.043699, 2.485591] |
| 29_damping097_all_neural_mse_no_guard | pointwise_easy | 0.056148 | [-0.322041, 0.372979] |
| 29_damping097_all_neural_mse_source_zero_guard | pointwise | 0.837963 | [0.401231, 1.345952] |
| 29_damping097_all_neural_mse_source_zero_guard | independent | 1.517094 | [0.862271, 2.191848] |
| 29_damping097_all_neural_mse_source_zero_guard | scene_uniform | 0.095749 | [0.023352, 0.177334] |
| 29_damping097_all_neural_mse_source_zero_guard | joint | 1.519452 | [0.866485, 2.196558] |
| 29_damping097_all_neural_mse_source_zero_guard | unary_exact | 1.518651 | [0.865256, 2.199178] |
| 29_damping097_all_neural_mse_source_zero_guard | joint_exact | 1.519452 | [0.866485, 2.196558] |
| 29_damping097_all_neural_mse_source_zero_guard | pointwise_hard | 0.956744 | [0.462629, 1.521684] |
| 29_damping097_all_neural_mse_source_zero_guard | pointwise_easy | 0.048585 | [-0.043574, 0.161893] |
| 29_damping097_easy_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_neural_mse_no_guard | pointwise | -1.095116 | [-1.932245, -0.271528] |
| 29_damping097_easy_neural_mse_no_guard | independent | 0.004459 | [-0.423865, 0.440859] |
| 29_damping097_easy_neural_mse_no_guard | scene_uniform | 0.003387 | [-0.271947, 0.254636] |
| 29_damping097_easy_neural_mse_no_guard | joint | 0.020315 | [-0.414015, 0.475409] |
| 29_damping097_easy_neural_mse_no_guard | unary_exact | 0.007699 | [-0.421372, 0.450043] |
| 29_damping097_easy_neural_mse_no_guard | joint_exact | 0.008335 | [-0.421057, 0.451332] |
| 29_damping097_easy_neural_mse_no_guard | pointwise_hard | -1.447290 | [-2.274095, -0.637778] |
| 29_damping097_easy_neural_mse_no_guard | pointwise_easy | 0.823118 | [0.115301, 1.640998] |
| 29_damping097_easy_neural_mse_source_zero_guard | pointwise | -1.296368 | [-2.005952, -0.621831] |
| 29_damping097_easy_neural_mse_source_zero_guard | independent | -0.278462 | [-0.571019, -0.050616] |
| 29_damping097_easy_neural_mse_source_zero_guard | scene_uniform | -0.143713 | [-0.343285, -0.023592] |
| 29_damping097_easy_neural_mse_source_zero_guard | joint | -0.278462 | [-0.571019, -0.050616] |
| 29_damping097_easy_neural_mse_source_zero_guard | unary_exact | -0.278462 | [-0.571019, -0.050616] |
| 29_damping097_easy_neural_mse_source_zero_guard | joint_exact | -0.278462 | [-0.571019, -0.050616] |
| 29_damping097_easy_neural_mse_source_zero_guard | pointwise_hard | -1.521613 | [-2.310413, -0.769551] |
| 29_damping097_easy_neural_mse_source_zero_guard | pointwise_easy | -0.002684 | [-0.164361, 0.153646] |
| 43_neural_all_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_neural_mse_no_guard | pointwise | 0.542981 | [0.091617, 1.133130] |
| 43_neural_all_neural_mse_no_guard | independent | 2.361207 | [1.284748, 3.636931] |
| 43_neural_all_neural_mse_no_guard | scene_uniform | -0.002354 | [-0.019155, 0.013516] |
| 43_neural_all_neural_mse_no_guard | joint | 2.362851 | [1.287461, 3.638556] |
| 43_neural_all_neural_mse_no_guard | unary_exact | 2.350479 | [1.283164, 3.595453] |
| 43_neural_all_neural_mse_no_guard | joint_exact | 2.350677 | [1.283164, 3.595656] |
| 43_neural_all_neural_mse_no_guard | pointwise_hard | 0.775563 | [0.185753, 1.527985] |
| 43_neural_all_neural_mse_no_guard | pointwise_easy | -0.047955 | [-0.098943, 0.002475] |
| 43_neural_all_neural_mse_source_zero_guard | pointwise | 0.112876 | [0.018112, 0.254823] |
| 43_neural_all_neural_mse_source_zero_guard | independent | 1.303185 | [0.445406, 2.340524] |
| 43_neural_all_neural_mse_source_zero_guard | scene_uniform | 0.004146 | [-0.004528, 0.016218] |
| 43_neural_all_neural_mse_source_zero_guard | joint | 1.300828 | [0.442967, 2.335400] |
| 43_neural_all_neural_mse_source_zero_guard | unary_exact | 1.286794 | [0.442967, 2.300028] |
| 43_neural_all_neural_mse_source_zero_guard | joint_exact | 1.286794 | [0.442967, 2.300028] |
| 43_neural_all_neural_mse_source_zero_guard | pointwise_hard | 0.148684 | [0.015316, 0.351224] |
| 43_neural_all_neural_mse_source_zero_guard | pointwise_easy | -0.002845 | [-0.030912, 0.025451] |
| 43_neural_easy_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_neural_easy_neural_mse_no_guard | pointwise | 1.688211 | [0.411395, 3.456485] |
| 43_neural_easy_neural_mse_no_guard | independent | 3.343795 | [1.844899, 5.112067] |
| 43_neural_easy_neural_mse_no_guard | scene_uniform | 0.023518 | [0.000000, 0.070553] |
| 43_neural_easy_neural_mse_no_guard | joint | 3.316058 | [1.778313, 5.080715] |
| 43_neural_easy_neural_mse_no_guard | unary_exact | 3.365400 | [1.849565, 5.143122] |
| 43_neural_easy_neural_mse_no_guard | joint_exact | 3.349759 | [1.846984, 5.126056] |
| 43_neural_easy_neural_mse_no_guard | pointwise_hard | 1.892214 | [0.403370, 3.939089] |
| 43_neural_easy_neural_mse_no_guard | pointwise_easy | -2.226157 | [-5.721958, 0.610632] |
| 43_neural_easy_neural_mse_source_zero_guard | pointwise | 1.654298 | [0.359953, 3.450254] |
| 43_neural_easy_neural_mse_source_zero_guard | independent | 2.822941 | [1.175331, 4.812481] |
| 43_neural_easy_neural_mse_source_zero_guard | scene_uniform | 0.023518 | [0.000000, 0.070553] |
| 43_neural_easy_neural_mse_source_zero_guard | joint | 2.826739 | [1.175311, 4.809218] |
| 43_neural_easy_neural_mse_source_zero_guard | unary_exact | 2.844184 | [1.176822, 4.841974] |
| 43_neural_easy_neural_mse_source_zero_guard | joint_exact | 2.827534 | [1.175311, 4.811624] |
| 43_neural_easy_neural_mse_source_zero_guard | pointwise_hard | 1.885686 | [0.390456, 3.936100] |
| 43_neural_easy_neural_mse_source_zero_guard | pointwise_easy | -2.479301 | [-5.919496, 0.231578] |
| 43_damping097_all_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_all_neural_mse_no_guard | pointwise | 1.821155 | [1.180381, 2.437452] |
| 43_damping097_all_neural_mse_no_guard | independent | 1.852294 | [1.435740, 2.233196] |
| 43_damping097_all_neural_mse_no_guard | scene_uniform | 1.003196 | [0.531189, 1.572306] |
| 43_damping097_all_neural_mse_no_guard | joint | 1.848194 | [1.433852, 2.230180] |
| 43_damping097_all_neural_mse_no_guard | unary_exact | 1.852476 | [1.431171, 2.240447] |
| 43_damping097_all_neural_mse_no_guard | joint_exact | 1.849727 | [1.431489, 2.233537] |
| 43_damping097_all_neural_mse_no_guard | pointwise_hard | 2.222347 | [1.511222, 2.970345] |
| 43_damping097_all_neural_mse_no_guard | pointwise_easy | -0.079158 | [-0.486011, 0.311049] |
| 43_damping097_all_neural_mse_source_zero_guard | pointwise | 1.036516 | [0.488104, 1.640446] |
| 43_damping097_all_neural_mse_source_zero_guard | independent | 1.352182 | [0.760392, 1.978596] |
| 43_damping097_all_neural_mse_source_zero_guard | scene_uniform | 0.464092 | [0.187973, 0.791616] |
| 43_damping097_all_neural_mse_source_zero_guard | joint | 1.351021 | [0.761453, 1.978479] |
| 43_damping097_all_neural_mse_source_zero_guard | unary_exact | 1.351645 | [0.758990, 1.981109] |
| 43_damping097_all_neural_mse_source_zero_guard | joint_exact | 1.351428 | [0.761849, 1.979293] |
| 43_damping097_all_neural_mse_source_zero_guard | pointwise_hard | 1.116092 | [0.534272, 1.751595] |
| 43_damping097_all_neural_mse_source_zero_guard | pointwise_easy | 0.088145 | [-0.126954, 0.347460] |
| 43_damping097_easy_ridge_no_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_no_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | pointwise | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | independent | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | scene_uniform | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | joint | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | unary_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | pointwise_hard | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | pointwise_easy | 0.000000 | [0.000000, 0.000000] |
| 43_damping097_easy_neural_mse_no_guard | pointwise | -0.300409 | [-0.780006, 0.159499] |
| 43_damping097_easy_neural_mse_no_guard | independent | 0.298198 | [0.109161, 0.497105] |
| 43_damping097_easy_neural_mse_no_guard | scene_uniform | 0.174769 | [-0.020018, 0.398796] |
| 43_damping097_easy_neural_mse_no_guard | joint | 0.299793 | [0.109266, 0.501840] |
| 43_damping097_easy_neural_mse_no_guard | unary_exact | 0.301225 | [0.110531, 0.503466] |
| 43_damping097_easy_neural_mse_no_guard | joint_exact | 0.302892 | [0.111455, 0.505196] |
| 43_damping097_easy_neural_mse_no_guard | pointwise_hard | -0.653396 | [-1.174948, -0.209637] |
| 43_damping097_easy_neural_mse_no_guard | pointwise_easy | 1.191145 | [0.427007, 2.141278] |
| 43_damping097_easy_neural_mse_source_zero_guard | pointwise | -0.497975 | [-0.858954, -0.200905] |
| 43_damping097_easy_neural_mse_source_zero_guard | independent | 0.075430 | [-0.040727, 0.211099] |
| 43_damping097_easy_neural_mse_source_zero_guard | scene_uniform | 0.016479 | [-0.106927, 0.119545] |
| 43_damping097_easy_neural_mse_source_zero_guard | joint | 0.072394 | [-0.043161, 0.208451] |
| 43_damping097_easy_neural_mse_source_zero_guard | unary_exact | 0.072465 | [-0.042970, 0.208469] |
| 43_damping097_easy_neural_mse_source_zero_guard | joint_exact | 0.072399 | [-0.043160, 0.208468] |
| 43_damping097_easy_neural_mse_source_zero_guard | pointwise_hard | -0.711377 | [-1.202936, -0.298207] |
| 43_damping097_easy_neural_mse_source_zero_guard | pointwise_easy | 0.267549 | [0.130650, 0.417584] |

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
| 17_all_neural_mse_no_guard | pointwise | -1.863985 | [-2.623618, -1.149651] |
| 17_all_neural_mse_no_guard | independent | -1.985137 | [-3.061997, -0.613473] |
| 17_all_neural_mse_no_guard | scene_uniform | -0.658502 | [-1.238123, -0.214016] |
| 17_all_neural_mse_no_guard | joint | -1.998324 | [-3.057301, -0.669643] |
| 17_all_neural_mse_no_guard | unary_exact | -1.980056 | [-3.056424, -0.614678] |
| 17_all_neural_mse_no_guard | joint_exact | -2.002864 | [-3.061754, -0.669722] |
| 17_all_neural_mse_no_guard | pointwise_hard | -2.281753 | [-3.285110, -1.399979] |
| 17_all_neural_mse_no_guard | pointwise_easy | -0.167840 | [-0.468114, 0.058780] |
| 17_all_neural_mse_source_zero_guard | pointwise | -0.984130 | [-1.624623, -0.439144] |
| 17_all_neural_mse_source_zero_guard | independent | -1.993570 | [-2.945731, -1.102302] |
| 17_all_neural_mse_source_zero_guard | scene_uniform | -0.142009 | [-0.273416, -0.041457] |
| 17_all_neural_mse_source_zero_guard | joint | -1.986759 | [-2.930190, -1.101438] |
| 17_all_neural_mse_source_zero_guard | unary_exact | -1.982444 | [-2.926375, -1.093365] |
| 17_all_neural_mse_source_zero_guard | joint_exact | -1.990614 | [-2.937772, -1.101435] |
| 17_all_neural_mse_source_zero_guard | pointwise_hard | -1.161543 | [-1.935263, -0.515081] |
| 17_all_neural_mse_source_zero_guard | pointwise_easy | -0.004191 | [-0.072718, 0.071669] |
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
| 17_easy_neural_mse_no_guard | pointwise | -0.553259 | [-1.108895, -0.018998] |
| 17_easy_neural_mse_no_guard | independent | 0.194570 | [-1.202877, 1.990033] |
| 17_easy_neural_mse_no_guard | scene_uniform | -1.060256 | [-1.644025, -0.548686] |
| 17_easy_neural_mse_no_guard | joint | 0.215941 | [-1.157571, 1.982857] |
| 17_easy_neural_mse_no_guard | unary_exact | 0.224437 | [-1.140959, 1.987290] |
| 17_easy_neural_mse_no_guard | joint_exact | 0.215119 | [-1.159598, 1.982955] |
| 17_easy_neural_mse_no_guard | pointwise_hard | -0.090878 | [-0.713672, 0.560113] |
| 17_easy_neural_mse_no_guard | pointwise_easy | -2.783185 | [-4.294314, -1.447860] |
| 17_easy_neural_mse_source_zero_guard | pointwise | -0.405828 | [-0.989125, 0.124946] |
| 17_easy_neural_mse_source_zero_guard | independent | 0.898322 | [-0.163461, 2.422289] |
| 17_easy_neural_mse_source_zero_guard | scene_uniform | -0.896655 | [-1.551372, -0.350840] |
| 17_easy_neural_mse_source_zero_guard | joint | 0.889360 | [-0.172564, 2.405255] |
| 17_easy_neural_mse_source_zero_guard | unary_exact | 0.890672 | [-0.172438, 2.408631] |
| 17_easy_neural_mse_source_zero_guard | joint_exact | 0.889793 | [-0.172438, 2.405996] |
| 17_easy_neural_mse_source_zero_guard | pointwise_hard | -0.043183 | [-0.674415, 0.594247] |
| 17_easy_neural_mse_source_zero_guard | pointwise_easy | -2.026331 | [-3.721080, -0.595399] |
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
| 29_all_neural_mse_no_guard | pointwise | -1.857832 | [-2.649322, -1.111350] |
| 29_all_neural_mse_no_guard | independent | -1.892768 | [-2.824737, -0.908841] |
| 29_all_neural_mse_no_guard | scene_uniform | -0.442379 | [-0.825538, -0.134117] |
| 29_all_neural_mse_no_guard | joint | -1.893523 | [-2.825332, -0.918781] |
| 29_all_neural_mse_no_guard | unary_exact | -1.883083 | [-2.818099, -0.904881] |
| 29_all_neural_mse_no_guard | joint_exact | -1.890607 | [-2.825421, -0.908113] |
| 29_all_neural_mse_no_guard | pointwise_hard | -2.225111 | [-3.203399, -1.342991] |
| 29_all_neural_mse_no_guard | pointwise_easy | -0.047068 | [-0.358962, 0.327565] |
| 29_all_neural_mse_source_zero_guard | pointwise | -1.049494 | [-1.741806, -0.472097] |
| 29_all_neural_mse_source_zero_guard | independent | -1.865791 | [-2.720163, -1.057255] |
| 29_all_neural_mse_source_zero_guard | scene_uniform | -0.089246 | [-0.161313, -0.024684] |
| 29_all_neural_mse_source_zero_guard | joint | -1.866352 | [-2.715959, -1.060000] |
| 29_all_neural_mse_source_zero_guard | unary_exact | -1.865571 | [-2.716193, -1.059654] |
| 29_all_neural_mse_source_zero_guard | joint_exact | -1.866352 | [-2.715959, -1.060000] |
| 29_all_neural_mse_source_zero_guard | pointwise_hard | -1.225118 | [-2.023636, -0.555650] |
| 29_all_neural_mse_source_zero_guard | pointwise_easy | -0.047845 | [-0.162422, 0.045380] |
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
| 29_easy_neural_mse_no_guard | pointwise | 0.437774 | [-0.418468, 1.449340] |
| 29_easy_neural_mse_no_guard | independent | 0.162746 | [-1.643196, 2.448880] |
| 29_easy_neural_mse_no_guard | scene_uniform | -0.641503 | [-1.060155, -0.278350] |
| 29_easy_neural_mse_no_guard | joint | 0.136892 | [-1.658222, 2.439275] |
| 29_easy_neural_mse_no_guard | unary_exact | 0.140467 | [-1.659965, 2.441709] |
| 29_easy_neural_mse_no_guard | joint_exact | 0.143649 | [-1.646825, 2.441709] |
| 29_easy_neural_mse_no_guard | pointwise_hard | 1.034357 | [0.089150, 2.255505] |
| 29_easy_neural_mse_no_guard | pointwise_easy | -3.198103 | [-5.378830, -1.543613] |
| 29_easy_neural_mse_source_zero_guard | pointwise | 0.650385 | [-0.130039, 1.600071] |
| 29_easy_neural_mse_source_zero_guard | independent | 1.184251 | [-0.059468, 3.073113] |
| 29_easy_neural_mse_source_zero_guard | scene_uniform | -0.498223 | [-0.963556, -0.120490] |
| 29_easy_neural_mse_source_zero_guard | joint | 1.160474 | [-0.070918, 3.040398] |
| 29_easy_neural_mse_source_zero_guard | unary_exact | 1.163338 | [-0.070584, 3.043169] |
| 29_easy_neural_mse_source_zero_guard | joint_exact | 1.160283 | [-0.070921, 3.039829] |
| 29_easy_neural_mse_source_zero_guard | pointwise_hard | 1.108528 | [0.200397, 2.291376] |
| 29_easy_neural_mse_source_zero_guard | pointwise_easy | -2.258428 | [-4.673309, -0.504306] |
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
| 43_all_neural_mse_no_guard | pointwise | -1.822049 | [-2.476111, -1.187538] |
| 43_all_neural_mse_no_guard | independent | -1.469981 | [-2.701982, 0.033692] |
| 43_all_neural_mse_no_guard | scene_uniform | -1.059072 | [-1.637785, -0.570622] |
| 43_all_neural_mse_no_guard | joint | -1.458192 | [-2.688226, 0.042391] |
| 43_all_neural_mse_no_guard | unary_exact | -1.471652 | [-2.688416, 0.014481] |
| 43_all_neural_mse_no_guard | joint_exact | -1.471167 | [-2.687982, 0.014818] |
| 43_all_neural_mse_no_guard | pointwise_hard | -2.144096 | [-2.918418, -1.466616] |
| 43_all_neural_mse_no_guard | pointwise_easy | 0.055626 | [-0.340841, 0.466049] |
| 43_all_neural_mse_source_zero_guard | pointwise | -1.292588 | [-2.067035, -0.602521] |
| 43_all_neural_mse_source_zero_guard | independent | -1.464866 | [-2.387162, -0.518645] |
| 43_all_neural_mse_source_zero_guard | scene_uniform | -0.494208 | [-0.839493, -0.204835] |
| 43_all_neural_mse_source_zero_guard | joint | -1.458379 | [-2.377980, -0.513889] |
| 43_all_neural_mse_source_zero_guard | unary_exact | -1.473220 | [-2.381493, -0.543136] |
| 43_all_neural_mse_source_zero_guard | joint_exact | -1.472939 | [-2.379199, -0.543482] |
| 43_all_neural_mse_source_zero_guard | pointwise_hard | -1.400711 | [-2.252918, -0.663773] |
| 43_all_neural_mse_source_zero_guard | pointwise_easy | -0.097403 | [-0.353232, 0.119614] |
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
| 43_easy_neural_mse_no_guard | pointwise | 0.239451 | [-0.854406, 1.691192] |
| 43_easy_neural_mse_no_guard | independent | 0.053733 | [-1.568763, 2.146276] |
| 43_easy_neural_mse_no_guard | scene_uniform | -0.987456 | [-1.587536, -0.500299] |
| 43_easy_neural_mse_no_guard | joint | 0.037269 | [-1.630559, 2.137942] |
| 43_easy_neural_mse_no_guard | unary_exact | 0.088761 | [-1.556230, 2.193364] |
| 43_easy_neural_mse_no_guard | joint_exact | 0.070485 | [-1.563536, 2.163961] |
| 43_easy_neural_mse_no_guard | pointwise_hard | 0.922868 | [-0.310897, 2.625536] |
| 43_easy_neural_mse_no_guard | pointwise_easy | -5.635954 | [-9.974484, -2.320904] |
| 43_easy_neural_mse_source_zero_guard | pointwise | 0.435812 | [-0.638072, 1.859457] |
| 43_easy_neural_mse_source_zero_guard | independent | 0.783300 | [-0.510912, 2.589852] |
| 43_easy_neural_mse_source_zero_guard | scene_uniform | -0.784008 | [-1.450909, -0.274505] |
| 43_easy_neural_mse_source_zero_guard | joint | 0.800063 | [-0.503645, 2.607462] |
| 43_easy_neural_mse_source_zero_guard | unary_exact | 0.817058 | [-0.497111, 2.632317] |
| 43_easy_neural_mse_source_zero_guard | joint_exact | 0.799393 | [-0.506599, 2.609873] |
| 43_easy_neural_mse_source_zero_guard | pointwise_hard | 0.986238 | [-0.228450, 2.664773] |
| 43_easy_neural_mse_source_zero_guard | pointwise_easy | -4.777870 | [-9.345562, -1.188828] |

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
| 17_neural_all_neural_mse_no_guard | independent | 2.626629 | [1.475606, 4.142321] | 6.349346 | 19.751472 |
| 17_neural_all_neural_mse_no_guard | scene_uniform | 0.007317 | [-0.036431, 0.046601] | -0.000000 | 0.114454 |
| 17_neural_all_neural_mse_no_guard | joint | 2.605651 | [1.471622, 4.096988] | 6.349346 | 19.751472 |
| 17_neural_all_neural_mse_no_guard | unary_exact | 2.621353 | [1.478803, 4.136711] | 6.349346 | 19.751472 |
| 17_neural_all_neural_mse_no_guard | joint_exact | 2.605651 | [1.471622, 4.096988] | 6.349346 | 19.751472 |
| 17_neural_all_neural_mse_source_zero_guard | independent | 1.054020 | [0.435509, 1.723843] | 4.536468 | 7.684761 |
| 17_neural_all_neural_mse_source_zero_guard | scene_uniform | 0.013248 | [-0.002600, 0.036595] | -0.000000 | 0.065402 |
| 17_neural_all_neural_mse_source_zero_guard | joint | 1.052532 | [0.435532, 1.720874] | 4.536468 | 7.684761 |
| 17_neural_all_neural_mse_source_zero_guard | unary_exact | 1.054993 | [0.441543, 1.720874] | 4.536468 | 7.684761 |
| 17_neural_all_neural_mse_source_zero_guard | joint_exact | 1.052532 | [0.435532, 1.720874] | 4.536468 | 7.684761 |
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
| 17_neural_easy_neural_mse_no_guard | independent | 4.205357 | [2.374940, 6.425005] | 8.431590 | 28.580772 |
| 17_neural_easy_neural_mse_no_guard | scene_uniform | 0.156496 | [0.048061, 0.302565] | -0.000000 | 0.752126 |
| 17_neural_easy_neural_mse_no_guard | joint | 4.223246 | [2.397146, 6.425462] | 8.444117 | 28.580772 |
| 17_neural_easy_neural_mse_no_guard | unary_exact | 4.232286 | [2.406284, 6.436649] | 8.234597 | 28.580772 |
| 17_neural_easy_neural_mse_no_guard | joint_exact | 4.223660 | [2.397146, 6.425553] | 8.431590 | 28.580772 |
| 17_neural_easy_neural_mse_source_zero_guard | independent | 3.553724 | [1.540412, 5.923154] | 8.431590 | 18.754088 |
| 17_neural_easy_neural_mse_source_zero_guard | scene_uniform | 0.149057 | [0.039689, 0.296839] | -0.000000 | 0.735775 |
| 17_neural_easy_neural_mse_source_zero_guard | joint | 3.545188 | [1.536841, 5.920486] | 8.444117 | 18.754088 |
| 17_neural_easy_neural_mse_source_zero_guard | unary_exact | 3.546432 | [1.537227, 5.922192] | 8.234597 | 18.754088 |
| 17_neural_easy_neural_mse_source_zero_guard | joint_exact | 3.545601 | [1.537227, 5.920525] | 8.431590 | 18.754088 |
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
| 17_damping097_all_neural_mse_no_guard | independent | 4.520686 | [3.777278, 5.267706] | -0.252067 | 58.943754 |
| 17_damping097_all_neural_mse_no_guard | scene_uniform | 0.653223 | [0.222853, 1.201602] | 1.131188 | 9.287116 |
| 17_damping097_all_neural_mse_no_guard | joint | 4.512678 | [3.762893, 5.249470] | -0.252067 | 58.829300 |
| 17_damping097_all_neural_mse_no_guard | unary_exact | 4.510912 | [3.760226, 5.248197] | -0.252067 | 58.943754 |
| 17_damping097_all_neural_mse_no_guard | joint_exact | 4.516712 | [3.765839, 5.253961] | -0.252067 | 58.943754 |
| 17_damping097_all_neural_mse_source_zero_guard | independent | 2.952426 | [1.655427, 4.295850] | -0.000000 | 30.134075 |
| 17_damping097_all_neural_mse_source_zero_guard | scene_uniform | 0.154572 | [0.047067, 0.289701] | -0.000000 | 0.572269 |
| 17_damping097_all_neural_mse_source_zero_guard | joint | 2.945001 | [1.646464, 4.280736] | -0.000000 | 30.101373 |
| 17_damping097_all_neural_mse_source_zero_guard | unary_exact | 2.943190 | [1.647763, 4.279436] | -0.000000 | 30.134075 |
| 17_damping097_all_neural_mse_source_zero_guard | joint_exact | 2.948452 | [1.651034, 4.286466] | -0.000000 | 30.134075 |
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
| 17_damping097_easy_neural_mse_no_guard | independent | 4.033090 | [3.012863, 4.955199] | 0.834844 | 67.184434 |
| 17_damping097_easy_neural_mse_no_guard | scene_uniform | 1.195719 | [0.722940, 1.734862] | 0.168457 | 19.816874 |
| 17_damping097_easy_neural_mse_no_guard | joint | 4.030898 | [3.011714, 4.950909] | 0.834844 | 67.135383 |
| 17_damping097_easy_neural_mse_no_guard | unary_exact | 4.031855 | [3.012863, 4.953511] | 0.834844 | 67.184434 |
| 17_damping097_easy_neural_mse_no_guard | joint_exact | 4.032053 | [3.012840, 4.953176] | 0.834844 | 67.184434 |
| 17_damping097_easy_neural_mse_source_zero_guard | independent | 2.704329 | [1.393696, 4.053146] | 0.834844 | 40.385873 |
| 17_damping097_easy_neural_mse_source_zero_guard | scene_uniform | 1.025902 | [0.486531, 1.635205] | 0.134152 | 14.306736 |
| 17_damping097_easy_neural_mse_source_zero_guard | joint | 2.704329 | [1.393696, 4.053146] | 0.834844 | 40.385873 |
| 17_damping097_easy_neural_mse_source_zero_guard | unary_exact | 2.704329 | [1.393696, 4.053146] | 0.834844 | 40.385873 |
| 17_damping097_easy_neural_mse_source_zero_guard | joint_exact | 2.704329 | [1.393696, 4.053146] | 0.834844 | 40.385873 |
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
| 29_neural_all_neural_mse_no_guard | independent | 2.420990 | [1.374340, 3.638309] | 7.763393 | 16.563113 |
| 29_neural_all_neural_mse_no_guard | scene_uniform | -0.003544 | [-0.044903, 0.035077] | -0.000000 | 0.065402 |
| 29_neural_all_neural_mse_no_guard | joint | 2.418332 | [1.373840, 3.624260] | 7.710521 | 16.514061 |
| 29_neural_all_neural_mse_no_guard | unary_exact | 2.429594 | [1.379142, 3.639538] | 5.881097 | 16.563113 |
| 29_neural_all_neural_mse_no_guard | joint_exact | 2.421428 | [1.373835, 3.634062] | 7.763393 | 16.563113 |
| 29_neural_all_neural_mse_source_zero_guard | independent | 1.013893 | [0.318936, 1.741243] | 2.531459 | 6.981687 |
| 29_neural_all_neural_mse_source_zero_guard | scene_uniform | 0.010941 | [-0.004348, 0.037171] | -0.000000 | 0.032701 |
| 29_neural_all_neural_mse_source_zero_guard | joint | 1.011747 | [0.316178, 1.737635] | 3.000004 | 6.981687 |
| 29_neural_all_neural_mse_source_zero_guard | unary_exact | 1.011747 | [0.316178, 1.737635] | 3.000004 | 6.981687 |
| 29_neural_all_neural_mse_source_zero_guard | joint_exact | 1.011747 | [0.316178, 1.737635] | 3.000004 | 6.981687 |
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
| 29_neural_easy_neural_mse_no_guard | independent | 4.012263 | [2.175233, 6.490619] | 17.015369 | 24.623937 |
| 29_neural_easy_neural_mse_no_guard | scene_uniform | 0.169388 | [0.059377, 0.298729] | 0.302621 | 0.523218 |
| 29_neural_easy_neural_mse_no_guard | joint | 3.986775 | [2.154048, 6.445515] | 17.015369 | 24.558535 |
| 29_neural_easy_neural_mse_no_guard | unary_exact | 3.993230 | [2.164846, 6.454738] | 17.015369 | 24.623937 |
| 29_neural_easy_neural_mse_no_guard | joint_exact | 3.993946 | [2.168538, 6.454756] | 17.015369 | 24.623937 |
| 29_neural_easy_neural_mse_source_zero_guard | independent | 3.621572 | [1.597981, 6.199518] | 17.015369 | 17.184434 |
| 29_neural_easy_neural_mse_source_zero_guard | scene_uniform | 0.161951 | [0.051940, 0.295854] | 0.302621 | 0.506867 |
| 29_neural_easy_neural_mse_source_zero_guard | joint | 3.598621 | [1.580220, 6.149995] | 17.015369 | 17.184434 |
| 29_neural_easy_neural_mse_source_zero_guard | unary_exact | 3.601541 | [1.581090, 6.153557] | 17.015369 | 17.184434 |
| 29_neural_easy_neural_mse_source_zero_guard | joint_exact | 3.598621 | [1.580220, 6.149995] | 17.015369 | 17.184434 |
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
| 29_damping097_all_neural_mse_no_guard | independent | 4.228070 | [3.336861, 5.085267] | 0.005559 | 52.517986 |
| 29_damping097_all_neural_mse_no_guard | scene_uniform | 0.433055 | [0.133823, 0.811743] | 2.781889 | 5.510137 |
| 29_damping097_all_neural_mse_no_guard | joint | 4.226195 | [3.330732, 5.086018] | 0.005559 | 52.419882 |
| 29_damping097_all_neural_mse_no_guard | unary_exact | 4.227514 | [3.330071, 5.088262] | 0.005559 | 52.517986 |
| 29_damping097_all_neural_mse_no_guard | joint_exact | 4.226548 | [3.330720, 5.085964] | 0.005559 | 52.517986 |
| 29_damping097_all_neural_mse_source_zero_guard | independent | 2.796330 | [1.509205, 4.132318] | -0.000000 | 25.588620 |
| 29_damping097_all_neural_mse_source_zero_guard | scene_uniform | 0.099922 | [0.024338, 0.185202] | -0.000000 | 0.474166 |
| 29_damping097_all_neural_mse_source_zero_guard | joint | 2.794808 | [1.504897, 4.130845] | -0.000000 | 25.572269 |
| 29_damping097_all_neural_mse_source_zero_guard | unary_exact | 2.794056 | [1.507046, 4.130028] | -0.000000 | 25.588620 |
| 29_damping097_all_neural_mse_source_zero_guard | joint_exact | 2.794808 | [1.504897, 4.130845] | -0.000000 | 25.588620 |
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
| 29_damping097_easy_neural_mse_no_guard | independent | 3.853650 | [2.794092, 4.827493] | -0.225174 | 61.968607 |
| 29_damping097_easy_neural_mse_no_guard | scene_uniform | 0.801448 | [0.477998, 1.187091] | 3.029203 | 13.521910 |
| 29_damping097_easy_neural_mse_no_guard | joint | 3.853104 | [2.793660, 4.828313] | -0.225174 | 61.837802 |
| 29_damping097_easy_neural_mse_no_guard | unary_exact | 3.855844 | [2.794857, 4.831882] | -0.293811 | 61.968607 |
| 29_damping097_easy_neural_mse_no_guard | joint_exact | 3.853650 | [2.794092, 4.827493] | -0.225174 | 61.968607 |
| 29_damping097_easy_neural_mse_source_zero_guard | independent | 2.487771 | [1.201731, 3.797235] | -0.000000 | 35.971223 |
| 29_damping097_easy_neural_mse_source_zero_guard | scene_uniform | 0.651640 | [0.271356, 1.088836] | -0.000000 | 9.450621 |
| 29_damping097_easy_neural_mse_source_zero_guard | joint | 2.487578 | [1.201731, 3.796889] | -0.000000 | 35.922171 |
| 29_damping097_easy_neural_mse_source_zero_guard | unary_exact | 2.487771 | [1.201731, 3.797235] | -0.000000 | 35.971223 |
| 29_damping097_easy_neural_mse_source_zero_guard | joint_exact | 2.487771 | [1.201731, 3.797235] | -0.000000 | 35.971223 |
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
| 43_neural_all_neural_mse_no_guard | independent | 3.179241 | [1.836432, 4.825510] | 5.496495 | 21.010464 |
| 43_neural_all_neural_mse_no_guard | scene_uniform | 0.004721 | [-0.037348, 0.040801] | 0.302724 | 0.228908 |
| 43_neural_all_neural_mse_no_guard | joint | 3.188648 | [1.843630, 4.832223] | 5.496495 | 20.961413 |
| 43_neural_all_neural_mse_no_guard | unary_exact | 3.176755 | [1.842045, 4.803816] | 5.496495 | 21.010464 |
| 43_neural_all_neural_mse_no_guard | joint_exact | 3.176946 | [1.842045, 4.804391] | 5.496495 | 21.010464 |
| 43_neural_all_neural_mse_source_zero_guard | independent | 1.598844 | [0.575565, 2.774022] | 4.183822 | 9.156311 |
| 43_neural_all_neural_mse_source_zero_guard | scene_uniform | 0.014903 | [-0.002712, 0.041648] | 0.302724 | 0.114454 |
| 43_neural_all_neural_mse_source_zero_guard | joint | 1.604520 | [0.581384, 2.782368] | 3.526765 | 9.139961 |
| 43_neural_all_neural_mse_source_zero_guard | unary_exact | 1.590646 | [0.581384, 2.752675] | 3.526765 | 9.156311 |
| 43_neural_all_neural_mse_source_zero_guard | joint_exact | 1.590646 | [0.581384, 2.752675] | 3.526765 | 9.156311 |
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
| 43_neural_easy_neural_mse_no_guard | independent | 4.573292 | [2.729092, 6.858526] | 17.916899 | 30.559189 |
| 43_neural_easy_neural_mse_no_guard | scene_uniform | 0.146996 | [0.030560, 0.295913] | 0.302724 | 0.457816 |
| 43_neural_easy_neural_mse_no_guard | joint | 4.544898 | [2.673359, 6.851707] | 16.997676 | 30.461086 |
| 43_neural_easy_neural_mse_no_guard | unary_exact | 4.594199 | [2.730577, 6.893847] | 17.916899 | 30.559189 |
| 43_neural_easy_neural_mse_no_guard | joint_exact | 4.578522 | [2.730650, 6.863759] | 17.916899 | 30.559189 |
| 43_neural_easy_neural_mse_source_zero_guard | independent | 3.913124 | [1.812608, 6.397659] | 17.916899 | 21.402878 |
| 43_neural_easy_neural_mse_source_zero_guard | scene_uniform | 0.139564 | [0.022317, 0.288473] | 0.302724 | 0.441465 |
| 43_neural_easy_neural_mse_source_zero_guard | joint | 3.916989 | [1.812741, 6.411836] | 16.997676 | 21.370177 |
| 43_neural_easy_neural_mse_source_zero_guard | unary_exact | 3.934413 | [1.814819, 6.459134] | 17.916899 | 21.402878 |
| 43_neural_easy_neural_mse_source_zero_guard | joint_exact | 3.917742 | [1.812741, 6.412668] | 17.916899 | 21.402878 |
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
| 43_damping097_all_neural_mse_no_guard | independent | 4.582405 | [3.768995, 5.336289] | 4.107552 | 62.034009 |
| 43_damping097_all_neural_mse_no_guard | scene_uniform | 1.044105 | [0.572980, 1.599402] | 0.822746 | 11.739699 |
| 43_damping097_all_neural_mse_no_guard | joint | 4.580720 | [3.766357, 5.336198] | 4.107552 | 61.968607 |
| 43_damping097_all_neural_mse_no_guard | unary_exact | 4.581539 | [3.766809, 5.338090] | 4.107552 | 62.034009 |
| 43_damping097_all_neural_mse_no_guard | joint_exact | 4.581306 | [3.766984, 5.335746] | 4.107552 | 62.034009 |
| 43_damping097_all_neural_mse_source_zero_guard | independent | 2.995686 | [1.643267, 4.394280] | -0.000000 | 33.306082 |
| 43_damping097_all_neural_mse_source_zero_guard | scene_uniform | 0.503470 | [0.213915, 0.844423] | 0.393626 | 2.975801 |
| 43_damping097_all_neural_mse_source_zero_guard | joint | 2.995224 | [1.643141, 4.393203] | -0.000000 | 33.306082 |
| 43_damping097_all_neural_mse_source_zero_guard | unary_exact | 2.995457 | [1.641456, 4.394234] | -0.000000 | 33.306082 |
| 43_damping097_all_neural_mse_source_zero_guard | joint_exact | 2.995224 | [1.643141, 4.393203] | -0.000000 | 33.306082 |
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
| 43_damping097_easy_neural_mse_no_guard | independent | 4.531553 | [3.663126, 5.376784] | 0.410655 | 69.228254 |
| 43_damping097_easy_neural_mse_no_guard | scene_uniform | 1.114692 | [0.639508, 1.683405] | 1.763311 | 19.587966 |
| 43_damping097_easy_neural_mse_no_guard | joint | 4.518928 | [3.648749, 5.361924] | 0.410655 | 69.113800 |
| 43_damping097_easy_neural_mse_no_guard | unary_exact | 4.519138 | [3.650294, 5.362483] | 0.410655 | 69.228254 |
| 43_damping097_easy_neural_mse_no_guard | joint_exact | 4.520696 | [3.650372, 5.365100] | 0.410655 | 69.228254 |
| 43_damping097_easy_neural_mse_source_zero_guard | independent | 3.170941 | [1.782018, 4.610828] | 0.410655 | 41.939176 |
| 43_damping097_easy_neural_mse_source_zero_guard | scene_uniform | 0.905811 | [0.392045, 1.556389] | 0.365702 | 13.260301 |
| 43_damping097_easy_neural_mse_source_zero_guard | joint | 3.158749 | [1.775946, 4.596411] | 0.410655 | 41.906475 |
| 43_damping097_easy_neural_mse_source_zero_guard | unary_exact | 3.160083 | [1.777280, 4.599046] | 0.410655 | 41.939176 |
| 43_damping097_easy_neural_mse_source_zero_guard | joint_exact | 3.160083 | [1.777280, 4.599046] | 0.410655 | 41.939176 |

## Within-Candidate Joint Contrasts

| View | Contrast | Gain (%) | CI |
|---|---|---:|---|
| 17_neural_all_ridge_no_guard | joint_vs_independent | -0.026720 | [-0.080566, 0.020143] |
| 17_neural_all_ridge_no_guard | joint_exact_vs_independent | -0.025982 | [-0.075574, 0.011257] |
| 17_neural_all_ridge_no_guard | joint_exact_vs_unary | -0.010240 | [-0.030719, 0.000000] |
| 17_neural_all_ridge_source_zero_guard | joint_vs_independent | 0.005795 | [-0.020099, 0.033404] |
| 17_neural_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_neural_all_neural_mse_no_guard | joint_vs_independent | -0.022517 | [-0.053620, 0.000000] |
| 17_neural_all_neural_mse_no_guard | joint_exact_vs_independent | -0.023519 | [-0.055626, 0.000000] |
| 17_neural_all_neural_mse_no_guard | joint_exact_vs_unary | -0.018943 | [-0.052963, 0.000000] |
| 17_neural_all_neural_mse_source_zero_guard | joint_vs_independent | -0.001533 | [-0.005830, 0.001230] |
| 17_neural_all_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_all_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_neural_easy_ridge_no_guard | joint_vs_independent | 0.038073 | [-0.029743, 0.156534] |
| 17_neural_easy_ridge_no_guard | joint_exact_vs_independent | -0.031757 | [-0.186162, 0.102397] |
| 17_neural_easy_ridge_no_guard | joint_exact_vs_unary | 0.033104 | [0.000000, 0.099313] |
| 17_neural_easy_ridge_source_zero_guard | joint_vs_independent | -0.007819 | [-0.023458, 0.000000] |
| 17_neural_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_neural_easy_neural_mse_no_guard | joint_vs_independent | 0.017991 | [-0.027128, 0.083547] |
| 17_neural_easy_neural_mse_no_guard | joint_exact_vs_independent | 0.015867 | [-0.038374, 0.089835] |
| 17_neural_easy_neural_mse_no_guard | joint_exact_vs_unary | -0.009836 | [-0.027152, 0.000000] |
| 17_neural_easy_neural_mse_source_zero_guard | joint_vs_independent | -0.009061 | [-0.028562, 0.002151] |
| 17_neural_easy_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_neural_easy_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_all_ridge_no_guard | joint_vs_independent | -0.004875 | [-0.012317, 0.000674] |
| 17_damping097_all_ridge_no_guard | joint_exact_vs_independent | -0.007899 | [-0.026613, 0.002702] |
| 17_damping097_all_ridge_no_guard | joint_exact_vs_unary | 0.020390 | [0.000000, 0.056821] |
| 17_damping097_all_ridge_source_zero_guard | joint_vs_independent | -0.002976 | [-0.010182, 0.002155] |
| 17_damping097_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_all_neural_mse_no_guard | joint_vs_independent | -0.008544 | [-0.033724, 0.010051] |
| 17_damping097_all_neural_mse_no_guard | joint_exact_vs_independent | -0.007326 | [-0.029661, 0.010504] |
| 17_damping097_all_neural_mse_no_guard | joint_exact_vs_unary | 0.006689 | [0.000361, 0.015501] |
| 17_damping097_all_neural_mse_source_zero_guard | joint_vs_independent | -0.007946 | [-0.032992, 0.010202] |
| 17_damping097_all_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_all_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_easy_ridge_no_guard | joint_vs_independent | -0.009693 | [-0.028249, 0.002179] |
| 17_damping097_easy_ridge_no_guard | joint_exact_vs_independent | -0.010347 | [-0.030373, 0.001733] |
| 17_damping097_easy_ridge_no_guard | joint_exact_vs_unary | 0.005164 | [-0.001880, 0.017371] |
| 17_damping097_easy_ridge_source_zero_guard | joint_vs_independent | -0.007455 | [-0.025134, 0.002346] |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_damping097_easy_neural_mse_no_guard | joint_vs_independent | -0.002315 | [-0.006946, 0.000000] |
| 17_damping097_easy_neural_mse_no_guard | joint_exact_vs_independent | -0.001147 | [-0.003442, 0.000000] |
| 17_damping097_easy_neural_mse_no_guard | joint_exact_vs_unary | 0.000187 | [-0.001926, 0.002486] |
| 17_damping097_easy_neural_mse_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_damping097_easy_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_damping097_easy_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_all_ridge_no_guard | joint_vs_independent | -0.013443 | [-0.036961, 0.007007] |
| 29_neural_all_ridge_no_guard | joint_exact_vs_independent | -0.020242 | [-0.055009, 0.009271] |
| 29_neural_all_ridge_no_guard | joint_exact_vs_unary | -0.011671 | [-0.035014, 0.000000] |
| 29_neural_all_ridge_source_zero_guard | joint_vs_independent | -0.014402 | [-0.037506, 0.005591] |
| 29_neural_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_all_neural_mse_no_guard | joint_vs_independent | -0.002872 | [-0.017706, 0.011405] |
| 29_neural_all_neural_mse_no_guard | joint_exact_vs_independent | 0.001359 | [-0.015678, 0.018518] |
| 29_neural_all_neural_mse_no_guard | joint_exact_vs_unary | -0.008824 | [-0.026471, 0.000000] |
| 29_neural_all_neural_mse_source_zero_guard | joint_vs_independent | -0.002166 | [-0.013500, 0.008559] |
| 29_neural_all_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_all_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_easy_ridge_no_guard | joint_vs_independent | -0.025788 | [-0.063109, 0.000000] |
| 29_neural_easy_ridge_no_guard | joint_exact_vs_independent | -0.067262 | [-0.190669, 0.000000] |
| 29_neural_easy_ridge_no_guard | joint_exact_vs_unary | -0.011839 | [-0.035518, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | joint_vs_independent | -0.007686 | [-0.023059, 0.000000] |
| 29_neural_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_neural_easy_neural_mse_no_guard | joint_vs_independent | -0.027180 | [-0.081062, 0.013241] |
| 29_neural_easy_neural_mse_no_guard | joint_exact_vs_independent | -0.030102 | [-0.098278, 0.010127] |
| 29_neural_easy_neural_mse_no_guard | joint_exact_vs_unary | -0.000176 | [-0.012446, 0.011920] |
| 29_neural_easy_neural_mse_source_zero_guard | joint_vs_independent | -0.024605 | [-0.072161, 0.000000] |
| 29_neural_easy_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_neural_easy_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_all_ridge_no_guard | joint_vs_independent | -0.007693 | [-0.017229, -0.000478] |
| 29_damping097_all_ridge_no_guard | joint_exact_vs_independent | -0.011976 | [-0.033300, 0.001042] |
| 29_damping097_all_ridge_no_guard | joint_exact_vs_unary | -0.001174 | [-0.007282, 0.003441] |
| 29_damping097_all_ridge_source_zero_guard | joint_vs_independent | -0.006270 | [-0.016136, 0.000665] |
| 29_damping097_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_all_neural_mse_no_guard | joint_vs_independent | -0.001912 | [-0.010856, 0.005236] |
| 29_damping097_all_neural_mse_no_guard | joint_exact_vs_independent | -0.001928 | [-0.012418, 0.006605] |
| 29_damping097_all_neural_mse_no_guard | joint_exact_vs_unary | -0.001004 | [-0.009545, 0.007937] |
| 29_damping097_all_neural_mse_source_zero_guard | joint_vs_independent | -0.001547 | [-0.010856, 0.006189] |
| 29_damping097_all_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_all_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_easy_ridge_no_guard | joint_vs_independent | 0.008799 | [-0.012885, 0.029873] |
| 29_damping097_easy_ridge_no_guard | joint_exact_vs_independent | -0.000303 | [-0.025994, 0.022752] |
| 29_damping097_easy_ridge_no_guard | joint_exact_vs_unary | 0.002622 | [0.000000, 0.006327] |
| 29_damping097_easy_ridge_source_zero_guard | joint_vs_independent | 0.003072 | [-0.016637, 0.022437] |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_damping097_easy_neural_mse_no_guard | joint_vs_independent | -0.000565 | [-0.002667, 0.001256] |
| 29_damping097_easy_neural_mse_no_guard | joint_exact_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_damping097_easy_neural_mse_no_guard | joint_exact_vs_unary | -0.002406 | [-0.007217, 0.000000] |
| 29_damping097_easy_neural_mse_source_zero_guard | joint_vs_independent | -0.000202 | [-0.000488, 0.000000] |
| 29_damping097_easy_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_damping097_easy_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_all_ridge_no_guard | joint_vs_independent | 0.047288 | [-0.030076, 0.138548] |
| 43_neural_all_ridge_no_guard | joint_exact_vs_independent | 0.022872 | [-0.005844, 0.064772] |
| 43_neural_all_ridge_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 43_neural_all_ridge_source_zero_guard | joint_vs_independent | 0.016031 | [-0.004465, 0.045891] |
| 43_neural_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_all_neural_mse_no_guard | joint_vs_independent | 0.009716 | [0.003013, 0.017559] |
| 43_neural_all_neural_mse_no_guard | joint_exact_vs_independent | -0.003234 | [-0.040174, 0.022139] |
| 43_neural_all_neural_mse_no_guard | joint_exact_vs_unary | 0.000217 | [0.000000, 0.000650] |
| 43_neural_all_neural_mse_source_zero_guard | joint_vs_independent | 0.005889 | [-0.000002, 0.012366] |
| 43_neural_all_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_all_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_easy_ridge_no_guard | joint_vs_independent | -0.011978 | [-0.038859, 0.008987] |
| 43_neural_easy_ridge_no_guard | joint_exact_vs_independent | -0.035236 | [-0.115233, 0.008103] |
| 43_neural_easy_ridge_no_guard | joint_exact_vs_unary | -0.010186 | [-0.033976, 0.003417] |
| 43_neural_easy_ridge_source_zero_guard | joint_vs_independent | -0.006208 | [-0.022673, 0.004050] |
| 43_neural_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_neural_easy_neural_mse_no_guard | joint_vs_independent | -0.028356 | [-0.084745, 0.011524] |
| 43_neural_easy_neural_mse_no_guard | joint_exact_vs_independent | 0.006232 | [-0.002333, 0.020331] |
| 43_neural_easy_neural_mse_no_guard | joint_exact_vs_unary | -0.019412 | [-0.059436, 0.002223] |
| 43_neural_easy_neural_mse_source_zero_guard | joint_vs_independent | 0.004198 | [-0.003355, 0.016678] |
| 43_neural_easy_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_neural_easy_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_all_ridge_no_guard | joint_vs_independent | -0.005246 | [-0.013797, 0.001998] |
| 43_damping097_all_ridge_no_guard | joint_exact_vs_independent | -0.007916 | [-0.027228, 0.004037] |
| 43_damping097_all_ridge_no_guard | joint_exact_vs_unary | -0.004239 | [-0.017326, 0.004610] |
| 43_damping097_all_ridge_source_zero_guard | joint_vs_independent | -0.002421 | [-0.009926, 0.003485] |
| 43_damping097_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_all_neural_mse_no_guard | joint_vs_independent | -0.001757 | [-0.005874, 0.002295] |
| 43_damping097_all_neural_mse_no_guard | joint_exact_vs_independent | -0.001347 | [-0.004816, 0.001961] |
| 43_damping097_all_neural_mse_no_guard | joint_exact_vs_unary | -0.000233 | [-0.006970, 0.005955] |
| 43_damping097_all_neural_mse_source_zero_guard | joint_vs_independent | -0.000472 | [-0.003426, 0.002455] |
| 43_damping097_all_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_all_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_easy_ridge_no_guard | joint_vs_independent | -0.016608 | [-0.032737, -0.002594] |
| 43_damping097_easy_ridge_no_guard | joint_exact_vs_independent | -0.014032 | [-0.032137, -0.000673] |
| 43_damping097_easy_ridge_no_guard | joint_exact_vs_unary | 0.003988 | [-0.014000, 0.020553] |
| 43_damping097_easy_ridge_source_zero_guard | joint_vs_independent | -0.010208 | [-0.026536, 0.000000] |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_damping097_easy_neural_mse_no_guard | joint_vs_independent | -0.013279 | [-0.031853, -0.000143] |
| 43_damping097_easy_neural_mse_no_guard | joint_exact_vs_independent | -0.015348 | [-0.036980, 0.000000] |
| 43_damping097_easy_neural_mse_no_guard | joint_exact_vs_unary | 0.001668 | [0.000000, 0.005005] |
| 43_damping097_easy_neural_mse_source_zero_guard | joint_vs_independent | -0.012839 | [-0.031115, 0.000000] |
| 43_damping097_easy_neural_mse_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_damping097_easy_neural_mse_source_zero_guard | joint_exact_vs_unary | undefined | undefined |

Undefined localities remain undefined; none are dropped to manufacture a CI.
Matched counts apply within each candidate. The two candidates share predicted-risk budgets, not necessarily intervention counts.
318,969 full targets; 6,116 joint targets / 1,152 queries. Joint contains zero zero-CV examples.
3,000 locality resamples, conditional on shared development data/models; no independent safety certification.
Image-pixel obs8/pred12 raw stride12. No t50, seconds, metric, true3D, foundation or deployment claim. Stage5C/SMC off.
