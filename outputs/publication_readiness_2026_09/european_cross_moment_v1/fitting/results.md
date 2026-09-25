# Cross-Moment fitting: Every Registered View

Batch mode changes pair weighting; fitting mode changes only its normalizer.
Control: {'product': 'cross_moment_batch_control', 'hurdle': 'cross_moment_fixed_fitting_scale'}.
Treatment is cross-moment ranking. All count controls are offline diagnostics.

| View | ADE gain vs CV (%) | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Predicted violations |
|---|---:|---:|---:|---:|---|---:|---:|
| neural_fold0_seed17_all_control_original | 2.226082 | 3.317301 | 1.936496 | -2.759284 | 3/4 | 28.589237 | 0 |
| neural_fold0_seed17_all_treatment_original | 2.328782 | 3.479330 | 2.076879 | -2.696877 | 3/4 | 29.533604 | 0 |
| neural_fold0_seed17_all_control_common | 2.226082 | 3.317301 | 1.936496 | -2.759284 | 3/4 | 28.589237 | 0 |
| neural_fold0_seed17_all_treatment_common | 2.328782 | 3.479330 | 2.076879 | -2.696877 | 3/4 | 29.533604 | 0 |
| neural_fold0_seed17_all_treatment_at_control | 2.215575 | 3.308388 | 1.945471 | -2.696877 | 3/4 | 28.589237 | 0 |
| neural_fold0_seed17_all_control_at_treatment | 2.340083 | 3.483620 | 2.066520 | -2.759284 | 3/4 | 29.533604 | 1909 |
| neural_fold0_seed17_easy_control_original | 0.332789 | 0.544375 | 0.067299 | -1.762257 | 2/4 | 11.779605 | 0 |
| neural_fold0_seed17_easy_treatment_original | 0.325462 | 0.530592 | 0.051879 | -2.236478 | 2/4 | 12.404401 | 0 |
| neural_fold0_seed17_easy_control_common | 0.332789 | 0.544375 | 0.067299 | -1.762257 | 2/4 | 11.779605 | 0 |
| neural_fold0_seed17_easy_treatment_common | 0.325462 | 0.530592 | 0.051879 | -2.236478 | 2/4 | 12.404401 | 0 |
| neural_fold0_seed17_easy_treatment_at_control | 0.319670 | 0.521525 | 0.051934 | -2.086091 | 2/4 | 11.779605 | 1 |
| neural_fold0_seed17_easy_control_at_treatment | 0.339128 | 0.554850 | 0.069030 | -1.845349 | 2/4 | 12.404401 | 1264 |
| damping097_fold0_seed17_all_control_original | 4.375189 | 6.859494 | 4.540174 | 1.585499 | 0/4 | 67.610044 | 0 |
| damping097_fold0_seed17_all_treatment_original | 4.362865 | 6.809777 | 4.566727 | 1.228762 | 0/4 | 68.312507 | 0 |
| damping097_fold0_seed17_all_control_common | 4.375189 | 6.859494 | 4.540174 | 1.585499 | 0/4 | 67.610044 | 0 |
| damping097_fold0_seed17_all_treatment_common | 4.362865 | 6.809777 | 4.566727 | 1.228762 | 0/4 | 68.312507 | 0 |
| damping097_fold0_seed17_all_treatment_at_control | 4.325496 | 6.754574 | 4.499762 | 1.232569 | 0/4 | 67.610044 | 24 |
| damping097_fold0_seed17_all_control_at_treatment | 4.422187 | 6.929398 | 4.610229 | 1.384255 | 0/4 | 68.312507 | 1444 |
| damping097_fold0_seed17_easy_control_original | 0.638494 | 1.054715 | 0.207404 | -0.740510 | 0/4 | 44.288781 | 0 |
| damping097_fold0_seed17_easy_treatment_original | 0.558637 | 0.924481 | 0.131028 | -0.740510 | 0/4 | 44.090410 | 0 |
| damping097_fold0_seed17_easy_control_common | 0.638494 | 1.054715 | 0.207404 | -0.740510 | 0/4 | 44.288781 | 0 |
| damping097_fold0_seed17_easy_treatment_common | 0.558637 | 0.924481 | 0.131028 | -0.740510 | 0/4 | 44.090410 | 0 |
| damping097_fold0_seed17_easy_treatment_at_control | 0.584760 | 0.969158 | 0.145661 | -0.740510 | 0/4 | 44.288781 | 401 |
| damping097_fold0_seed17_easy_control_at_treatment | 0.612901 | 1.011093 | 0.190336 | -0.740510 | 0/4 | 44.090410 | 0 |
| neural_fold0_seed29_all_control_original | 2.111865 | 3.294770 | 1.591448 | -1.655855 | 3/4 | 34.654656 | 0 |
| neural_fold0_seed29_all_treatment_original | 2.149392 | 3.334322 | 1.598703 | -1.626532 | 3/4 | 34.732322 | 0 |
| neural_fold0_seed29_all_control_common | 2.111865 | 3.294770 | 1.591448 | -1.655855 | 3/4 | 34.654656 | 0 |
| neural_fold0_seed29_all_treatment_common | 2.149392 | 3.334322 | 1.598703 | -1.626532 | 3/4 | 34.732322 | 0 |
| neural_fold0_seed29_all_treatment_at_control | 2.139946 | 3.314155 | 1.593576 | -1.626532 | 3/4 | 34.654656 | 8 |
| neural_fold0_seed29_all_control_at_treatment | 2.166883 | 3.380499 | 1.647041 | -1.655855 | 3/4 | 34.732322 | 165 |
| neural_fold0_seed29_easy_control_original | 0.310448 | 0.533308 | 0.017749 | -2.776698 | 2/4 | 16.831399 | 0 |
| neural_fold0_seed29_easy_treatment_original | 0.346564 | 0.581963 | 0.034714 | -2.670519 | 2/4 | 18.534129 | 0 |
| neural_fold0_seed29_easy_control_common | 0.310448 | 0.533308 | 0.017749 | -2.776698 | 2/4 | 16.831399 | 0 |
| neural_fold0_seed29_easy_treatment_common | 0.346564 | 0.581963 | 0.034714 | -2.670519 | 2/4 | 18.534129 | 0 |
| neural_fold0_seed29_easy_treatment_at_control | 0.326568 | 0.548019 | 0.033228 | -2.776698 | 2/4 | 16.831399 | 5 |
| neural_fold0_seed29_easy_control_at_treatment | 0.330358 | 0.563240 | 0.025274 | -2.670385 | 2/4 | 18.534129 | 3447 |
| damping097_fold0_seed29_all_control_original | 4.494621 | 7.066909 | 4.714663 | 1.634624 | 0/4 | 65.113334 | 0 |
| damping097_fold0_seed29_all_treatment_original | 4.469176 | 7.020331 | 4.641402 | 1.830959 | 0/4 | 64.960969 | 0 |
| damping097_fold0_seed29_all_control_common | 4.494621 | 7.066909 | 4.714663 | 1.634624 | 0/4 | 65.113334 | 0 |
| damping097_fold0_seed29_all_treatment_common | 4.469176 | 7.020331 | 4.641402 | 1.830959 | 0/4 | 64.960969 | 0 |
| damping097_fold0_seed29_all_treatment_at_control | 4.519052 | 7.109845 | 4.683706 | 1.089879 | 0/4 | 65.113334 | 342 |
| damping097_fold0_seed29_all_control_at_treatment | 4.473586 | 7.031633 | 4.671444 | 1.687990 | 0/4 | 64.960969 | 34 |
| damping097_fold0_seed29_easy_control_original | 0.505857 | 0.839834 | 0.080289 | -1.263071 | 0/4 | 41.227133 | 0 |
| damping097_fold0_seed29_easy_treatment_original | 0.551591 | 0.909302 | 0.133384 | -1.284091 | 0/4 | 42.223442 | 0 |
| damping097_fold0_seed29_easy_control_common | 0.505857 | 0.839834 | 0.080289 | -1.263071 | 0/4 | 41.227133 | 0 |
| damping097_fold0_seed29_easy_treatment_common | 0.551591 | 0.909302 | 0.133384 | -1.284091 | 0/4 | 42.223442 | 0 |
| damping097_fold0_seed29_easy_treatment_at_control | 0.522081 | 0.860781 | 0.117076 | -1.284091 | 0/4 | 41.227133 | 0 |
| damping097_fold0_seed29_easy_control_at_treatment | 0.533766 | 0.884592 | 0.092403 | -1.310473 | 0/4 | 42.223442 | 2014 |
| neural_fold0_seed43_all_control_original | 2.006631 | 3.075791 | 1.407716 | -2.837189 | 4/4 | 36.300496 | 0 |
| neural_fold0_seed43_all_treatment_original | 2.492767 | 3.787298 | 2.017978 | -2.894050 | 4/4 | 37.641606 | 0 |
| neural_fold0_seed43_all_control_common | 2.006631 | 3.075791 | 1.407716 | -2.837189 | 4/4 | 36.300496 | 0 |
| neural_fold0_seed43_all_treatment_common | 2.492767 | 3.787298 | 2.017978 | -2.894050 | 4/4 | 37.641606 | 0 |
| neural_fold0_seed43_all_treatment_at_control | 2.234673 | 3.420254 | 1.735434 | -2.831733 | 4/4 | 36.300496 | 0 |
| neural_fold0_seed43_all_control_at_treatment | 2.235847 | 3.403860 | 1.647276 | -2.837189 | 4/4 | 37.641606 | 2711 |
| neural_fold0_seed43_easy_control_original | 0.373928 | 0.620363 | 0.044111 | -2.893174 | 2/4 | 19.611073 | 0 |
| neural_fold0_seed43_easy_treatment_original | 0.423960 | 0.699662 | 0.081615 | -3.011298 | 2/4 | 20.274455 | 0 |
| neural_fold0_seed43_easy_control_common | 0.373928 | 0.620363 | 0.044111 | -2.893174 | 2/4 | 19.611073 | 0 |
| neural_fold0_seed43_easy_treatment_common | 0.423960 | 0.699662 | 0.081615 | -3.011298 | 2/4 | 20.274455 | 0 |
| neural_fold0_seed43_easy_treatment_at_control | 0.407987 | 0.673740 | 0.075138 | -2.807935 | 2/4 | 19.611073 | 7 |
| neural_fold0_seed43_easy_control_at_treatment | 0.387314 | 0.639192 | 0.045463 | -3.193386 | 2/4 | 20.274455 | 1348 |
| damping097_fold0_seed43_all_control_original | 4.630725 | 7.229988 | 4.824467 | 1.860211 | 0/4 | 70.688018 | 0 |
| damping097_fold0_seed43_all_treatment_original | 4.640884 | 7.296842 | 4.881531 | 1.860211 | 0/4 | 70.909145 | 0 |
| damping097_fold0_seed43_all_control_common | 4.630725 | 7.229988 | 4.824467 | 1.860211 | 0/4 | 70.688018 | 0 |
| damping097_fold0_seed43_all_treatment_common | 4.640884 | 7.296842 | 4.881531 | 1.860211 | 0/4 | 70.909145 | 0 |
| damping097_fold0_seed43_all_treatment_at_control | 4.610781 | 7.231870 | 4.836734 | 1.853480 | 0/4 | 70.688018 | 28 |
| damping097_fold0_seed43_all_control_at_treatment | 4.662057 | 7.296942 | 4.865103 | 1.846917 | 0/4 | 70.909145 | 475 |
| damping097_fold0_seed43_easy_control_original | 0.701725 | 1.188686 | 0.282024 | -0.958676 | 0/4 | 45.698159 | 0 |
| damping097_fold0_seed43_easy_treatment_original | 0.891320 | 1.477112 | 0.518702 | -1.251739 | 0/4 | 46.652914 | 0 |
| damping097_fold0_seed43_easy_control_common | 0.701725 | 1.188686 | 0.282024 | -0.958676 | 0/4 | 45.698159 | 0 |
| damping097_fold0_seed43_easy_treatment_common | 0.891320 | 1.477112 | 0.518702 | -1.251739 | 0/4 | 46.652914 | 0 |
| damping097_fold0_seed43_easy_treatment_at_control | 0.826960 | 1.372115 | 0.469524 | -1.194050 | 0/4 | 45.698159 | 0 |
| damping097_fold0_seed43_easy_control_at_treatment | 0.752658 | 1.257089 | 0.325894 | -0.803699 | 0/4 | 46.652914 | 1930 |
| neural_fold1_seed17_all_control_original | 3.281879 | 4.789355 | 2.331359 | -3.662656 | 4/4 | 53.522444 | 0 |
| neural_fold1_seed17_all_treatment_original | 3.303577 | 4.833214 | 2.389235 | -3.778932 | 4/4 | 53.176923 | 0 |
| neural_fold1_seed17_all_control_common | 3.281879 | 4.789355 | 2.331359 | -3.662656 | 4/4 | 53.522444 | 0 |
| neural_fold1_seed17_all_treatment_common | 3.303577 | 4.833214 | 2.389235 | -3.778932 | 4/4 | 53.176923 | 0 |
| neural_fold1_seed17_all_treatment_at_control | 3.323970 | 4.880466 | 2.407143 | -3.561882 | 4/4 | 53.522444 | 1055 |
| neural_fold1_seed17_all_control_at_treatment | 3.240716 | 4.720322 | 2.322199 | -3.650380 | 4/4 | 53.176923 | 35 |
| neural_fold1_seed17_easy_control_original | 0.378773 | 0.661296 | 0.056568 | -2.817163 | 1/4 | 14.179542 | 0 |
| neural_fold1_seed17_easy_treatment_original | 0.445506 | 0.766966 | 0.055208 | -3.316268 | 1/4 | 15.753691 | 0 |
| neural_fold1_seed17_easy_control_common | 0.378773 | 0.661296 | 0.056568 | -2.817163 | 1/4 | 14.179542 | 0 |
| neural_fold1_seed17_easy_treatment_common | 0.445506 | 0.766966 | 0.055208 | -3.316268 | 1/4 | 15.753691 | 0 |
| neural_fold1_seed17_easy_treatment_at_control | 0.396004 | 0.683561 | 0.051923 | -2.686525 | 1/4 | 14.179542 | 0 |
| neural_fold1_seed17_easy_control_at_treatment | 0.430249 | 0.745950 | 0.059119 | -3.373010 | 1/4 | 15.753691 | 4647 |
| damping097_fold1_seed17_all_control_original | 4.366782 | 6.889977 | 4.339656 | -0.564990 | 0/4 | 66.114625 | 0 |
| damping097_fold1_seed17_all_treatment_original | 4.416599 | 6.944509 | 4.529768 | -0.790655 | 0/4 | 66.940825 | 0 |
| damping097_fold1_seed17_all_control_common | 4.366782 | 6.889977 | 4.339656 | -0.564990 | 0/4 | 66.114625 | 0 |
| damping097_fold1_seed17_all_treatment_common | 4.416599 | 6.944509 | 4.529768 | -0.790655 | 0/4 | 66.940825 | 0 |
| damping097_fold1_seed17_all_treatment_at_control | 4.344629 | 6.847921 | 4.415267 | -0.547549 | 0/4 | 66.114625 | 0 |
| damping097_fold1_seed17_all_control_at_treatment | 4.429521 | 7.004295 | 4.541739 | 0.233289 | 0/4 | 66.940825 | 2439 |
| damping097_fold1_seed17_easy_control_original | 0.808908 | 1.358741 | 0.222619 | -0.309701 | 0/4 | 39.620334 | 0 |
| damping097_fold1_seed17_easy_treatment_original | 0.684946 | 1.158565 | 0.147025 | -1.687588 | 0/4 | 34.672958 | 0 |
| damping097_fold1_seed17_easy_control_common | 0.808908 | 1.358741 | 0.222619 | -0.309701 | 0/4 | 39.620334 | 0 |
| damping097_fold1_seed17_easy_treatment_common | 0.684946 | 1.158565 | 0.147025 | -1.687588 | 0/4 | 34.672958 | 0 |
| damping097_fold1_seed17_easy_treatment_at_control | 0.916055 | 1.508303 | 0.425907 | 0.478697 | 0/4 | 39.620334 | 14605 |
| damping097_fold1_seed17_easy_control_at_treatment | 0.645231 | 1.087495 | 0.129736 | -1.344685 | 0/4 | 34.672958 | 0 |
| neural_fold1_seed29_all_control_original | 3.297682 | 4.873784 | 2.601430 | -4.329509 | 3/4 | 43.407846 | 0 |
| neural_fold1_seed29_all_treatment_original | 3.202596 | 4.638396 | 2.210366 | -4.438819 | 3/4 | 43.173095 | 0 |
| neural_fold1_seed29_all_control_common | 3.297682 | 4.873784 | 2.601430 | -4.329509 | 3/4 | 43.407846 | 0 |
| neural_fold1_seed29_all_treatment_common | 3.202596 | 4.638396 | 2.210366 | -4.438819 | 3/4 | 43.173095 | 0 |
| neural_fold1_seed29_all_treatment_at_control | 3.197300 | 4.667950 | 2.196049 | -4.187846 | 3/4 | 43.407846 | 802 |
| neural_fold1_seed29_all_control_at_treatment | 3.277714 | 4.824107 | 2.478390 | -4.322909 | 3/4 | 43.173095 | 109 |
| neural_fold1_seed29_easy_control_original | 0.471294 | 0.837976 | 0.077317 | -3.178346 | 1/4 | 14.324186 | 0 |
| neural_fold1_seed29_easy_treatment_original | 0.536963 | 0.921212 | 0.108034 | -2.895069 | 1/4 | 15.916628 | 0 |
| neural_fold1_seed29_easy_control_common | 0.471294 | 0.837976 | 0.077317 | -3.178346 | 1/4 | 14.324186 | 0 |
| neural_fold1_seed29_easy_treatment_common | 0.536963 | 0.921212 | 0.108034 | -2.895069 | 1/4 | 15.916628 | 0 |
| neural_fold1_seed29_easy_treatment_at_control | 0.495718 | 0.856108 | 0.106418 | -2.820387 | 1/4 | 14.324186 | 0 |
| neural_fold1_seed29_easy_control_at_treatment | 0.514108 | 0.906625 | 0.081680 | -3.330751 | 1/4 | 15.916628 | 4701 |
| damping097_fold1_seed29_all_control_original | 4.384809 | 7.004164 | 4.694431 | -0.438749 | 0/4 | 65.520804 | 0 |
| damping097_fold1_seed29_all_treatment_original | 4.542899 | 7.368262 | 5.007385 | 0.881553 | 0/4 | 66.008259 | 0 |
| damping097_fold1_seed29_all_control_common | 4.384809 | 7.004164 | 4.694431 | -0.438749 | 0/4 | 65.520804 | 0 |
| damping097_fold1_seed29_all_treatment_common | 4.542899 | 7.368262 | 5.007385 | 0.881553 | 0/4 | 66.008259 | 0 |
| damping097_fold1_seed29_all_treatment_at_control | 4.490696 | 7.196978 | 4.907921 | -0.343500 | 0/4 | 65.520804 | 0 |
| damping097_fold1_seed29_all_control_at_treatment | 4.446970 | 7.205229 | 4.818504 | 0.175989 | 0/4 | 66.008259 | 1439 |
| damping097_fold1_seed29_easy_control_original | 1.039105 | 1.645581 | 0.496558 | -1.230417 | 0/4 | 42.061333 | 0 |
| damping097_fold1_seed29_easy_treatment_original | 0.765570 | 1.324283 | 0.215643 | -1.653257 | 0/4 | 36.346360 | 0 |
| damping097_fold1_seed29_easy_control_common | 1.039105 | 1.645581 | 0.496558 | -1.230417 | 0/4 | 42.061333 | 0 |
| damping097_fold1_seed29_easy_treatment_common | 0.765570 | 1.324283 | 0.215643 | -1.653257 | 0/4 | 36.346360 | 0 |
| damping097_fold1_seed29_easy_treatment_at_control | 1.034361 | 1.731345 | 0.377537 | -0.923495 | 0/4 | 42.061333 | 16871 |
| damping097_fold1_seed29_easy_control_at_treatment | 0.793939 | 1.333379 | 0.295936 | -1.653257 | 0/4 | 36.346360 | 0 |
| neural_fold1_seed43_all_control_original | 3.475632 | 5.163616 | 2.616718 | -2.785595 | 4/4 | 51.435772 | 0 |
| neural_fold1_seed43_all_treatment_original | 3.382369 | 5.016233 | 2.538406 | -2.814784 | 4/4 | 50.332140 | 0 |
| neural_fold1_seed43_all_control_common | 3.475632 | 5.163616 | 2.616718 | -2.785595 | 4/4 | 51.435772 | 0 |
| neural_fold1_seed43_all_treatment_common | 3.382369 | 5.016233 | 2.538406 | -2.814784 | 4/4 | 50.332140 | 0 |
| neural_fold1_seed43_all_treatment_at_control | 3.524493 | 5.211844 | 2.667804 | -2.735346 | 4/4 | 51.435772 | 3264 |
| neural_fold1_seed43_all_control_at_treatment | 3.368962 | 5.018460 | 2.518290 | -2.776077 | 4/4 | 50.332140 | 6 |
| neural_fold1_seed43_easy_control_original | 0.457482 | 0.804443 | 0.095600 | -2.492550 | 1/4 | 16.656787 | 0 |
| neural_fold1_seed43_easy_treatment_original | 0.408695 | 0.724192 | 0.071384 | -2.495144 | 1/4 | 15.985732 | 0 |
| neural_fold1_seed43_easy_control_common | 0.457482 | 0.804443 | 0.095600 | -2.492550 | 1/4 | 16.656787 | 0 |
| neural_fold1_seed43_easy_treatment_common | 0.408695 | 0.724192 | 0.071384 | -2.495144 | 1/4 | 15.985732 | 0 |
| neural_fold1_seed43_easy_treatment_at_control | 0.422513 | 0.749834 | 0.075164 | -2.531669 | 1/4 | 16.656787 | 2201 |
| neural_fold1_seed43_easy_control_at_treatment | 0.446450 | 0.786700 | 0.092271 | -2.432649 | 1/4 | 15.985732 | 220 |
| damping097_fold1_seed43_all_control_original | 4.506847 | 7.184891 | 4.882285 | 0.252048 | 0/4 | 65.708469 | 0 |
| damping097_fold1_seed43_all_treatment_original | 4.460525 | 7.060325 | 4.802528 | 0.233173 | 0/4 | 65.786380 | 0 |
| damping097_fold1_seed43_all_control_common | 4.506847 | 7.184891 | 4.882285 | 0.252048 | 0/4 | 65.708469 | 0 |
| damping097_fold1_seed43_all_treatment_common | 4.460525 | 7.060325 | 4.802528 | 0.233173 | 0/4 | 65.786380 | 0 |
| damping097_fold1_seed43_all_treatment_at_control | 4.442924 | 7.026818 | 4.768251 | 0.233173 | 0/4 | 65.708469 | 44 |
| damping097_fold1_seed43_all_control_at_treatment | 4.523257 | 7.203485 | 4.883645 | 0.233173 | 0/4 | 65.786380 | 274 |
| damping097_fold1_seed43_easy_control_original | 0.813741 | 1.371460 | 0.208664 | -1.211075 | 0/4 | 39.629819 | 0 |
| damping097_fold1_seed43_easy_treatment_original | 0.709043 | 1.188332 | 0.254777 | -1.164963 | 0/4 | 36.551979 | 0 |
| damping097_fold1_seed43_easy_control_common | 0.813741 | 1.371460 | 0.208664 | -1.211075 | 0/4 | 39.629819 | 0 |
| damping097_fold1_seed43_easy_treatment_common | 0.709043 | 1.188332 | 0.254777 | -1.164963 | 0/4 | 36.551979 | 0 |
| damping097_fold1_seed43_easy_treatment_at_control | 0.935215 | 1.572805 | 0.377568 | -0.481313 | 0/4 | 39.629819 | 9086 |
| damping097_fold1_seed43_easy_control_at_treatment | 0.601469 | 1.022851 | 0.123960 | -1.183970 | 0/4 | 36.551979 | 0 |
| neural_fold2_seed17_all_control_original | 0.613641 | 0.874707 | 0.599564 | 0.131986 | 0/0 | 10.290572 | 0 |
| neural_fold2_seed17_all_treatment_original | 0.654525 | 0.910119 | 0.649111 | -0.202046 | 0/0 | 10.962051 | 0 |
| neural_fold2_seed17_all_control_common | 0.613641 | 0.874707 | 0.599564 | 0.131986 | 0/0 | 10.290572 | 0 |
| neural_fold2_seed17_all_treatment_common | 0.654525 | 0.910119 | 0.649111 | -0.202046 | 0/0 | 10.962051 | 0 |
| neural_fold2_seed17_all_treatment_at_control | 0.612894 | 0.850179 | 0.608721 | -0.010071 | 0/0 | 10.290572 | 0 |
| neural_fold2_seed17_all_control_at_treatment | 0.637881 | 0.944046 | 0.619923 | 0.010772 | 0/0 | 10.962051 | 944 |
| neural_fold2_seed17_easy_control_original | 0.144482 | 0.153073 | 0.089705 | 0.173793 | 0/0 | 0.482271 | 0 |
| neural_fold2_seed17_easy_treatment_original | 0.144661 | 0.150742 | 0.091630 | 0.172195 | 0/0 | 0.495074 | 0 |
| neural_fold2_seed17_easy_control_common | 0.144482 | 0.153073 | 0.089705 | 0.173793 | 0/0 | 0.482271 | 0 |
| neural_fold2_seed17_easy_treatment_common | 0.144661 | 0.150742 | 0.091630 | 0.172195 | 0/0 | 0.495074 | 0 |
| neural_fold2_seed17_easy_treatment_at_control | 0.143715 | 0.151804 | 0.090900 | 0.163249 | 0/0 | 0.482271 | 8 |
| neural_fold2_seed17_easy_control_at_treatment | 0.144621 | 0.153327 | 0.089765 | 0.157848 | 0/0 | 0.495074 | 26 |
| damping097_fold2_seed17_all_control_original | 3.419451 | 5.460161 | 2.988070 | -2.843555 | 0/0 | 64.849024 | 0 |
| damping097_fold2_seed17_all_treatment_original | 3.413146 | 5.447159 | 2.966700 | -2.913036 | 0/0 | 64.565921 | 0 |
| damping097_fold2_seed17_all_control_common | 3.419451 | 5.460161 | 2.988070 | -2.843555 | 0/0 | 64.849024 | 0 |
| damping097_fold2_seed17_all_treatment_common | 3.413146 | 5.447159 | 2.966700 | -2.913036 | 0/0 | 64.565921 | 0 |
| damping097_fold2_seed17_all_treatment_at_control | 3.416420 | 5.451093 | 2.968142 | -2.867770 | 0/0 | 64.849024 | 421 |
| damping097_fold2_seed17_all_control_at_treatment | 3.432026 | 5.489121 | 3.010066 | -2.891154 | 0/0 | 64.565921 | 23 |
| damping097_fold2_seed17_easy_control_original | 0.466020 | 0.762282 | 0.169469 | -3.380052 | 0/0 | 28.257638 | 0 |
| damping097_fold2_seed17_easy_treatment_original | 0.557041 | 0.910743 | 0.224853 | -3.248427 | 0/0 | 29.316783 | 0 |
| damping097_fold2_seed17_easy_control_common | 0.466020 | 0.762282 | 0.169469 | -3.380052 | 0/0 | 28.257638 | 0 |
| damping097_fold2_seed17_easy_treatment_common | 0.557041 | 0.910743 | 0.224853 | -3.248427 | 0/0 | 29.316783 | 0 |
| damping097_fold2_seed17_easy_treatment_at_control | 0.526361 | 0.863321 | 0.213708 | -3.181368 | 0/0 | 28.257638 | 1 |
| damping097_fold2_seed17_easy_control_at_treatment | 0.499060 | 0.816890 | 0.187884 | -3.471680 | 0/0 | 29.316783 | 1490 |
| neural_fold2_seed29_all_control_original | 0.634182 | 0.817347 | 0.571158 | -0.122335 | 0/0 | 7.679340 | 0 |
| neural_fold2_seed29_all_treatment_original | 0.589195 | 0.813889 | 0.487485 | -0.130158 | 0/0 | 7.922609 | 0 |
| neural_fold2_seed29_all_control_common | 0.634182 | 0.817347 | 0.571158 | -0.122335 | 0/0 | 7.679340 | 0 |
| neural_fold2_seed29_all_treatment_common | 0.589195 | 0.813889 | 0.487485 | -0.130158 | 0/0 | 7.922609 | 0 |
| neural_fold2_seed29_all_treatment_at_control | 0.580078 | 0.793685 | 0.483300 | -0.231359 | 0/0 | 7.679340 | 1 |
| neural_fold2_seed29_all_control_at_treatment | 0.668480 | 0.855056 | 0.605547 | -0.047257 | 0/0 | 7.922609 | 343 |
| neural_fold2_seed29_easy_control_original | 0.276658 | 0.317228 | 0.106861 | -0.223220 | 0/0 | 0.826546 | 0 |
| neural_fold2_seed29_easy_treatment_original | 0.281422 | 0.315371 | 0.114118 | -0.225669 | 0/0 | 0.820144 | 0 |
| neural_fold2_seed29_easy_control_common | 0.276658 | 0.317228 | 0.106861 | -0.223220 | 0/0 | 0.826546 | 0 |
| neural_fold2_seed29_easy_treatment_common | 0.281422 | 0.315371 | 0.114118 | -0.225669 | 0/0 | 0.820144 | 0 |
| neural_fold2_seed29_easy_treatment_at_control | 0.275097 | 0.314942 | 0.101960 | -0.228243 | 0/0 | 0.826546 | 15 |
| neural_fold2_seed29_easy_control_at_treatment | 0.274510 | 0.314136 | 0.102500 | -0.221626 | 0/0 | 0.820144 | 6 |
| damping097_fold2_seed29_all_control_original | 3.577220 | 5.670895 | 3.290416 | -2.190022 | 0/0 | 59.351282 | 0 |
| damping097_fold2_seed29_all_treatment_original | 3.474267 | 5.503250 | 3.171890 | -2.235028 | 0/0 | 58.364690 | 0 |
| damping097_fold2_seed29_all_control_common | 3.577220 | 5.670895 | 3.290416 | -2.190022 | 0/0 | 59.351282 | 0 |
| damping097_fold2_seed29_all_treatment_common | 3.474267 | 5.503250 | 3.171890 | -2.235028 | 0/0 | 58.364690 | 0 |
| damping097_fold2_seed29_all_treatment_at_control | 3.514251 | 5.548170 | 3.210051 | -2.190022 | 0/0 | 59.351282 | 1408 |
| damping097_fold2_seed29_all_control_at_treatment | 3.532092 | 5.596723 | 3.253081 | -2.190022 | 0/0 | 58.364690 | 21 |
| damping097_fold2_seed29_easy_control_original | 0.425814 | 0.703574 | 0.127930 | -2.666981 | 0/0 | 26.482199 | 0 |
| damping097_fold2_seed29_easy_treatment_original | 0.491559 | 0.795018 | 0.225026 | -2.413911 | 0/0 | 24.385959 | 0 |
| damping097_fold2_seed29_easy_control_common | 0.425814 | 0.703574 | 0.127930 | -2.666981 | 0/0 | 26.482199 | 0 |
| damping097_fold2_seed29_easy_treatment_common | 0.491559 | 0.795018 | 0.225026 | -2.413911 | 0/0 | 24.385959 | 0 |
| damping097_fold2_seed29_easy_treatment_at_control | 0.529095 | 0.849438 | 0.228830 | -2.317242 | 0/0 | 26.482199 | 3024 |
| damping097_fold2_seed29_easy_control_at_treatment | 0.399016 | 0.656353 | 0.131302 | -2.688701 | 0/0 | 24.385959 | 77 |
| neural_fold2_seed43_all_control_original | 0.561045 | 0.839171 | 0.531454 | -0.487873 | 0/0 | 9.839599 | 0 |
| neural_fold2_seed43_all_treatment_original | 0.536681 | 0.799545 | 0.506786 | -0.527903 | 0/0 | 10.009603 | 0 |
| neural_fold2_seed43_all_control_common | 0.561045 | 0.839171 | 0.531454 | -0.487873 | 0/0 | 9.839599 | 0 |
| neural_fold2_seed43_all_treatment_common | 0.536681 | 0.799545 | 0.506786 | -0.527903 | 0/0 | 10.009603 | 0 |
| neural_fold2_seed43_all_treatment_at_control | 0.515810 | 0.773361 | 0.479223 | -0.369508 | 0/0 | 9.839599 | 0 |
| neural_fold2_seed43_all_control_at_treatment | 0.577375 | 0.856140 | 0.550918 | -0.478813 | 0/0 | 10.009603 | 239 |
| neural_fold2_seed43_easy_control_original | 0.157530 | 0.165736 | 0.076813 | -0.033643 | 0/0 | 0.428211 | 0 |
| neural_fold2_seed43_easy_treatment_original | 0.158199 | 0.165893 | 0.078975 | -0.031968 | 0/0 | 0.443148 | 0 |
| neural_fold2_seed43_easy_control_common | 0.157530 | 0.165736 | 0.076813 | -0.033643 | 0/0 | 0.428211 | 0 |
| neural_fold2_seed43_easy_treatment_common | 0.158199 | 0.165893 | 0.078975 | -0.031968 | 0/0 | 0.443148 | 0 |
| neural_fold2_seed43_easy_treatment_at_control | 0.158566 | 0.166454 | 0.078975 | -0.032639 | 0/0 | 0.428211 | 2 |
| neural_fold2_seed43_easy_control_at_treatment | 0.157063 | 0.165609 | 0.076813 | -0.033022 | 0/0 | 0.443148 | 23 |
| damping097_fold2_seed43_all_control_original | 3.452928 | 5.468897 | 3.134996 | -2.233540 | 0/0 | 60.789558 | 0 |
| damping097_fold2_seed43_all_treatment_original | 3.394463 | 5.386980 | 3.069190 | -2.259264 | 0/0 | 60.212683 | 0 |
| damping097_fold2_seed43_all_control_common | 3.452928 | 5.468897 | 3.134996 | -2.233540 | 0/0 | 60.789558 | 0 |
| damping097_fold2_seed43_all_treatment_common | 3.394463 | 5.386980 | 3.069190 | -2.259264 | 0/0 | 60.212683 | 0 |
| damping097_fold2_seed43_all_treatment_at_control | 3.460779 | 5.517841 | 3.131844 | -2.232582 | 0/0 | 60.789558 | 811 |
| damping097_fold2_seed43_all_control_at_treatment | 3.377850 | 5.376542 | 3.044714 | -2.233421 | 0/0 | 60.212683 | 0 |
| damping097_fold2_seed43_easy_control_original | 0.318491 | 0.514958 | 0.086261 | -2.672612 | 0/0 | 23.088523 | 0 |
| damping097_fold2_seed43_easy_treatment_original | 0.433275 | 0.710758 | 0.138432 | -2.724553 | 0/0 | 25.903190 | 0 |
| damping097_fold2_seed43_easy_control_common | 0.318491 | 0.514958 | 0.086261 | -2.672612 | 0/0 | 23.088523 | 0 |
| damping097_fold2_seed43_easy_treatment_common | 0.433275 | 0.710758 | 0.138432 | -2.724553 | 0/0 | 25.903190 | 0 |
| damping097_fold2_seed43_easy_treatment_at_control | 0.337530 | 0.550976 | 0.093263 | -2.478369 | 0/0 | 23.088523 | 0 |
| damping097_fold2_seed43_easy_control_at_treatment | 0.399072 | 0.648041 | 0.115608 | -2.759596 | 0/0 | 25.903190 | 3957 |

## Ranking and Coverage Components

Both anchors retained. Positive ranks favor the new ordering. Units: pp of CV-normalized ADE.
Exact accounting, not unique causal mediation. Common-pool and full-anchor effects both retained.

| Group | Subset | Component | Mean (pp) | Conditional 95% CI (pp) |
|---|---|---|---:|---|
| neural_fold0_seed17_all | all | full_total | 0.102700 | [0.036149, 0.162996] |
| neural_fold0_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | all | total | 0.102700 | [0.036149, 0.162996] |
| neural_fold0_seed17_all | all | ranking_at_control_count | -0.010507 | [-0.043800, 0.020952] |
| neural_fold0_seed17_all | all | ranking_at_treatment_count | -0.011301 | [-0.061429, 0.037068] |
| neural_fold0_seed17_all | all | coverage_with_treatment_ranking | 0.113207 | [0.056791, 0.162220] |
| neural_fold0_seed17_all | all | coverage_with_control_ranking | 0.114001 | [0.060279, 0.165154] |
| neural_fold0_seed17_all | easy | full_total | -0.051985 | [-0.104589, -0.003016] |
| neural_fold0_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | easy | total | -0.051985 | [-0.104589, -0.003016] |
| neural_fold0_seed17_all | easy | ranking_at_control_count | -0.043882 | [-0.062943, -0.027112] |
| neural_fold0_seed17_all | easy | ranking_at_treatment_count | -0.043244 | [-0.102403, 0.015338] |
| neural_fold0_seed17_all | easy | coverage_with_treatment_ranking | -0.008102 | [-0.046034, 0.029102] |
| neural_fold0_seed17_all | easy | coverage_with_control_ranking | -0.008741 | [-0.045618, 0.018327] |
| neural_fold0_seed17_all | hard | full_total | 0.140383 | [0.052705, 0.227153] |
| neural_fold0_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | hard | total | 0.140383 | [0.052705, 0.227153] |
| neural_fold0_seed17_all | hard | ranking_at_control_count | 0.008975 | [-0.028874, 0.038099] |
| neural_fold0_seed17_all | hard | ranking_at_treatment_count | 0.010359 | [-0.048760, 0.065409] |
| neural_fold0_seed17_all | hard | coverage_with_treatment_ranking | 0.131408 | [0.056775, 0.205452] |
| neural_fold0_seed17_all | hard | coverage_with_control_ranking | 0.130024 | [0.055727, 0.206009] |
| neural_fold0_seed17_easy | all | full_total | -0.007327 | [-0.019327, 0.003960] |
| neural_fold0_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | all | total | -0.007327 | [-0.019327, 0.003960] |
| neural_fold0_seed17_easy | all | ranking_at_control_count | -0.013120 | [-0.022048, -0.005696] |
| neural_fold0_seed17_easy | all | ranking_at_treatment_count | -0.013666 | [-0.022295, -0.007280] |
| neural_fold0_seed17_easy | all | coverage_with_treatment_ranking | 0.005793 | [0.001286, 0.012766] |
| neural_fold0_seed17_easy | all | coverage_with_control_ranking | 0.006339 | [-0.000218, 0.014030] |
| neural_fold0_seed17_easy | easy | full_total | 0.188385 | [0.063564, 0.317793] |
| neural_fold0_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | easy | total | 0.188385 | [0.063564, 0.317793] |
| neural_fold0_seed17_easy | easy | ranking_at_control_count | 0.050270 | [-0.052909, 0.171786] |
| neural_fold0_seed17_easy | easy | ranking_at_treatment_count | 0.046105 | [-0.042680, 0.161294] |
| neural_fold0_seed17_easy | easy | coverage_with_treatment_ranking | 0.138115 | [0.054375, 0.229241] |
| neural_fold0_seed17_easy | easy | coverage_with_control_ranking | 0.142280 | [0.067765, 0.220525] |
| neural_fold0_seed17_easy | hard | full_total | -0.015421 | [-0.036510, -0.001834] |
| neural_fold0_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | hard | total | -0.015421 | [-0.036510, -0.001834] |
| neural_fold0_seed17_easy | hard | ranking_at_control_count | -0.015365 | [-0.036421, -0.001906] |
| neural_fold0_seed17_easy | hard | ranking_at_treatment_count | -0.017151 | [-0.038095, -0.003141] |
| neural_fold0_seed17_easy | hard | coverage_with_treatment_ranking | -0.000056 | [-0.000322, 0.000129] |
| neural_fold0_seed17_easy | hard | coverage_with_control_ranking | 0.001730 | [0.000095, 0.004436] |
| damping097_fold0_seed17_all | all | full_total | -0.012324 | [-0.088464, 0.050941] |
| damping097_fold0_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | all | total | -0.012324 | [-0.088464, 0.050941] |
| damping097_fold0_seed17_all | all | ranking_at_control_count | -0.049693 | [-0.089861, -0.004062] |
| damping097_fold0_seed17_all | all | ranking_at_treatment_count | -0.059322 | [-0.112987, -0.013120] |
| damping097_fold0_seed17_all | all | coverage_with_treatment_ranking | 0.037369 | [-0.010865, 0.081793] |
| damping097_fold0_seed17_all | all | coverage_with_control_ranking | 0.046998 | [-0.003311, 0.095991] |
| damping097_fold0_seed17_all | easy | full_total | -0.001213 | [-0.127070, 0.128839] |
| damping097_fold0_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | easy | total | -0.001213 | [-0.127070, 0.128839] |
| damping097_fold0_seed17_all | easy | ranking_at_control_count | 0.041078 | [-0.030826, 0.142014] |
| damping097_fold0_seed17_all | easy | ranking_at_treatment_count | 0.000928 | [-0.044200, 0.053661] |
| damping097_fold0_seed17_all | easy | coverage_with_treatment_ranking | -0.042292 | [-0.121925, 0.010784] |
| damping097_fold0_seed17_all | easy | coverage_with_control_ranking | -0.002141 | [-0.086376, 0.074929] |
| damping097_fold0_seed17_all | hard | full_total | 0.026553 | [-0.012211, 0.067905] |
| damping097_fold0_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | hard | total | 0.026553 | [-0.012211, 0.067905] |
| damping097_fold0_seed17_all | hard | ranking_at_control_count | -0.040412 | [-0.085250, 0.010056] |
| damping097_fold0_seed17_all | hard | ranking_at_treatment_count | -0.043502 | [-0.108071, 0.005041] |
| damping097_fold0_seed17_all | hard | coverage_with_treatment_ranking | 0.066965 | [0.029152, 0.104418] |
| damping097_fold0_seed17_all | hard | coverage_with_control_ranking | 0.070055 | [0.022016, 0.119356] |
| damping097_fold0_seed17_easy | all | full_total | -0.079857 | [-0.124087, -0.041549] |
| damping097_fold0_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | all | total | -0.079857 | [-0.124087, -0.041549] |
| damping097_fold0_seed17_easy | all | ranking_at_control_count | -0.053735 | [-0.082878, -0.028167] |
| damping097_fold0_seed17_easy | all | ranking_at_treatment_count | -0.054264 | [-0.086292, -0.026908] |
| damping097_fold0_seed17_easy | all | coverage_with_treatment_ranking | -0.026123 | [-0.045825, -0.008683] |
| damping097_fold0_seed17_easy | all | coverage_with_control_ranking | -0.025594 | [-0.044768, -0.009140] |
| damping097_fold0_seed17_easy | easy | full_total | 0.000106 | [-0.017376, 0.016191] |
| damping097_fold0_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | easy | total | 0.000106 | [-0.017376, 0.016191] |
| damping097_fold0_seed17_easy | easy | ranking_at_control_count | -0.004854 | [-0.021791, 0.010764] |
| damping097_fold0_seed17_easy | easy | ranking_at_treatment_count | 0.000731 | [-0.012871, 0.012678] |
| damping097_fold0_seed17_easy | easy | coverage_with_treatment_ranking | 0.004960 | [-0.001120, 0.012689] |
| damping097_fold0_seed17_easy | easy | coverage_with_control_ranking | -0.000625 | [-0.005275, 0.004334] |
| damping097_fold0_seed17_easy | hard | full_total | -0.076376 | [-0.128080, -0.029406] |
| damping097_fold0_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | hard | total | -0.076376 | [-0.128080, -0.029406] |
| damping097_fold0_seed17_easy | hard | ranking_at_control_count | -0.061743 | [-0.104321, -0.023892] |
| damping097_fold0_seed17_easy | hard | ranking_at_treatment_count | -0.059308 | [-0.103782, -0.021352] |
| damping097_fold0_seed17_easy | hard | coverage_with_treatment_ranking | -0.014633 | [-0.027966, -0.004732] |
| damping097_fold0_seed17_easy | hard | coverage_with_control_ranking | -0.017068 | [-0.031510, -0.005049] |
| neural_fold0_seed29_all | all | full_total | 0.037527 | [-0.002457, 0.075904] |
| neural_fold0_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | all | total | 0.037527 | [-0.002457, 0.075904] |
| neural_fold0_seed29_all | all | ranking_at_control_count | 0.028081 | [-0.011618, 0.066035] |
| neural_fold0_seed29_all | all | ranking_at_treatment_count | -0.017490 | [-0.057116, 0.026698] |
| neural_fold0_seed29_all | all | coverage_with_treatment_ranking | 0.009446 | [-0.025110, 0.042217] |
| neural_fold0_seed29_all | all | coverage_with_control_ranking | 0.055018 | [0.016225, 0.099047] |
| neural_fold0_seed29_all | easy | full_total | 0.033926 | [-0.024284, 0.095919] |
| neural_fold0_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | easy | total | 0.033926 | [-0.024284, 0.095919] |
| neural_fold0_seed29_all | easy | ranking_at_control_count | 0.118733 | [-0.011403, 0.333114] |
| neural_fold0_seed29_all | easy | ranking_at_treatment_count | 0.008805 | [-0.039087, 0.059885] |
| neural_fold0_seed29_all | easy | coverage_with_treatment_ranking | -0.084807 | [-0.246812, 0.010688] |
| neural_fold0_seed29_all | easy | coverage_with_control_ranking | 0.025121 | [0.006214, 0.047343] |
| neural_fold0_seed29_all | hard | full_total | 0.007255 | [-0.026259, 0.045953] |
| neural_fold0_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | hard | total | 0.007255 | [-0.026259, 0.045953] |
| neural_fold0_seed29_all | hard | ranking_at_control_count | 0.002128 | [-0.041984, 0.045855] |
| neural_fold0_seed29_all | hard | ranking_at_treatment_count | -0.048338 | [-0.116035, 0.012621] |
| neural_fold0_seed29_all | hard | coverage_with_treatment_ranking | 0.005127 | [-0.033175, 0.045136] |
| neural_fold0_seed29_all | hard | coverage_with_control_ranking | 0.055593 | [0.012942, 0.106774] |
| neural_fold0_seed29_easy | all | full_total | 0.036116 | [0.009873, 0.068612] |
| neural_fold0_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | all | total | 0.036116 | [0.009873, 0.068612] |
| neural_fold0_seed29_easy | all | ranking_at_control_count | 0.016120 | [0.006488, 0.027033] |
| neural_fold0_seed29_easy | all | ranking_at_treatment_count | 0.016205 | [0.009362, 0.024307] |
| neural_fold0_seed29_easy | all | coverage_with_treatment_ranking | 0.019996 | [0.002228, 0.045164] |
| neural_fold0_seed29_easy | all | coverage_with_control_ranking | 0.019911 | [-0.006204, 0.049728] |
| neural_fold0_seed29_easy | easy | full_total | 0.316179 | [0.139623, 0.500288] |
| neural_fold0_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | easy | total | 0.316179 | [0.139623, 0.500288] |
| neural_fold0_seed29_easy | easy | ranking_at_control_count | -0.048099 | [-0.105841, 0.004229] |
| neural_fold0_seed29_easy | easy | ranking_at_treatment_count | -0.064751 | [-0.157546, 0.021870] |
| neural_fold0_seed29_easy | easy | coverage_with_treatment_ranking | 0.364278 | [0.155514, 0.592798] |
| neural_fold0_seed29_easy | easy | coverage_with_control_ranking | 0.380931 | [0.161975, 0.629201] |
| neural_fold0_seed29_easy | hard | full_total | 0.016965 | [0.002812, 0.031999] |
| neural_fold0_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | hard | total | 0.016965 | [0.002812, 0.031999] |
| neural_fold0_seed29_easy | hard | ranking_at_control_count | 0.015480 | [0.001201, 0.030614] |
| neural_fold0_seed29_easy | hard | ranking_at_treatment_count | 0.009440 | [0.001653, 0.022126] |
| neural_fold0_seed29_easy | hard | coverage_with_treatment_ranking | 0.001486 | [0.000222, 0.003309] |
| neural_fold0_seed29_easy | hard | coverage_with_control_ranking | 0.007526 | [0.000157, 0.021268] |
| damping097_fold0_seed29_all | all | full_total | -0.025445 | [-0.103091, 0.050650] |
| damping097_fold0_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | all | total | -0.025445 | [-0.103091, 0.050650] |
| damping097_fold0_seed29_all | all | ranking_at_control_count | 0.024432 | [-0.029822, 0.083370] |
| damping097_fold0_seed29_all | all | ranking_at_treatment_count | -0.004410 | [-0.054578, 0.041523] |
| damping097_fold0_seed29_all | all | coverage_with_treatment_ranking | -0.049876 | [-0.100764, 0.008253] |
| damping097_fold0_seed29_all | all | coverage_with_control_ranking | -0.021034 | [-0.070925, 0.028340] |
| damping097_fold0_seed29_all | easy | full_total | 0.020302 | [-0.054787, 0.085458] |
| damping097_fold0_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | easy | total | 0.020302 | [-0.054787, 0.085458] |
| damping097_fold0_seed29_all | easy | ranking_at_control_count | 0.114189 | [0.023911, 0.247049] |
| damping097_fold0_seed29_all | easy | ranking_at_treatment_count | 0.006268 | [-0.060527, 0.072810] |
| damping097_fold0_seed29_all | easy | coverage_with_treatment_ranking | -0.093887 | [-0.280044, 0.001784] |
| damping097_fold0_seed29_all | easy | coverage_with_control_ranking | 0.014034 | [-0.024901, 0.072892] |
| damping097_fold0_seed29_all | hard | full_total | -0.073260 | [-0.137725, -0.006324] |
| damping097_fold0_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | hard | total | -0.073260 | [-0.137725, -0.006324] |
| damping097_fold0_seed29_all | hard | ranking_at_control_count | -0.030957 | [-0.074494, 0.017738] |
| damping097_fold0_seed29_all | hard | ranking_at_treatment_count | -0.030042 | [-0.085861, 0.025408] |
| damping097_fold0_seed29_all | hard | coverage_with_treatment_ranking | -0.042304 | [-0.087190, 0.009756] |
| damping097_fold0_seed29_all | hard | coverage_with_control_ranking | -0.043219 | [-0.094367, -0.000168] |
| damping097_fold0_seed29_easy | all | full_total | 0.045734 | [0.006495, 0.088158] |
| damping097_fold0_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | all | total | 0.045734 | [0.006495, 0.088158] |
| damping097_fold0_seed29_easy | all | ranking_at_control_count | 0.016224 | [-0.010022, 0.043191] |
| damping097_fold0_seed29_easy | all | ranking_at_treatment_count | 0.017825 | [-0.009356, 0.045455] |
| damping097_fold0_seed29_easy | all | coverage_with_treatment_ranking | 0.029510 | [0.009757, 0.050592] |
| damping097_fold0_seed29_easy | all | coverage_with_control_ranking | 0.027909 | [0.010380, 0.049541] |
| damping097_fold0_seed29_easy | easy | full_total | 0.011864 | [-0.002642, 0.025302] |
| damping097_fold0_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | easy | total | 0.011864 | [-0.002642, 0.025302] |
| damping097_fold0_seed29_easy | easy | ranking_at_control_count | 0.005263 | [-0.004781, 0.014476] |
| damping097_fold0_seed29_easy | easy | ranking_at_treatment_count | 0.002060 | [-0.011872, 0.015093] |
| damping097_fold0_seed29_easy | easy | coverage_with_treatment_ranking | 0.006601 | [-0.008364, 0.019524] |
| damping097_fold0_seed29_easy | easy | coverage_with_control_ranking | 0.009804 | [0.001411, 0.021484] |
| damping097_fold0_seed29_easy | hard | full_total | 0.053094 | [0.014588, 0.100579] |
| damping097_fold0_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | hard | total | 0.053094 | [0.014588, 0.100579] |
| damping097_fold0_seed29_easy | hard | ranking_at_control_count | 0.036786 | [0.003572, 0.073464] |
| damping097_fold0_seed29_easy | hard | ranking_at_treatment_count | 0.040981 | [0.010197, 0.076505] |
| damping097_fold0_seed29_easy | hard | coverage_with_treatment_ranking | 0.016308 | [-0.000295, 0.033304] |
| damping097_fold0_seed29_easy | hard | coverage_with_control_ranking | 0.012113 | [0.000066, 0.026972] |
| neural_fold0_seed43_all | all | full_total | 0.486135 | [0.305101, 0.662304] |
| neural_fold0_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | all | total | 0.486135 | [0.305101, 0.662304] |
| neural_fold0_seed43_all | all | ranking_at_control_count | 0.228042 | [0.149663, 0.307156] |
| neural_fold0_seed43_all | all | ranking_at_treatment_count | 0.256919 | [0.155773, 0.348122] |
| neural_fold0_seed43_all | all | coverage_with_treatment_ranking | 0.258094 | [0.142384, 0.383840] |
| neural_fold0_seed43_all | all | coverage_with_control_ranking | 0.229216 | [0.132819, 0.332087] |
| neural_fold0_seed43_all | easy | full_total | -0.124768 | [-0.235618, -0.031441] |
| neural_fold0_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | easy | total | -0.124768 | [-0.235618, -0.031441] |
| neural_fold0_seed43_all | easy | ranking_at_control_count | -0.154811 | [-0.313885, -0.004014] |
| neural_fold0_seed43_all | easy | ranking_at_treatment_count | -0.114115 | [-0.183338, -0.049200] |
| neural_fold0_seed43_all | easy | coverage_with_treatment_ranking | 0.030043 | [-0.074792, 0.104208] |
| neural_fold0_seed43_all | easy | coverage_with_control_ranking | -0.010654 | [-0.100927, 0.047429] |
| neural_fold0_seed43_all | hard | full_total | 0.610262 | [0.387622, 0.812842] |
| neural_fold0_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | hard | total | 0.610262 | [0.387622, 0.812842] |
| neural_fold0_seed43_all | hard | ranking_at_control_count | 0.327718 | [0.213396, 0.422308] |
| neural_fold0_seed43_all | hard | ranking_at_treatment_count | 0.370702 | [0.242110, 0.475260] |
| neural_fold0_seed43_all | hard | coverage_with_treatment_ranking | 0.282544 | [0.152993, 0.438392] |
| neural_fold0_seed43_all | hard | coverage_with_control_ranking | 0.239560 | [0.128307, 0.365159] |
| neural_fold0_seed43_easy | all | full_total | 0.050032 | [0.013444, 0.090595] |
| neural_fold0_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | all | total | 0.050032 | [0.013444, 0.090595] |
| neural_fold0_seed43_easy | all | ranking_at_control_count | 0.034059 | [0.009704, 0.063002] |
| neural_fold0_seed43_easy | all | ranking_at_treatment_count | 0.036646 | [0.009891, 0.064955] |
| neural_fold0_seed43_easy | all | coverage_with_treatment_ranking | 0.015973 | [0.002627, 0.028600] |
| neural_fold0_seed43_easy | all | coverage_with_control_ranking | 0.013386 | [0.003245, 0.025426] |
| neural_fold0_seed43_easy | easy | full_total | -0.047240 | [-0.148942, 0.050901] |
| neural_fold0_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | easy | total | -0.047240 | [-0.148942, 0.050901] |
| neural_fold0_seed43_easy | easy | ranking_at_control_count | -0.220417 | [-0.356800, -0.095943] |
| neural_fold0_seed43_easy | easy | ranking_at_treatment_count | -0.247546 | [-0.366049, -0.138300] |
| neural_fold0_seed43_easy | easy | coverage_with_treatment_ranking | 0.173177 | [0.053024, 0.290156] |
| neural_fold0_seed43_easy | easy | coverage_with_control_ranking | 0.200306 | [0.091487, 0.298068] |
| neural_fold0_seed43_easy | hard | full_total | 0.037504 | [0.017928, 0.059240] |
| neural_fold0_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | hard | total | 0.037504 | [0.017928, 0.059240] |
| neural_fold0_seed43_easy | hard | ranking_at_control_count | 0.031027 | [0.013745, 0.051289] |
| neural_fold0_seed43_easy | hard | ranking_at_treatment_count | 0.036151 | [0.018105, 0.055698] |
| neural_fold0_seed43_easy | hard | coverage_with_treatment_ranking | 0.006476 | [0.000761, 0.012874] |
| neural_fold0_seed43_easy | hard | coverage_with_control_ranking | 0.001352 | [-0.000716, 0.003958] |
| damping097_fold0_seed43_all | all | full_total | 0.010160 | [-0.094845, 0.113785] |
| damping097_fold0_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | all | total | 0.010160 | [-0.094845, 0.113785] |
| damping097_fold0_seed43_all | all | ranking_at_control_count | -0.019944 | [-0.078439, 0.042249] |
| damping097_fold0_seed43_all | all | ranking_at_treatment_count | -0.021173 | [-0.073383, 0.027603] |
| damping097_fold0_seed43_all | all | coverage_with_treatment_ranking | 0.030103 | [-0.018972, 0.080646] |
| damping097_fold0_seed43_all | all | coverage_with_control_ranking | 0.031333 | [-0.026528, 0.091615] |
| damping097_fold0_seed43_all | easy | full_total | -0.006862 | [-0.053119, 0.038238] |
| damping097_fold0_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | easy | total | -0.006862 | [-0.053119, 0.038238] |
| damping097_fold0_seed43_all | easy | ranking_at_control_count | 0.000283 | [-0.031783, 0.039689] |
| damping097_fold0_seed43_all | easy | ranking_at_treatment_count | -0.000179 | [-0.028713, 0.037421] |
| damping097_fold0_seed43_all | easy | coverage_with_treatment_ranking | -0.007145 | [-0.044181, 0.020296] |
| damping097_fold0_seed43_all | easy | coverage_with_control_ranking | -0.006684 | [-0.038507, 0.017301] |
| damping097_fold0_seed43_all | hard | full_total | 0.057063 | [-0.044215, 0.157348] |
| damping097_fold0_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | hard | total | 0.057063 | [-0.044215, 0.157348] |
| damping097_fold0_seed43_all | hard | ranking_at_control_count | 0.012266 | [-0.040701, 0.069491] |
| damping097_fold0_seed43_all | hard | ranking_at_treatment_count | 0.016428 | [-0.014531, 0.054075] |
| damping097_fold0_seed43_all | hard | coverage_with_treatment_ranking | 0.044797 | [-0.009260, 0.097345] |
| damping097_fold0_seed43_all | hard | coverage_with_control_ranking | 0.040635 | [-0.034853, 0.106989] |
| damping097_fold0_seed43_easy | all | full_total | 0.189595 | [0.110763, 0.257868] |
| damping097_fold0_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | all | total | 0.189595 | [0.110763, 0.257868] |
| damping097_fold0_seed43_easy | all | ranking_at_control_count | 0.125235 | [0.063880, 0.174588] |
| damping097_fold0_seed43_easy | all | ranking_at_treatment_count | 0.138662 | [0.085700, 0.188746] |
| damping097_fold0_seed43_easy | all | coverage_with_treatment_ranking | 0.064360 | [0.035477, 0.089406] |
| damping097_fold0_seed43_easy | all | coverage_with_control_ranking | 0.050933 | [0.015153, 0.088056] |
| damping097_fold0_seed43_easy | easy | full_total | 0.004747 | [-0.062655, 0.097939] |
| damping097_fold0_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | easy | total | 0.004747 | [-0.062655, 0.097939] |
| damping097_fold0_seed43_easy | easy | ranking_at_control_count | 0.004781 | [-0.050439, 0.079219] |
| damping097_fold0_seed43_easy | easy | ranking_at_treatment_count | 0.034736 | [-0.041339, 0.159673] |
| damping097_fold0_seed43_easy | easy | coverage_with_treatment_ranking | -0.000034 | [-0.021457, 0.021600] |
| damping097_fold0_seed43_easy | easy | coverage_with_control_ranking | -0.029989 | [-0.069062, -0.003203] |
| damping097_fold0_seed43_easy | hard | full_total | 0.236678 | [0.137593, 0.319910] |
| damping097_fold0_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | hard | total | 0.236678 | [0.137593, 0.319910] |
| damping097_fold0_seed43_easy | hard | ranking_at_control_count | 0.187500 | [0.114056, 0.247850] |
| damping097_fold0_seed43_easy | hard | ranking_at_treatment_count | 0.192808 | [0.116936, 0.262575] |
| damping097_fold0_seed43_easy | hard | coverage_with_treatment_ranking | 0.049178 | [0.021716, 0.073028] |
| damping097_fold0_seed43_easy | hard | coverage_with_control_ranking | 0.043870 | [0.011029, 0.081386] |
| neural_fold1_seed17_all | all | full_total | 0.021697 | [-0.027880, 0.075380] |
| neural_fold1_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | all | total | 0.021697 | [-0.027880, 0.075380] |
| neural_fold1_seed17_all | all | ranking_at_control_count | 0.042091 | [-0.027864, 0.126426] |
| neural_fold1_seed17_all | all | ranking_at_treatment_count | 0.062861 | [0.009867, 0.123478] |
| neural_fold1_seed17_all | all | coverage_with_treatment_ranking | -0.020394 | [-0.057925, 0.016152] |
| neural_fold1_seed17_all | all | coverage_with_control_ranking | -0.041163 | [-0.067746, -0.013948] |
| neural_fold1_seed17_all | easy | full_total | 0.015593 | [-0.082038, 0.111092] |
| neural_fold1_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | easy | total | 0.015593 | [-0.082038, 0.111092] |
| neural_fold1_seed17_all | easy | ranking_at_control_count | -0.057042 | [-0.127944, 0.003714] |
| neural_fold1_seed17_all | easy | ranking_at_treatment_count | 0.001561 | [-0.117018, 0.119557] |
| neural_fold1_seed17_all | easy | coverage_with_treatment_ranking | 0.072635 | [0.006575, 0.156892] |
| neural_fold1_seed17_all | easy | coverage_with_control_ranking | 0.014032 | [-0.014709, 0.043561] |
| neural_fold1_seed17_all | hard | full_total | 0.057877 | [-0.012694, 0.139971] |
| neural_fold1_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | hard | total | 0.057877 | [-0.012694, 0.139971] |
| neural_fold1_seed17_all | hard | ranking_at_control_count | 0.075784 | [-0.001822, 0.176572] |
| neural_fold1_seed17_all | hard | ranking_at_treatment_count | 0.067036 | [0.017226, 0.123971] |
| neural_fold1_seed17_all | hard | coverage_with_treatment_ranking | -0.017907 | [-0.054424, 0.020806] |
| neural_fold1_seed17_all | hard | coverage_with_control_ranking | -0.009159 | [-0.039070, 0.024428] |
| neural_fold1_seed17_easy | all | full_total | 0.066733 | [0.046610, 0.085884] |
| neural_fold1_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | all | total | 0.066733 | [0.046610, 0.085884] |
| neural_fold1_seed17_easy | all | ranking_at_control_count | 0.017230 | [0.000900, 0.034989] |
| neural_fold1_seed17_easy | all | ranking_at_treatment_count | 0.015257 | [0.001882, 0.030044] |
| neural_fold1_seed17_easy | all | coverage_with_treatment_ranking | 0.049502 | [0.031156, 0.073198] |
| neural_fold1_seed17_easy | all | coverage_with_control_ranking | 0.051475 | [0.037629, 0.067533] |
| neural_fold1_seed17_easy | easy | full_total | 0.467047 | [0.377719, 0.548693] |
| neural_fold1_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | easy | total | 0.467047 | [0.377719, 0.548693] |
| neural_fold1_seed17_easy | easy | ranking_at_control_count | -0.161852 | [-0.245804, -0.083965] |
| neural_fold1_seed17_easy | easy | ranking_at_treatment_count | -0.175591 | [-0.294538, -0.046483] |
| neural_fold1_seed17_easy | easy | coverage_with_treatment_ranking | 0.628899 | [0.498284, 0.760031] |
| neural_fold1_seed17_easy | easy | coverage_with_control_ranking | 0.642638 | [0.457761, 0.816711] |
| neural_fold1_seed17_easy | hard | full_total | -0.001360 | [-0.033232, 0.025423] |
| neural_fold1_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | hard | total | -0.001360 | [-0.033232, 0.025423] |
| neural_fold1_seed17_easy | hard | ranking_at_control_count | -0.004645 | [-0.034706, 0.020118] |
| neural_fold1_seed17_easy | hard | ranking_at_treatment_count | -0.003911 | [-0.033552, 0.018618] |
| neural_fold1_seed17_easy | hard | coverage_with_treatment_ranking | 0.003286 | [-0.000286, 0.007055] |
| neural_fold1_seed17_easy | hard | coverage_with_control_ranking | 0.002551 | [-0.001518, 0.007310] |
| damping097_fold1_seed17_all | all | full_total | 0.049816 | [0.002273, 0.093402] |
| damping097_fold1_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | all | total | 0.049816 | [0.002273, 0.093402] |
| damping097_fold1_seed17_all | all | ranking_at_control_count | -0.022154 | [-0.047629, 0.003886] |
| damping097_fold1_seed17_all | all | ranking_at_treatment_count | -0.012923 | [-0.066437, 0.027469] |
| damping097_fold1_seed17_all | all | coverage_with_treatment_ranking | 0.071970 | [0.032083, 0.122838] |
| damping097_fold1_seed17_all | all | coverage_with_control_ranking | 0.062739 | [0.011077, 0.142596] |
| damping097_fold1_seed17_all | easy | full_total | -0.028969 | [-0.120335, 0.061539] |
| damping097_fold1_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | easy | total | -0.028969 | [-0.120335, 0.061539] |
| damping097_fold1_seed17_all | easy | ranking_at_control_count | -0.002102 | [-0.023259, 0.016997] |
| damping097_fold1_seed17_all | easy | ranking_at_treatment_count | 0.098917 | [-0.095517, 0.388640] |
| damping097_fold1_seed17_all | easy | coverage_with_treatment_ranking | -0.026867 | [-0.120286, 0.069615] |
| damping097_fold1_seed17_all | easy | coverage_with_control_ranking | -0.127885 | [-0.327467, -0.013168] |
| damping097_fold1_seed17_all | hard | full_total | 0.190113 | [0.068432, 0.365277] |
| damping097_fold1_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | hard | total | 0.190113 | [0.068432, 0.365277] |
| damping097_fold1_seed17_all | hard | ranking_at_control_count | 0.075612 | [-0.061975, 0.283734] |
| damping097_fold1_seed17_all | hard | ranking_at_treatment_count | -0.011970 | [-0.113707, 0.055188] |
| damping097_fold1_seed17_all | hard | coverage_with_treatment_ranking | 0.114501 | [0.036091, 0.212595] |
| damping097_fold1_seed17_all | hard | coverage_with_control_ranking | 0.202083 | [0.041843, 0.413606] |
| damping097_fold1_seed17_easy | all | full_total | -0.123962 | [-0.283362, 0.079781] |
| damping097_fold1_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | all | total | -0.123962 | [-0.283362, 0.079781] |
| damping097_fold1_seed17_easy | all | ranking_at_control_count | 0.107147 | [0.022743, 0.194911] |
| damping097_fold1_seed17_easy | all | ranking_at_treatment_count | 0.039715 | [-0.000885, 0.076607] |
| damping097_fold1_seed17_easy | all | coverage_with_treatment_ranking | -0.231109 | [-0.343601, -0.095408] |
| damping097_fold1_seed17_easy | all | coverage_with_control_ranking | -0.163676 | [-0.301713, 0.031666] |
| damping097_fold1_seed17_easy | easy | full_total | 0.208600 | [-0.048840, 0.573007] |
| damping097_fold1_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | easy | total | 0.208600 | [-0.048840, 0.573007] |
| damping097_fold1_seed17_easy | easy | ranking_at_control_count | -0.054546 | [-0.285089, 0.108456] |
| damping097_fold1_seed17_easy | easy | ranking_at_treatment_count | 0.049333 | [-0.007478, 0.138228] |
| damping097_fold1_seed17_easy | easy | coverage_with_treatment_ranking | 0.263146 | [-0.054564, 0.820132] |
| damping097_fold1_seed17_easy | easy | coverage_with_control_ranking | 0.159266 | [-0.048077, 0.437301] |
| damping097_fold1_seed17_easy | hard | full_total | -0.075594 | [-0.174895, 0.049428] |
| damping097_fold1_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | hard | total | -0.075594 | [-0.174895, 0.049428] |
| damping097_fold1_seed17_easy | hard | ranking_at_control_count | 0.203288 | [-0.006844, 0.507264] |
| damping097_fold1_seed17_easy | hard | ranking_at_treatment_count | 0.017289 | [-0.034469, 0.073352] |
| damping097_fold1_seed17_easy | hard | coverage_with_treatment_ranking | -0.278882 | [-0.465679, -0.144923] |
| damping097_fold1_seed17_easy | hard | coverage_with_control_ranking | -0.092883 | [-0.186107, 0.035391] |
| neural_fold1_seed29_all | all | full_total | -0.095086 | [-0.267524, 0.033210] |
| neural_fold1_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | all | total | -0.095086 | [-0.267524, 0.033210] |
| neural_fold1_seed29_all | all | ranking_at_control_count | -0.100382 | [-0.258807, 0.019622] |
| neural_fold1_seed29_all | all | ranking_at_treatment_count | -0.075119 | [-0.220631, 0.037615] |
| neural_fold1_seed29_all | all | coverage_with_treatment_ranking | 0.005296 | [-0.038398, 0.044127] |
| neural_fold1_seed29_all | all | coverage_with_control_ranking | -0.019967 | [-0.084454, 0.030272] |
| neural_fold1_seed29_all | easy | full_total | -0.052115 | [-0.212615, 0.077433] |
| neural_fold1_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | easy | total | -0.052115 | [-0.212615, 0.077433] |
| neural_fold1_seed29_all | easy | ranking_at_control_count | -0.085478 | [-0.230138, 0.038236] |
| neural_fold1_seed29_all | easy | ranking_at_treatment_count | -0.059306 | [-0.217407, 0.071640] |
| neural_fold1_seed29_all | easy | coverage_with_treatment_ranking | 0.033363 | [-0.007067, 0.099516] |
| neural_fold1_seed29_all | easy | coverage_with_control_ranking | 0.007191 | [-0.005516, 0.027617] |
| neural_fold1_seed29_all | hard | full_total | -0.391063 | [-1.168190, 0.085102] |
| neural_fold1_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | hard | total | -0.391063 | [-1.168190, 0.085102] |
| neural_fold1_seed29_all | hard | ranking_at_control_count | -0.405381 | [-1.178065, 0.072933] |
| neural_fold1_seed29_all | hard | ranking_at_treatment_count | -0.268024 | [-0.802607, 0.091078] |
| neural_fold1_seed29_all | hard | coverage_with_treatment_ranking | 0.014318 | [-0.010533, 0.039819] |
| neural_fold1_seed29_all | hard | coverage_with_control_ranking | -0.123040 | [-0.377992, 0.018667] |
| neural_fold1_seed29_easy | all | full_total | 0.065669 | [0.041065, 0.092814] |
| neural_fold1_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | all | total | 0.065669 | [0.041065, 0.092814] |
| neural_fold1_seed29_easy | all | ranking_at_control_count | 0.024424 | [0.008155, 0.040967] |
| neural_fold1_seed29_easy | all | ranking_at_treatment_count | 0.022856 | [0.007882, 0.040143] |
| neural_fold1_seed29_easy | all | coverage_with_treatment_ranking | 0.041245 | [0.019067, 0.062345] |
| neural_fold1_seed29_easy | all | coverage_with_control_ranking | 0.042813 | [0.027882, 0.058001] |
| neural_fold1_seed29_easy | easy | full_total | 0.378652 | [0.137513, 0.599325] |
| neural_fold1_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | easy | total | 0.378652 | [0.137513, 0.599325] |
| neural_fold1_seed29_easy | easy | ranking_at_control_count | -0.067501 | [-0.225046, 0.126284] |
| neural_fold1_seed29_easy | easy | ranking_at_treatment_count | -0.083607 | [-0.248258, 0.105794] |
| neural_fold1_seed29_easy | easy | coverage_with_treatment_ranking | 0.446153 | [0.253743, 0.638038] |
| neural_fold1_seed29_easy | easy | coverage_with_control_ranking | 0.462259 | [0.275838, 0.638569] |
| neural_fold1_seed29_easy | hard | full_total | 0.030717 | [0.008659, 0.055270] |
| neural_fold1_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | hard | total | 0.030717 | [0.008659, 0.055270] |
| neural_fold1_seed29_easy | hard | ranking_at_control_count | 0.029101 | [0.005746, 0.054428] |
| neural_fold1_seed29_easy | hard | ranking_at_treatment_count | 0.026355 | [0.006674, 0.046957] |
| neural_fold1_seed29_easy | hard | coverage_with_treatment_ranking | 0.001616 | [-0.013333, 0.012749] |
| neural_fold1_seed29_easy | hard | coverage_with_control_ranking | 0.004363 | [0.001019, 0.008736] |
| damping097_fold1_seed29_all | all | full_total | 0.158090 | [0.042074, 0.322907] |
| damping097_fold1_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | all | total | 0.158090 | [0.042074, 0.322907] |
| damping097_fold1_seed29_all | all | ranking_at_control_count | 0.105887 | [0.021889, 0.208496] |
| damping097_fold1_seed29_all | all | ranking_at_treatment_count | 0.095929 | [-0.024375, 0.270648] |
| damping097_fold1_seed29_all | all | coverage_with_treatment_ranking | 0.052203 | [0.005207, 0.120729] |
| damping097_fold1_seed29_all | all | coverage_with_control_ranking | 0.062161 | [0.014885, 0.113384] |
| damping097_fold1_seed29_all | easy | full_total | -0.287263 | [-0.806028, -0.010817] |
| damping097_fold1_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | easy | total | -0.287263 | [-0.806028, -0.010817] |
| damping097_fold1_seed29_all | easy | ranking_at_control_count | -0.111542 | [-0.296764, -0.005834] |
| damping097_fold1_seed29_all | easy | ranking_at_treatment_count | -0.098149 | [-0.278208, 0.004363] |
| damping097_fold1_seed29_all | easy | coverage_with_treatment_ranking | -0.175722 | [-0.509436, 0.000000] |
| damping097_fold1_seed29_all | easy | coverage_with_control_ranking | -0.189115 | [-0.532297, -0.008464] |
| damping097_fold1_seed29_all | hard | full_total | 0.312955 | [0.124972, 0.568761] |
| damping097_fold1_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | hard | total | 0.312955 | [0.124972, 0.568761] |
| damping097_fold1_seed29_all | hard | ranking_at_control_count | 0.213490 | [0.066647, 0.393957] |
| damping097_fold1_seed29_all | hard | ranking_at_treatment_count | 0.188881 | [0.024397, 0.461567] |
| damping097_fold1_seed29_all | hard | coverage_with_treatment_ranking | 0.099464 | [0.033908, 0.206777] |
| damping097_fold1_seed29_all | hard | coverage_with_control_ranking | 0.124074 | [0.021255, 0.271161] |
| damping097_fold1_seed29_easy | all | full_total | -0.273535 | [-0.418331, -0.114116] |
| damping097_fold1_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | all | total | -0.273535 | [-0.418331, -0.114116] |
| damping097_fold1_seed29_easy | all | ranking_at_control_count | -0.004744 | [-0.082519, 0.080036] |
| damping097_fold1_seed29_easy | all | ranking_at_treatment_count | -0.028369 | [-0.073362, 0.013961] |
| damping097_fold1_seed29_easy | all | coverage_with_treatment_ranking | -0.268791 | [-0.357984, -0.177320] |
| damping097_fold1_seed29_easy | all | coverage_with_control_ranking | -0.245165 | [-0.357984, -0.101376] |
| damping097_fold1_seed29_easy | easy | full_total | 0.135381 | [0.013698, 0.257455] |
| damping097_fold1_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | easy | total | 0.135381 | [0.013698, 0.257455] |
| damping097_fold1_seed29_easy | easy | ranking_at_control_count | 0.038680 | [-0.085210, 0.145231] |
| damping097_fold1_seed29_easy | easy | ranking_at_treatment_count | 0.076877 | [0.024079, 0.140411] |
| damping097_fold1_seed29_easy | easy | coverage_with_treatment_ranking | 0.096702 | [-0.047302, 0.293521] |
| damping097_fold1_seed29_easy | easy | coverage_with_control_ranking | 0.058504 | [-0.050035, 0.182764] |
| damping097_fold1_seed29_easy | hard | full_total | -0.280915 | [-0.404061, -0.151867] |
| damping097_fold1_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | hard | total | -0.280915 | [-0.404061, -0.151867] |
| damping097_fold1_seed29_easy | hard | ranking_at_control_count | -0.119021 | [-0.182627, -0.065188] |
| damping097_fold1_seed29_easy | hard | ranking_at_treatment_count | -0.080292 | [-0.136545, -0.025283] |
| damping097_fold1_seed29_easy | hard | coverage_with_treatment_ranking | -0.161894 | [-0.270267, -0.051379] |
| damping097_fold1_seed29_easy | hard | coverage_with_control_ranking | -0.200622 | [-0.273463, -0.119855] |
| neural_fold1_seed43_all | all | full_total | -0.093263 | [-0.190897, 0.006341] |
| neural_fold1_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | all | total | -0.093263 | [-0.190897, 0.006341] |
| neural_fold1_seed43_all | all | ranking_at_control_count | 0.048861 | [-0.034286, 0.139912] |
| neural_fold1_seed43_all | all | ranking_at_treatment_count | 0.013408 | [-0.050719, 0.085666] |
| neural_fold1_seed43_all | all | coverage_with_treatment_ranking | -0.142124 | [-0.212063, -0.070774] |
| neural_fold1_seed43_all | all | coverage_with_control_ranking | -0.106670 | [-0.173628, -0.044282] |
| neural_fold1_seed43_all | easy | full_total | -0.032467 | [-0.192588, 0.086580] |
| neural_fold1_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | easy | total | -0.032467 | [-0.192588, 0.086580] |
| neural_fold1_seed43_all | easy | ranking_at_control_count | -0.026051 | [-0.117150, 0.041520] |
| neural_fold1_seed43_all | easy | ranking_at_treatment_count | -0.054227 | [-0.147490, 0.017903] |
| neural_fold1_seed43_all | easy | coverage_with_treatment_ranking | -0.006416 | [-0.087753, 0.069603] |
| neural_fold1_seed43_all | easy | coverage_with_control_ranking | 0.021760 | [-0.076757, 0.142639] |
| neural_fold1_seed43_all | hard | full_total | -0.078312 | [-0.161421, 0.021687] |
| neural_fold1_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | hard | total | -0.078312 | [-0.161421, 0.021687] |
| neural_fold1_seed43_all | hard | ranking_at_control_count | 0.051086 | [-0.024611, 0.145214] |
| neural_fold1_seed43_all | hard | ranking_at_treatment_count | 0.020116 | [-0.058705, 0.111852] |
| neural_fold1_seed43_all | hard | coverage_with_treatment_ranking | -0.129398 | [-0.193299, -0.064404] |
| neural_fold1_seed43_all | hard | coverage_with_control_ranking | -0.098428 | [-0.164518, -0.036450] |
| neural_fold1_seed43_easy | all | full_total | -0.048786 | [-0.080556, -0.016690] |
| neural_fold1_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | all | total | -0.048786 | [-0.080556, -0.016690] |
| neural_fold1_seed43_easy | all | ranking_at_control_count | -0.034968 | [-0.056182, -0.015363] |
| neural_fold1_seed43_easy | all | ranking_at_treatment_count | -0.037755 | [-0.059234, -0.017746] |
| neural_fold1_seed43_easy | all | coverage_with_treatment_ranking | -0.013818 | [-0.033682, 0.006112] |
| neural_fold1_seed43_easy | all | coverage_with_control_ranking | -0.011031 | [-0.029345, 0.008385] |
| neural_fold1_seed43_easy | easy | full_total | 0.174027 | [-0.110879, 0.445313] |
| neural_fold1_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | easy | total | 0.174027 | [-0.110879, 0.445313] |
| neural_fold1_seed43_easy | easy | ranking_at_control_count | 0.191907 | [0.034007, 0.402926] |
| neural_fold1_seed43_easy | easy | ranking_at_treatment_count | 0.133196 | [-0.055856, 0.343107] |
| neural_fold1_seed43_easy | easy | coverage_with_treatment_ranking | -0.017880 | [-0.293906, 0.272298] |
| neural_fold1_seed43_easy | easy | coverage_with_control_ranking | 0.040831 | [-0.195813, 0.311143] |
| neural_fold1_seed43_easy | hard | full_total | -0.024216 | [-0.055890, -0.002382] |
| neural_fold1_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | hard | total | -0.024216 | [-0.055890, -0.002382] |
| neural_fold1_seed43_easy | hard | ranking_at_control_count | -0.020436 | [-0.044154, -0.003044] |
| neural_fold1_seed43_easy | hard | ranking_at_treatment_count | -0.020886 | [-0.048187, -0.000093] |
| neural_fold1_seed43_easy | hard | coverage_with_treatment_ranking | -0.003779 | [-0.012942, 0.001859] |
| neural_fold1_seed43_easy | hard | coverage_with_control_ranking | -0.003329 | [-0.010178, 0.002818] |
| damping097_fold1_seed43_all | all | full_total | -0.046322 | [-0.105026, 0.009182] |
| damping097_fold1_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | all | total | -0.046322 | [-0.105026, 0.009182] |
| damping097_fold1_seed43_all | all | ranking_at_control_count | -0.063923 | [-0.149520, 0.009130] |
| damping097_fold1_seed43_all | all | ranking_at_treatment_count | -0.062732 | [-0.127074, 0.000528] |
| damping097_fold1_seed43_all | all | coverage_with_treatment_ranking | 0.017601 | [-0.014435, 0.067740] |
| damping097_fold1_seed43_all | all | coverage_with_control_ranking | 0.016410 | [-0.001705, 0.039369] |
| damping097_fold1_seed43_all | easy | full_total | 0.007022 | [-0.013139, 0.030973] |
| damping097_fold1_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | easy | total | 0.007022 | [-0.013139, 0.030973] |
| damping097_fold1_seed43_all | easy | ranking_at_control_count | 0.007515 | [-0.008888, 0.028668] |
| damping097_fold1_seed43_all | easy | ranking_at_treatment_count | 0.004698 | [-0.015718, 0.028649] |
| damping097_fold1_seed43_all | easy | coverage_with_treatment_ranking | -0.000493 | [-0.003785, 0.002305] |
| damping097_fold1_seed43_all | easy | coverage_with_control_ranking | 0.002324 | [-0.000106, 0.007078] |
| damping097_fold1_seed43_all | hard | full_total | -0.079757 | [-0.206442, 0.021914] |
| damping097_fold1_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | hard | total | -0.079757 | [-0.206442, 0.021914] |
| damping097_fold1_seed43_all | hard | ranking_at_control_count | -0.114034 | [-0.249717, 0.004529] |
| damping097_fold1_seed43_all | hard | ranking_at_treatment_count | -0.081117 | [-0.210813, 0.023108] |
| damping097_fold1_seed43_all | hard | coverage_with_treatment_ranking | 0.034277 | [-0.021566, 0.123828] |
| damping097_fold1_seed43_all | hard | coverage_with_control_ranking | 0.001360 | [-0.005117, 0.008191] |
| damping097_fold1_seed43_easy | all | full_total | -0.104697 | [-0.229352, 0.009282] |
| damping097_fold1_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | all | total | -0.104697 | [-0.229352, 0.009282] |
| damping097_fold1_seed43_easy | all | ranking_at_control_count | 0.121474 | [0.027638, 0.239496] |
| damping097_fold1_seed43_easy | all | ranking_at_treatment_count | 0.107574 | [0.030765, 0.204479] |
| damping097_fold1_seed43_easy | all | coverage_with_treatment_ranking | -0.226171 | [-0.292430, -0.167686] |
| damping097_fold1_seed43_easy | all | coverage_with_control_ranking | -0.212271 | [-0.286719, -0.148580] |
| damping097_fold1_seed43_easy | easy | full_total | -0.057509 | [-0.142033, 0.034822] |
| damping097_fold1_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | easy | total | -0.057509 | [-0.142033, 0.034822] |
| damping097_fold1_seed43_easy | easy | ranking_at_control_count | -0.114400 | [-0.314356, 0.040565] |
| damping097_fold1_seed43_easy | easy | ranking_at_treatment_count | -0.066278 | [-0.120217, -0.028602] |
| damping097_fold1_seed43_easy | easy | coverage_with_treatment_ranking | 0.056891 | [-0.082868, 0.252302] |
| damping097_fold1_seed43_easy | easy | coverage_with_control_ranking | 0.008769 | [-0.078907, 0.095682] |
| damping097_fold1_seed43_easy | hard | full_total | 0.046114 | [-0.073253, 0.167815] |
| damping097_fold1_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | hard | total | 0.046114 | [-0.073253, 0.167815] |
| damping097_fold1_seed43_easy | hard | ranking_at_control_count | 0.168905 | [0.029959, 0.337477] |
| damping097_fold1_seed43_easy | hard | ranking_at_treatment_count | 0.130817 | [0.019010, 0.261961] |
| damping097_fold1_seed43_easy | hard | coverage_with_treatment_ranking | -0.122791 | [-0.185412, -0.064403] |
| damping097_fold1_seed43_easy | hard | coverage_with_control_ranking | -0.084703 | [-0.124482, -0.045772] |
| neural_fold2_seed17_all | all | full_total | 0.040884 | [-0.022746, 0.116647] |
| neural_fold2_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | all | total | 0.040884 | [-0.022746, 0.116647] |
| neural_fold2_seed17_all | all | ranking_at_control_count | -0.000747 | [-0.067114, 0.076750] |
| neural_fold2_seed17_all | all | ranking_at_treatment_count | 0.016644 | [-0.052647, 0.097963] |
| neural_fold2_seed17_all | all | coverage_with_treatment_ranking | 0.041631 | [0.022714, 0.060263] |
| neural_fold2_seed17_all | all | coverage_with_control_ranking | 0.024240 | [0.011198, 0.039501] |
| neural_fold2_seed17_all | easy | full_total | 0.301242 | [0.162947, 0.468080] |
| neural_fold2_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | easy | total | 0.301242 | [0.162947, 0.468080] |
| neural_fold2_seed17_all | easy | ranking_at_control_count | 0.098962 | [-0.000412, 0.199104] |
| neural_fold2_seed17_all | easy | ranking_at_treatment_count | 0.111038 | [-0.019122, 0.218701] |
| neural_fold2_seed17_all | easy | coverage_with_treatment_ranking | 0.202281 | [0.092797, 0.326543] |
| neural_fold2_seed17_all | easy | coverage_with_control_ranking | 0.190204 | [0.068051, 0.324780] |
| neural_fold2_seed17_all | hard | full_total | 0.049547 | [-0.041166, 0.153403] |
| neural_fold2_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | hard | total | 0.049547 | [-0.041166, 0.153403] |
| neural_fold2_seed17_all | hard | ranking_at_control_count | 0.009157 | [-0.087355, 0.123222] |
| neural_fold2_seed17_all | hard | ranking_at_treatment_count | 0.029187 | [-0.062111, 0.134766] |
| neural_fold2_seed17_all | hard | coverage_with_treatment_ranking | 0.040390 | [0.017227, 0.062561] |
| neural_fold2_seed17_all | hard | coverage_with_control_ranking | 0.020360 | [-0.001539, 0.043614] |
| neural_fold2_seed17_easy | all | full_total | 0.000179 | [-0.005423, 0.005050] |
| neural_fold2_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | all | total | 0.000179 | [-0.005423, 0.005050] |
| neural_fold2_seed17_easy | all | ranking_at_control_count | -0.000767 | [-0.004522, 0.002576] |
| neural_fold2_seed17_easy | all | ranking_at_treatment_count | 0.000040 | [-0.005619, 0.004939] |
| neural_fold2_seed17_easy | all | coverage_with_treatment_ranking | 0.000946 | [-0.005254, 0.007256] |
| neural_fold2_seed17_easy | all | coverage_with_control_ranking | 0.000139 | [-0.000015, 0.000305] |
| neural_fold2_seed17_easy | easy | full_total | 0.000938 | [-0.002851, 0.004620] |
| neural_fold2_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | easy | total | 0.000938 | [-0.002851, 0.004620] |
| neural_fold2_seed17_easy | easy | ranking_at_control_count | -0.000125 | [-0.025374, 0.024062] |
| neural_fold2_seed17_easy | easy | ranking_at_treatment_count | 0.000336 | [-0.004524, 0.004475] |
| neural_fold2_seed17_easy | easy | coverage_with_treatment_ranking | 0.001063 | [-0.023768, 0.027968] |
| neural_fold2_seed17_easy | easy | coverage_with_control_ranking | 0.000602 | [-0.005712, 0.006326] |
| neural_fold2_seed17_easy | hard | full_total | 0.001925 | [-0.001598, 0.006237] |
| neural_fold2_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | hard | total | 0.001925 | [-0.001598, 0.006237] |
| neural_fold2_seed17_easy | hard | ranking_at_control_count | 0.001195 | [-0.001598, 0.004048] |
| neural_fold2_seed17_easy | hard | ranking_at_treatment_count | 0.001865 | [-0.001598, 0.006057] |
| neural_fold2_seed17_easy | hard | coverage_with_treatment_ranking | 0.000730 | [0.000000, 0.002189] |
| neural_fold2_seed17_easy | hard | coverage_with_control_ranking | 0.000060 | [0.000000, 0.000180] |
| damping097_fold2_seed17_all | all | full_total | -0.006305 | [-0.018552, 0.006576] |
| damping097_fold2_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | all | total | -0.006305 | [-0.018552, 0.006576] |
| damping097_fold2_seed17_all | all | ranking_at_control_count | -0.003031 | [-0.045491, 0.027300] |
| damping097_fold2_seed17_all | all | ranking_at_treatment_count | -0.018880 | [-0.096635, 0.032677] |
| damping097_fold2_seed17_all | all | coverage_with_treatment_ranking | -0.003274 | [-0.036731, 0.038064] |
| damping097_fold2_seed17_all | all | coverage_with_control_ranking | 0.012575 | [-0.042366, 0.088678] |
| damping097_fold2_seed17_all | easy | full_total | -0.001523 | [-0.034095, 0.025828] |
| damping097_fold2_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | easy | total | -0.001523 | [-0.034095, 0.025828] |
| damping097_fold2_seed17_all | easy | ranking_at_control_count | -0.003122 | [-0.021928, 0.010115] |
| damping097_fold2_seed17_all | easy | ranking_at_treatment_count | -0.021194 | [-0.075294, 0.009603] |
| damping097_fold2_seed17_all | easy | coverage_with_treatment_ranking | 0.001599 | [-0.012179, 0.016975] |
| damping097_fold2_seed17_all | easy | coverage_with_control_ranking | 0.019671 | [-0.000024, 0.047149] |
| damping097_fold2_seed17_all | hard | full_total | -0.021370 | [-0.041291, -0.002003] |
| damping097_fold2_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | hard | total | -0.021370 | [-0.041291, -0.002003] |
| damping097_fold2_seed17_all | hard | ranking_at_control_count | -0.019928 | [-0.082701, 0.022431] |
| damping097_fold2_seed17_all | hard | ranking_at_treatment_count | -0.043366 | [-0.161750, 0.031774] |
| damping097_fold2_seed17_all | hard | coverage_with_treatment_ranking | -0.001442 | [-0.042562, 0.049201] |
| damping097_fold2_seed17_all | hard | coverage_with_control_ranking | 0.021996 | [-0.052744, 0.125407] |
| damping097_fold2_seed17_easy | all | full_total | 0.091022 | [0.036435, 0.154632] |
| damping097_fold2_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | all | total | 0.091022 | [0.036435, 0.154632] |
| damping097_fold2_seed17_easy | all | ranking_at_control_count | 0.060342 | [0.016026, 0.103202] |
| damping097_fold2_seed17_easy | all | ranking_at_treatment_count | 0.057982 | [0.020172, 0.093457] |
| damping097_fold2_seed17_easy | all | coverage_with_treatment_ranking | 0.030680 | [0.011909, 0.056203] |
| damping097_fold2_seed17_easy | all | coverage_with_control_ranking | 0.033040 | [0.010491, 0.065918] |
| damping097_fold2_seed17_easy | easy | full_total | -0.174866 | [-0.259101, -0.097751] |
| damping097_fold2_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | easy | total | -0.174866 | [-0.259101, -0.097751] |
| damping097_fold2_seed17_easy | easy | ranking_at_control_count | -0.239484 | [-0.318058, -0.166394] |
| damping097_fold2_seed17_easy | easy | ranking_at_treatment_count | -0.224754 | [-0.296589, -0.152092] |
| damping097_fold2_seed17_easy | easy | coverage_with_treatment_ranking | 0.064618 | [0.035546, 0.092151] |
| damping097_fold2_seed17_easy | easy | coverage_with_control_ranking | 0.049887 | [0.021718, 0.078903] |
| damping097_fold2_seed17_easy | hard | full_total | 0.055384 | [0.010562, 0.103506] |
| damping097_fold2_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | hard | total | 0.055384 | [0.010562, 0.103506] |
| damping097_fold2_seed17_easy | hard | ranking_at_control_count | 0.044239 | [0.007782, 0.078584] |
| damping097_fold2_seed17_easy | hard | ranking_at_treatment_count | 0.036969 | [0.005346, 0.064702] |
| damping097_fold2_seed17_easy | hard | coverage_with_treatment_ranking | 0.011145 | [0.000000, 0.027912] |
| damping097_fold2_seed17_easy | hard | coverage_with_control_ranking | 0.018415 | [0.002099, 0.046537] |
| neural_fold2_seed29_all | all | full_total | -0.044987 | [-0.098448, 0.002865] |
| neural_fold2_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | all | total | -0.044987 | [-0.098448, 0.002865] |
| neural_fold2_seed29_all | all | ranking_at_control_count | -0.054104 | [-0.112969, -0.001221] |
| neural_fold2_seed29_all | all | ranking_at_treatment_count | -0.079285 | [-0.168522, -0.004461] |
| neural_fold2_seed29_all | all | coverage_with_treatment_ranking | 0.009117 | [0.002563, 0.016806] |
| neural_fold2_seed29_all | all | coverage_with_control_ranking | 0.034297 | [0.003467, 0.074231] |
| neural_fold2_seed29_all | easy | full_total | -0.040827 | [-0.265106, 0.148948] |
| neural_fold2_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | easy | total | -0.040827 | [-0.265106, 0.148948] |
| neural_fold2_seed29_all | easy | ranking_at_control_count | 0.007389 | [-0.141805, 0.114993] |
| neural_fold2_seed29_all | easy | ranking_at_treatment_count | -0.052915 | [-0.243474, 0.111022] |
| neural_fold2_seed29_all | easy | coverage_with_treatment_ranking | -0.048215 | [-0.146261, 0.047499] |
| neural_fold2_seed29_all | easy | coverage_with_control_ranking | 0.012088 | [-0.029255, 0.052410] |
| neural_fold2_seed29_all | hard | full_total | -0.083673 | [-0.185222, -0.002952] |
| neural_fold2_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | hard | total | -0.083673 | [-0.185222, -0.002952] |
| neural_fold2_seed29_all | hard | ranking_at_control_count | -0.087858 | [-0.187708, -0.007607] |
| neural_fold2_seed29_all | hard | ranking_at_treatment_count | -0.118062 | [-0.251861, -0.007505] |
| neural_fold2_seed29_all | hard | coverage_with_treatment_ranking | 0.004184 | [-0.000231, 0.011250] |
| neural_fold2_seed29_all | hard | coverage_with_control_ranking | 0.034388 | [0.001636, 0.076766] |
| neural_fold2_seed29_easy | all | full_total | 0.004763 | [0.001909, 0.007892] |
| neural_fold2_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | all | total | 0.004763 | [0.001909, 0.007892] |
| neural_fold2_seed29_easy | all | ranking_at_control_count | -0.001561 | [-0.013196, 0.006006] |
| neural_fold2_seed29_easy | all | ranking_at_treatment_count | 0.006912 | [0.001949, 0.013861] |
| neural_fold2_seed29_easy | all | coverage_with_treatment_ranking | 0.006325 | [-0.000536, 0.019561] |
| neural_fold2_seed29_easy | all | coverage_with_control_ranking | -0.002149 | [-0.006380, 0.000010] |
| neural_fold2_seed29_easy | easy | full_total | -0.014232 | [-0.043572, 0.008469] |
| neural_fold2_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | easy | total | -0.014232 | [-0.043572, 0.008469] |
| neural_fold2_seed29_easy | easy | ranking_at_control_count | 0.012841 | [-0.038211, 0.074430] |
| neural_fold2_seed29_easy | easy | ranking_at_treatment_count | -0.015511 | [-0.044358, 0.007051] |
| neural_fold2_seed29_easy | easy | coverage_with_treatment_ranking | -0.027073 | [-0.078546, 0.000121] |
| neural_fold2_seed29_easy | easy | coverage_with_control_ranking | 0.001279 | [-0.000759, 0.004086] |
| neural_fold2_seed29_easy | hard | full_total | 0.007257 | [-0.000548, 0.020862] |
| neural_fold2_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | hard | total | 0.007257 | [-0.000548, 0.020862] |
| neural_fold2_seed29_easy | hard | ranking_at_control_count | -0.004901 | [-0.036918, 0.019996] |
| neural_fold2_seed29_easy | hard | ranking_at_treatment_count | 0.011618 | [0.000866, 0.025591] |
| neural_fold2_seed29_easy | hard | coverage_with_treatment_ranking | 0.012158 | [-0.000730, 0.037204] |
| neural_fold2_seed29_easy | hard | coverage_with_control_ranking | -0.004361 | [-0.013082, 0.000000] |
| damping097_fold2_seed29_all | all | full_total | -0.102953 | [-0.179578, -0.008051] |
| damping097_fold2_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | all | total | -0.102953 | [-0.179578, -0.008051] |
| damping097_fold2_seed29_all | all | ranking_at_control_count | -0.062969 | [-0.085373, -0.042577] |
| damping097_fold2_seed29_all | all | ranking_at_treatment_count | -0.057825 | [-0.086523, -0.034524] |
| damping097_fold2_seed29_all | all | coverage_with_treatment_ranking | -0.039984 | [-0.119149, 0.069419] |
| damping097_fold2_seed29_all | all | coverage_with_control_ranking | -0.045128 | [-0.127110, 0.068321] |
| damping097_fold2_seed29_all | easy | full_total | 0.034933 | [0.016340, 0.056593] |
| damping097_fold2_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | easy | total | 0.034933 | [0.016340, 0.056593] |
| damping097_fold2_seed29_all | easy | ranking_at_control_count | 0.011245 | [-0.007289, 0.033678] |
| damping097_fold2_seed29_all | easy | ranking_at_treatment_count | 0.005484 | [-0.030858, 0.031782] |
| damping097_fold2_seed29_all | easy | coverage_with_treatment_ranking | 0.023688 | [-0.000308, 0.052219] |
| damping097_fold2_seed29_all | easy | coverage_with_control_ranking | 0.029449 | [0.002621, 0.062678] |
| damping097_fold2_seed29_all | hard | full_total | -0.118526 | [-0.216234, 0.006148] |
| damping097_fold2_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | hard | total | -0.118526 | [-0.216234, 0.006148] |
| damping097_fold2_seed29_all | hard | ranking_at_control_count | -0.080365 | [-0.101900, -0.059335] |
| damping097_fold2_seed29_all | hard | ranking_at_treatment_count | -0.081191 | [-0.122774, -0.046516] |
| damping097_fold2_seed29_all | hard | coverage_with_treatment_ranking | -0.038161 | [-0.133626, 0.093163] |
| damping097_fold2_seed29_all | hard | coverage_with_control_ranking | -0.037335 | [-0.138282, 0.112475] |
| damping097_fold2_seed29_easy | all | full_total | 0.065745 | [-0.032648, 0.224854] |
| damping097_fold2_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | all | total | 0.065745 | [-0.032648, 0.224854] |
| damping097_fold2_seed29_easy | all | ranking_at_control_count | 0.103281 | [0.032264, 0.216029] |
| damping097_fold2_seed29_easy | all | ranking_at_treatment_count | 0.092543 | [0.012948, 0.211876] |
| damping097_fold2_seed29_easy | all | coverage_with_treatment_ranking | -0.037536 | [-0.085885, 0.016407] |
| damping097_fold2_seed29_easy | all | coverage_with_control_ranking | -0.026798 | [-0.078233, 0.029352] |
| damping097_fold2_seed29_easy | easy | full_total | -0.320057 | [-0.411752, -0.212999] |
| damping097_fold2_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | easy | total | -0.320057 | [-0.411752, -0.212999] |
| damping097_fold2_seed29_easy | easy | ranking_at_control_count | -0.144291 | [-0.215595, -0.085572] |
| damping097_fold2_seed29_easy | easy | ranking_at_treatment_count | -0.164635 | [-0.225009, -0.091518] |
| damping097_fold2_seed29_easy | easy | coverage_with_treatment_ranking | -0.175766 | [-0.278491, -0.059281] |
| damping097_fold2_seed29_easy | easy | coverage_with_control_ranking | -0.155422 | [-0.240327, -0.062949] |
| damping097_fold2_seed29_easy | hard | full_total | 0.097096 | [0.023965, 0.223075] |
| damping097_fold2_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | hard | total | 0.097096 | [0.023965, 0.223075] |
| damping097_fold2_seed29_easy | hard | ranking_at_control_count | 0.100900 | [0.029115, 0.211144] |
| damping097_fold2_seed29_easy | hard | ranking_at_treatment_count | 0.093724 | [0.025609, 0.203204] |
| damping097_fold2_seed29_easy | hard | coverage_with_treatment_ranking | -0.003804 | [-0.024126, 0.019299] |
| damping097_fold2_seed29_easy | hard | coverage_with_control_ranking | 0.003372 | [-0.015257, 0.024157] |
| neural_fold2_seed43_all | all | full_total | -0.024365 | [-0.092460, 0.018535] |
| neural_fold2_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | all | total | -0.024365 | [-0.092460, 0.018535] |
| neural_fold2_seed43_all | all | ranking_at_control_count | -0.045235 | [-0.107962, -0.002937] |
| neural_fold2_seed43_all | all | ranking_at_treatment_count | -0.040694 | [-0.101959, -0.003730] |
| neural_fold2_seed43_all | all | coverage_with_treatment_ranking | 0.020870 | [0.000867, 0.052779] |
| neural_fold2_seed43_all | all | coverage_with_control_ranking | 0.016330 | [0.004149, 0.031790] |
| neural_fold2_seed43_all | easy | full_total | 0.080814 | [-0.039241, 0.206058] |
| neural_fold2_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | easy | total | 0.080814 | [-0.039241, 0.206058] |
| neural_fold2_seed43_all | easy | ranking_at_control_count | -0.040067 | [-0.154893, 0.079549] |
| neural_fold2_seed43_all | easy | ranking_at_treatment_count | 0.028150 | [-0.088513, 0.139505] |
| neural_fold2_seed43_all | easy | coverage_with_treatment_ranking | 0.120882 | [0.056802, 0.207116] |
| neural_fold2_seed43_all | easy | coverage_with_control_ranking | 0.052664 | [-0.001433, 0.110921] |
| neural_fold2_seed43_all | hard | full_total | -0.024668 | [-0.107803, 0.032545] |
| neural_fold2_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | hard | total | -0.024668 | [-0.107803, 0.032545] |
| neural_fold2_seed43_all | hard | ranking_at_control_count | -0.052231 | [-0.124250, -0.001390] |
| neural_fold2_seed43_all | hard | ranking_at_treatment_count | -0.044132 | [-0.116278, 0.000976] |
| neural_fold2_seed43_all | hard | coverage_with_treatment_ranking | 0.027563 | [0.000000, 0.078190] |
| neural_fold2_seed43_all | hard | coverage_with_control_ranking | 0.019464 | [0.002735, 0.043458] |
| neural_fold2_seed43_easy | all | full_total | 0.000670 | [-0.001013, 0.002582] |
| neural_fold2_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | all | total | 0.000670 | [-0.001013, 0.002582] |
| neural_fold2_seed43_easy | all | ranking_at_control_count | 0.001036 | [-0.000124, 0.002511] |
| neural_fold2_seed43_easy | all | ranking_at_treatment_count | 0.001136 | [-0.000864, 0.003221] |
| neural_fold2_seed43_easy | all | coverage_with_treatment_ranking | -0.000367 | [-0.001248, 0.000226] |
| neural_fold2_seed43_easy | all | coverage_with_control_ranking | -0.000467 | [-0.001273, 0.000073] |
| neural_fold2_seed43_easy | easy | full_total | -0.002619 | [-0.010949, 0.004178] |
| neural_fold2_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | easy | total | -0.002619 | [-0.010949, 0.004178] |
| neural_fold2_seed43_easy | easy | ranking_at_control_count | -0.005959 | [-0.014594, -0.000336] |
| neural_fold2_seed43_easy | easy | ranking_at_treatment_count | -0.003390 | [-0.008334, 0.000065] |
| neural_fold2_seed43_easy | easy | coverage_with_treatment_ranking | 0.003340 | [-0.006765, 0.017797] |
| neural_fold2_seed43_easy | easy | coverage_with_control_ranking | 0.000771 | [-0.009041, 0.011968] |
| neural_fold2_seed43_easy | hard | full_total | 0.002162 | [0.000000, 0.004777] |
| neural_fold2_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | hard | total | 0.002162 | [0.000000, 0.004777] |
| neural_fold2_seed43_easy | hard | ranking_at_control_count | 0.002162 | [0.000000, 0.004777] |
| neural_fold2_seed43_easy | hard | ranking_at_treatment_count | 0.002162 | [0.000000, 0.004777] |
| neural_fold2_seed43_easy | hard | coverage_with_treatment_ranking | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | hard | coverage_with_control_ranking | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | all | full_total | -0.058464 | [-0.097058, -0.015519] |
| damping097_fold2_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | all | total | -0.058464 | [-0.097058, -0.015519] |
| damping097_fold2_seed43_all | all | ranking_at_control_count | 0.007851 | [-0.025022, 0.043526] |
| damping097_fold2_seed43_all | all | ranking_at_treatment_count | 0.016614 | [-0.013594, 0.053768] |
| damping097_fold2_seed43_all | all | coverage_with_treatment_ranking | -0.066316 | [-0.082115, -0.049895] |
| damping097_fold2_seed43_all | all | coverage_with_control_ranking | -0.075078 | [-0.094973, -0.054171] |
| damping097_fold2_seed43_all | easy | full_total | 0.017671 | [-0.012458, 0.052001] |
| damping097_fold2_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | easy | total | 0.017671 | [-0.012458, 0.052001] |
| damping097_fold2_seed43_all | easy | ranking_at_control_count | -0.000097 | [-0.016648, 0.018274] |
| damping097_fold2_seed43_all | easy | ranking_at_treatment_count | 0.001626 | [-0.011596, 0.015945] |
| damping097_fold2_seed43_all | easy | coverage_with_treatment_ranking | 0.017767 | [0.000000, 0.046631] |
| damping097_fold2_seed43_all | easy | coverage_with_control_ranking | 0.016045 | [-0.003903, 0.038610] |
| damping097_fold2_seed43_all | hard | full_total | -0.065805 | [-0.108855, -0.017299] |
| damping097_fold2_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | hard | total | -0.065805 | [-0.108855, -0.017299] |
| damping097_fold2_seed43_all | hard | ranking_at_control_count | -0.003152 | [-0.040434, 0.038007] |
| damping097_fold2_seed43_all | hard | ranking_at_treatment_count | 0.024477 | [-0.014429, 0.073882] |
| damping097_fold2_seed43_all | hard | coverage_with_treatment_ranking | -0.062654 | [-0.088950, -0.035811] |
| damping097_fold2_seed43_all | hard | coverage_with_control_ranking | -0.090282 | [-0.122826, -0.060852] |
| damping097_fold2_seed43_easy | all | full_total | 0.114783 | [0.058525, 0.174809] |
| damping097_fold2_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | all | total | 0.114783 | [0.058525, 0.174809] |
| damping097_fold2_seed43_easy | all | ranking_at_control_count | 0.019038 | [0.001335, 0.035475] |
| damping097_fold2_seed43_easy | all | ranking_at_treatment_count | 0.034202 | [0.007913, 0.063558] |
| damping097_fold2_seed43_easy | all | coverage_with_treatment_ranking | 0.095745 | [0.044345, 0.154699] |
| damping097_fold2_seed43_easy | all | coverage_with_control_ranking | 0.080581 | [0.046610, 0.115316] |
| damping097_fold2_seed43_easy | easy | full_total | 0.068822 | [-0.012703, 0.139767] |
| damping097_fold2_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | easy | total | 0.068822 | [-0.012703, 0.139767] |
| damping097_fold2_seed43_easy | easy | ranking_at_control_count | -0.171509 | [-0.221438, -0.119210] |
| damping097_fold2_seed43_easy | easy | ranking_at_treatment_count | -0.137292 | [-0.194054, -0.082601] |
| damping097_fold2_seed43_easy | easy | coverage_with_treatment_ranking | 0.240331 | [0.162983, 0.313106] |
| damping097_fold2_seed43_easy | easy | coverage_with_control_ranking | 0.206114 | [0.119586, 0.299995] |
| damping097_fold2_seed43_easy | hard | full_total | 0.052172 | [0.020735, 0.088372] |
| damping097_fold2_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | hard | total | 0.052172 | [0.020735, 0.088372] |
| damping097_fold2_seed43_easy | hard | ranking_at_control_count | 0.007002 | [-0.012763, 0.023256] |
| damping097_fold2_seed43_easy | hard | ranking_at_treatment_count | 0.022824 | [0.005113, 0.044387] |
| damping097_fold2_seed43_easy | hard | coverage_with_treatment_ranking | 0.045169 | [0.007948, 0.094706] |
| damping097_fold2_seed43_easy | hard | coverage_with_control_ranking | 0.029347 | [0.012113, 0.047977] |

Three seeds, 3,000 paired locality resamples, dependent unadjusted development views.
Detector-track image pixels, obs8/pred12 rawstride12. Not independent confirmation, seconds, metric,
human gold, physical safety, true 3D or foundation. No deployment, Stage5C or SMC.
