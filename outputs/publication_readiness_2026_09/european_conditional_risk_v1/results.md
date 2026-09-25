# Conditional Event-Risk Results

Fresh moment-head fitting and fixed controls on cached_verified forecasts/utility. Source development only; all policies retained, no winner selection.

## Full Pointwise Population

| Policy (seed/event/head/support) | ADE vs CV (%) | Locality CI | ADE vs damping097 (%) | ADE vs previous policy (%) | Hard gain (%) | Easy degradation (%) | Worst easy degradation (%) | Zero-CV harmed | Switch rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17_all_ridge_no_guard | 0.606207 | [0.237650, 1.055199] | -3.740215 | -3.959692 | 0.825459 | -0.008753 | 0.921751 | 0/4 | 0.039816 |
| 17_all_ridge_source_zero_guard | 0.271810 | [0.076816, 0.521747] | -4.089550 | -4.326650 | 0.295398 | -0.058912 | 0.187959 | 0/4 | 0.006487 |
| 17_all_neural_underharm4_no_guard | 0.147819 | [0.040380, 0.284172] | -4.229044 | -4.456688 | 0.148466 | -0.040621 | 0.042697 | 0/4 | 0.002931 |
| 17_all_neural_underharm4_source_zero_guard | 0.016001 | [0.000096, 0.034307] | -4.362725 | -4.597395 | 0.016444 | -0.013799 | -0.000000 | 0/4 | 0.000812 |
| 17_easy_ridge_no_guard | 0.277774 | [0.109166, 0.443941] | -4.088441 | -4.324462 | 0.312680 | -0.192841 | 0.049563 | 1/4 | 0.027250 |
| 17_easy_ridge_source_zero_guard | 0.240094 | [0.099757, 0.406665] | -4.124568 | -4.361705 | 0.295898 | -0.093345 | -0.000000 | 0/4 | 0.009760 |
| 17_easy_neural_underharm4_no_guard | 0.238185 | [0.124989, 0.359507] | -4.128946 | -4.365721 | 0.247413 | -0.226638 | 0.022458 | 0/4 | 0.012456 |
| 17_easy_neural_underharm4_source_zero_guard | 0.214052 | [0.093352, 0.344594] | -4.151961 | -4.389840 | 0.244522 | -0.158505 | -0.000000 | 0/4 | 0.005502 |
| 29_all_ridge_no_guard | 0.480054 | [0.199900, 0.830184] | -3.873743 | -3.997207 | 0.655257 | 0.070439 | 0.863692 | 0/4 | 0.027115 |
| 29_all_ridge_source_zero_guard | 0.217289 | [0.073080, 0.401116] | -4.147813 | -4.287992 | 0.242706 | 0.017061 | 0.287289 | 0/4 | 0.004220 |
| 29_all_neural_underharm4_no_guard | 0.046832 | [0.019850, 0.079819] | -4.332842 | -4.468154 | 0.036051 | -0.008396 | 0.001165 | 0/4 | 0.000925 |
| 29_all_neural_underharm4_source_zero_guard | 0.017877 | [0.002758, 0.036321] | -4.360703 | -4.497941 | 0.020089 | -0.003094 | -0.000000 | 0/4 | 0.000599 |
| 29_easy_ridge_no_guard | 0.318885 | [0.209381, 0.438062] | -4.045106 | -4.182755 | 0.357574 | -0.088498 | -0.000000 | 1/4 | 0.011079 |
| 29_easy_ridge_source_zero_guard | 0.265211 | [0.128488, 0.404835] | -4.097812 | -4.238194 | 0.319398 | -0.044474 | -0.000000 | 0/4 | 0.007568 |
| 29_easy_neural_underharm4_no_guard | 0.166325 | [0.061813, 0.287492] | -4.202634 | -4.342747 | 0.188589 | -0.090090 | -0.000000 | 0/4 | 0.002944 |
| 29_easy_neural_underharm4_source_zero_guard | 0.166112 | [0.061582, 0.287473] | -4.202856 | -4.342984 | 0.188589 | -0.089751 | -0.000000 | 0/4 | 0.002913 |
| 43_all_ridge_no_guard | 0.606235 | [0.251856, 1.075739] | -3.745114 | -4.133064 | 0.755218 | 0.026442 | 0.730712 | 0/4 | 0.025498 |
| 43_all_ridge_source_zero_guard | 0.241997 | [0.082750, 0.474423] | -4.121684 | -4.528991 | 0.269635 | -0.065549 | 0.150197 | 0/4 | 0.005251 |
| 43_all_neural_underharm4_no_guard | 0.103280 | [0.032439, 0.186061] | -4.274868 | -4.676564 | 0.092770 | -0.015670 | 0.008640 | 0/4 | 0.001937 |
| 43_all_neural_underharm4_source_zero_guard | 0.016701 | [0.004871, 0.031215] | -4.361943 | -4.768194 | 0.011881 | -0.006184 | -0.000000 | 0/4 | 0.000640 |
| 43_easy_ridge_no_guard | 0.366214 | [0.199611, 0.545921] | -3.995168 | -4.400903 | 0.428506 | -0.076708 | 0.003324 | 0/4 | 0.011744 |
| 43_easy_ridge_source_zero_guard | 0.337737 | [0.165255, 0.528394] | -4.022451 | -4.429559 | 0.413598 | -0.067814 | -0.000000 | 0/4 | 0.008082 |
| 43_easy_neural_underharm4_no_guard | 0.431034 | [0.146754, 0.776618] | -3.920518 | -4.327777 | 0.502771 | -0.073234 | 0.817013 | 0/4 | 0.005345 |
| 43_easy_neural_underharm4_source_zero_guard | 0.427527 | [0.142612, 0.773629] | -3.923647 | -4.331178 | 0.502771 | -0.073234 | 0.817013 | 0/4 | 0.005333 |

## Joint Pilot

1,152 queries / 6,116 targets, not the full cohort. This pilot has no zero-CV cases; none of its safety flags certifies zero-event protection.

| Policy | Rule | ADE vs CV (%) | Locality CI | Easy degradation (%) | Worst easy degradation (%) | Switch rate |
|---|---|---:|---|---:|---:|---:|
| 17_all_ridge_no_guard | independent | 2.551955 | [1.033084, 4.435375] | 2.918330 | 17.406240 | 0.148463 |
| 17_all_ridge_no_guard | scene_uniform | 0.179160 | [-0.033752, 0.499692] | -0.190789 | -0.000000 | 0.005723 |
| 17_all_ridge_no_guard | joint | 2.545106 | [1.035037, 4.425403] | 2.920140 | 17.406240 | 0.148300 |
| 17_all_ridge_no_guard | unary_exact | 2.543997 | [1.035038, 4.424193] | 2.918330 | 17.406240 | 0.148463 |
| 17_all_ridge_no_guard | joint_exact | 2.545178 | [1.035038, 4.425403] | 2.918330 | 17.406240 | 0.148463 |
| 17_all_ridge_source_zero_guard | independent | 1.040416 | [0.304117, 2.011643] | 0.631898 | 3.644104 | 0.048234 |
| 17_all_ridge_source_zero_guard | scene_uniform | 0.059167 | [-0.024039, 0.177480] | -0.189158 | -0.000000 | 0.003270 |
| 17_all_ridge_source_zero_guard | joint | 1.041145 | [0.304308, 2.012697] | 0.633707 | 3.644104 | 0.048071 |
| 17_all_ridge_source_zero_guard | unary_exact | 1.041217 | [0.304384, 2.012697] | 0.631898 | 3.644104 | 0.048234 |
| 17_all_ridge_source_zero_guard | joint_exact | 1.041217 | [0.304384, 2.012697] | 0.631898 | 3.644104 | 0.048234 |
| 17_all_neural_underharm4_no_guard | independent | 0.725259 | [0.122601, 1.507700] | 0.166977 | 4.333576 | 0.067855 |
| 17_all_neural_underharm4_no_guard | scene_uniform | 0.013743 | [-0.034107, 0.067268] | -0.001632 | -0.000000 | 0.001472 |
| 17_all_neural_underharm4_no_guard | joint | 0.726624 | [0.122732, 1.509765] | 0.166977 | 4.333576 | 0.067691 |
| 17_all_neural_underharm4_no_guard | unary_exact | 0.725941 | [0.122732, 1.509031] | 0.166977 | 4.333576 | 0.067855 |
| 17_all_neural_underharm4_no_guard | joint_exact | 0.725941 | [0.122732, 1.509031] | 0.166977 | 4.333576 | 0.067855 |
| 17_all_neural_underharm4_source_zero_guard | independent | 0.165453 | [0.011777, 0.356270] | 0.292802 | 4.333576 | 0.023545 |
| 17_all_neural_underharm4_source_zero_guard | scene_uniform | 0.010054 | [-0.003901, 0.034062] | -0.000000 | -0.000000 | 0.000491 |
| 17_all_neural_underharm4_source_zero_guard | joint | 0.166135 | [0.011777, 0.357652] | 0.292802 | 4.333576 | 0.023545 |
| 17_all_neural_underharm4_source_zero_guard | unary_exact | 0.166135 | [0.011777, 0.357652] | 0.292802 | 4.333576 | 0.023545 |
| 17_all_neural_underharm4_source_zero_guard | joint_exact | 0.166135 | [0.011777, 0.357652] | 0.292802 | 4.333576 | 0.023545 |
| 17_easy_ridge_no_guard | independent | 1.305292 | [0.586330, 2.335321] | 0.424000 | 3.997974 | 0.084042 |
| 17_easy_ridge_no_guard | scene_uniform | 0.025990 | [0.007376, 0.048428] | -0.159773 | -0.000000 | 0.001472 |
| 17_easy_ridge_no_guard | joint | 1.303431 | [0.586330, 2.329740] | 0.422860 | 3.997974 | 0.083878 |
| 17_easy_ridge_no_guard | unary_exact | 1.305292 | [0.586330, 2.335321] | 0.424000 | 3.997974 | 0.084042 |
| 17_easy_ridge_no_guard | joint_exact | 1.305292 | [0.586330, 2.335321] | 0.424000 | 3.997974 | 0.084042 |
| 17_easy_ridge_source_zero_guard | independent | 0.537250 | [0.184247, 0.902581] | -0.222264 | 0.137417 | 0.021419 |
| 17_easy_ridge_source_zero_guard | scene_uniform | 0.008931 | [0.001613, 0.017534] | -0.117819 | -0.000000 | 0.000818 |
| 17_easy_ridge_source_zero_guard | joint | 0.537250 | [0.184247, 0.902581] | -0.222264 | 0.137417 | 0.021419 |
| 17_easy_ridge_source_zero_guard | unary_exact | 0.537250 | [0.184247, 0.902581] | -0.222264 | 0.137417 | 0.021419 |
| 17_easy_ridge_source_zero_guard | joint_exact | 0.537250 | [0.184247, 0.902581] | -0.222264 | 0.137417 | 0.021419 |
| 17_easy_neural_underharm4_no_guard | independent | 1.230058 | [0.555853, 2.107936] | -0.179552 | 1.076986 | 0.047090 |
| 17_easy_neural_underharm4_no_guard | scene_uniform | 0.125778 | [0.025054, 0.251426] | -0.207326 | -0.000000 | 0.003761 |
| 17_easy_neural_underharm4_no_guard | joint | 1.230058 | [0.555853, 2.107936] | -0.179552 | 1.076986 | 0.047090 |
| 17_easy_neural_underharm4_no_guard | unary_exact | 1.230058 | [0.555853, 2.107936] | -0.179552 | 1.076986 | 0.047090 |
| 17_easy_neural_underharm4_no_guard | joint_exact | 1.230058 | [0.555853, 2.107936] | -0.179552 | 1.076986 | 0.047090 |
| 17_easy_neural_underharm4_source_zero_guard | independent | 0.837667 | [0.228124, 1.724308] | -0.203453 | -0.000000 | 0.016024 |
| 17_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.118340 | [0.015219, 0.247202] | -0.167005 | -0.000000 | 0.003597 |
| 17_easy_neural_underharm4_source_zero_guard | joint | 0.837667 | [0.228124, 1.724308] | -0.203453 | -0.000000 | 0.016024 |
| 17_easy_neural_underharm4_source_zero_guard | unary_exact | 0.837667 | [0.228124, 1.724308] | -0.203453 | -0.000000 | 0.016024 |
| 17_easy_neural_underharm4_source_zero_guard | joint_exact | 0.837667 | [0.228124, 1.724308] | -0.203453 | -0.000000 | 0.016024 |
| 29_all_ridge_no_guard | independent | 2.508671 | [1.064853, 4.197329] | 2.271866 | 17.670016 | 0.095160 |
| 29_all_ridge_no_guard | scene_uniform | 0.130825 | [-0.056798, 0.436209] | -0.001639 | -0.000000 | 0.003434 |
| 29_all_ridge_no_guard | joint | 2.482530 | [1.060944, 4.158706] | 2.271866 | 17.670016 | 0.095160 |
| 29_all_ridge_no_guard | unary_exact | 2.506423 | [1.062661, 4.194608] | 2.271866 | 17.670016 | 0.095160 |
| 29_all_ridge_no_guard | joint_exact | 2.482530 | [1.060944, 4.158706] | 2.271866 | 17.670016 | 0.095160 |
| 29_all_ridge_source_zero_guard | independent | 1.174574 | [0.395029, 2.272798] | -0.012882 | 0.617488 | 0.028777 |
| 29_all_ridge_source_zero_guard | scene_uniform | 0.022951 | [0.000000, 0.056030] | -0.000000 | -0.000000 | 0.000818 |
| 29_all_ridge_source_zero_guard | joint | 1.174574 | [0.395029, 2.272798] | -0.012882 | 0.617488 | 0.028777 |
| 29_all_ridge_source_zero_guard | unary_exact | 1.174574 | [0.395029, 2.272798] | -0.012882 | 0.617488 | 0.028777 |
| 29_all_ridge_source_zero_guard | joint_exact | 1.174574 | [0.395029, 2.272798] | -0.012882 | 0.617488 | 0.028777 |
| 29_all_neural_underharm4_no_guard | independent | 0.623789 | [0.210043, 1.108166] | 0.452095 | 4.175322 | 0.038260 |
| 29_all_neural_underharm4_no_guard | scene_uniform | 0.007112 | [-0.034766, 0.048586] | -0.000000 | -0.000000 | 0.000981 |
| 29_all_neural_underharm4_no_guard | joint | 0.624753 | [0.210043, 1.109523] | 0.452095 | 4.175322 | 0.038260 |
| 29_all_neural_underharm4_no_guard | unary_exact | 0.624753 | [0.210043, 1.109523] | 0.452095 | 4.175322 | 0.038260 |
| 29_all_neural_underharm4_no_guard | joint_exact | 0.624753 | [0.210043, 1.109523] | 0.452095 | 4.175322 | 0.038260 |
| 29_all_neural_underharm4_source_zero_guard | independent | 0.212668 | [0.011075, 0.450011] | -0.043611 | -0.000000 | 0.013407 |
| 29_all_neural_underharm4_source_zero_guard | scene_uniform | 0.012045 | [-0.004348, 0.040483] | -0.000000 | -0.000000 | 0.000491 |
| 29_all_neural_underharm4_source_zero_guard | joint | 0.212668 | [0.011075, 0.450011] | -0.043611 | -0.000000 | 0.013407 |
| 29_all_neural_underharm4_source_zero_guard | unary_exact | 0.212668 | [0.011075, 0.450011] | -0.043611 | -0.000000 | 0.013407 |
| 29_all_neural_underharm4_source_zero_guard | joint_exact | 0.212668 | [0.011075, 0.450011] | -0.043611 | -0.000000 | 0.013407 |
| 29_easy_ridge_no_guard | independent | 1.523264 | [0.730735, 2.478144] | 0.850489 | 5.893563 | 0.053630 |
| 29_easy_ridge_no_guard | scene_uniform | 0.014555 | [0.000907, 0.035299] | -0.001639 | -0.000000 | 0.000654 |
| 29_easy_ridge_no_guard | joint | 1.520107 | [0.727657, 2.476136] | 0.852419 | 5.893563 | 0.053303 |
| 29_easy_ridge_no_guard | unary_exact | 1.523264 | [0.730735, 2.478144] | 0.850489 | 5.893563 | 0.053630 |
| 29_easy_ridge_no_guard | joint_exact | 1.523264 | [0.730735, 2.478144] | 0.850489 | 5.893563 | 0.053630 |
| 29_easy_ridge_source_zero_guard | independent | 0.711935 | [0.215555, 1.338585] | -0.071761 | -0.000000 | 0.015370 |
| 29_easy_ridge_source_zero_guard | scene_uniform | 0.004701 | [0.000000, 0.012242] | -0.000000 | -0.000000 | 0.000327 |
| 29_easy_ridge_source_zero_guard | joint | 0.708778 | [0.215555, 1.337484] | -0.069831 | -0.000000 | 0.015043 |
| 29_easy_ridge_source_zero_guard | unary_exact | 0.711935 | [0.215555, 1.338585] | -0.071761 | -0.000000 | 0.015370 |
| 29_easy_ridge_source_zero_guard | joint_exact | 0.711935 | [0.215555, 1.338585] | -0.071761 | -0.000000 | 0.015370 |
| 29_easy_neural_underharm4_no_guard | independent | 0.757208 | [0.267058, 1.345995] | 0.153711 | 2.103410 | 0.020929 |
| 29_easy_neural_underharm4_no_guard | scene_uniform | 0.028805 | [0.003061, 0.065049] | -0.022536 | -0.000000 | 0.001308 |
| 29_easy_neural_underharm4_no_guard | joint | 0.757208 | [0.267058, 1.345995] | 0.153711 | 2.103410 | 0.020929 |
| 29_easy_neural_underharm4_no_guard | unary_exact | 0.757208 | [0.267058, 1.345995] | 0.153711 | 2.103410 | 0.020929 |
| 29_easy_neural_underharm4_no_guard | joint_exact | 0.757208 | [0.267058, 1.345995] | 0.153711 | 2.103410 | 0.020929 |
| 29_easy_neural_underharm4_source_zero_guard | independent | 0.709456 | [0.203943, 1.313875] | -0.022536 | -0.000000 | 0.008502 |
| 29_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.028805 | [0.003061, 0.065049] | -0.022536 | -0.000000 | 0.001308 |
| 29_easy_neural_underharm4_source_zero_guard | joint | 0.709456 | [0.203943, 1.313875] | -0.022536 | -0.000000 | 0.008502 |
| 29_easy_neural_underharm4_source_zero_guard | unary_exact | 0.709456 | [0.203943, 1.313875] | -0.022536 | -0.000000 | 0.008502 |
| 29_easy_neural_underharm4_source_zero_guard | joint_exact | 0.709456 | [0.203943, 1.313875] | -0.022536 | -0.000000 | 0.008502 |
| 43_all_ridge_no_guard | independent | 2.754096 | [1.324746, 4.470683] | 0.780540 | 6.096276 | 0.101864 |
| 43_all_ridge_no_guard | scene_uniform | 0.221282 | [-0.071182, 0.685394] | -0.068161 | -0.000000 | 0.005723 |
| 43_all_ridge_no_guard | joint | 2.727975 | [1.324746, 4.413968] | 0.780540 | 6.096276 | 0.101700 |
| 43_all_ridge_no_guard | unary_exact | 2.727932 | [1.311205, 4.427382] | 0.773511 | 6.096276 | 0.101864 |
| 43_all_ridge_no_guard | joint_exact | 2.734787 | [1.324746, 4.427553] | 0.780540 | 6.096276 | 0.101864 |
| 43_all_ridge_source_zero_guard | independent | 1.160826 | [0.405416, 2.228375] | -0.174208 | 0.343868 | 0.039405 |
| 43_all_ridge_source_zero_guard | scene_uniform | 0.034295 | [-0.073947, 0.166872] | -0.066528 | -0.000000 | 0.001962 |
| 43_all_ridge_source_zero_guard | joint | 1.160826 | [0.405416, 2.228375] | -0.174208 | 0.343868 | 0.039405 |
| 43_all_ridge_source_zero_guard | unary_exact | 1.160826 | [0.405416, 2.228375] | -0.174208 | 0.343868 | 0.039405 |
| 43_all_ridge_source_zero_guard | joint_exact | 1.160826 | [0.405416, 2.228375] | -0.174208 | 0.343868 | 0.039405 |
| 43_all_neural_underharm4_no_guard | independent | 0.831712 | [0.217215, 1.622977] | 0.062086 | 1.225279 | 0.051504 |
| 43_all_neural_underharm4_no_guard | scene_uniform | 0.007064 | [-0.035944, 0.047648] | -0.001633 | -0.000000 | 0.001472 |
| 43_all_neural_underharm4_no_guard | joint | 0.825642 | [0.217215, 1.606632] | 0.062086 | 1.225279 | 0.051504 |
| 43_all_neural_underharm4_no_guard | unary_exact | 0.825642 | [0.217215, 1.606632] | 0.062086 | 1.225279 | 0.051504 |
| 43_all_neural_underharm4_no_guard | joint_exact | 0.825642 | [0.217215, 1.606632] | 0.062086 | 1.225279 | 0.051504 |
| 43_all_neural_underharm4_source_zero_guard | independent | 0.234533 | [0.068491, 0.431051] | 0.161168 | 1.225279 | 0.019294 |
| 43_all_neural_underharm4_source_zero_guard | scene_uniform | 0.010755 | [-0.004538, 0.036803] | -0.000000 | -0.000000 | 0.000491 |
| 43_all_neural_underharm4_source_zero_guard | joint | 0.234533 | [0.068491, 0.431051] | 0.161168 | 1.225279 | 0.019294 |
| 43_all_neural_underharm4_source_zero_guard | unary_exact | 0.234533 | [0.068491, 0.431051] | 0.161168 | 1.225279 | 0.019294 |
| 43_all_neural_underharm4_source_zero_guard | joint_exact | 0.234533 | [0.068491, 0.431051] | 0.161168 | 1.225279 | 0.019294 |
| 43_easy_ridge_no_guard | independent | 1.404575 | [0.784180, 2.270805] | 0.209852 | 1.775571 | 0.058862 |
| 43_easy_ridge_no_guard | scene_uniform | 0.006573 | [0.000603, 0.014686] | -0.054822 | -0.000000 | 0.000818 |
| 43_easy_ridge_no_guard | joint | 1.398609 | [0.779268, 2.258847] | 0.209852 | 1.775571 | 0.058862 |
| 43_easy_ridge_no_guard | unary_exact | 1.398609 | [0.779268, 2.258847] | 0.209852 | 1.775571 | 0.058862 |
| 43_easy_ridge_no_guard | joint_exact | 1.398609 | [0.779268, 2.258847] | 0.209852 | 1.775571 | 0.058862 |
| 43_easy_ridge_source_zero_guard | independent | 0.671625 | [0.309968, 1.039284] | -0.007131 | 0.297231 | 0.019130 |
| 43_easy_ridge_source_zero_guard | scene_uniform | 0.006271 | [0.000000, 0.014383] | -0.053189 | -0.000000 | 0.000654 |
| 43_easy_ridge_source_zero_guard | joint | 0.673220 | [0.310118, 1.043922] | -0.007131 | 0.297231 | 0.019130 |
| 43_easy_ridge_source_zero_guard | unary_exact | 0.673220 | [0.310118, 1.043922] | -0.007131 | 0.297231 | 0.019130 |
| 43_easy_ridge_source_zero_guard | joint_exact | 0.673220 | [0.310118, 1.043922] | -0.007131 | 0.297231 | 0.019130 |
| 43_easy_neural_underharm4_no_guard | independent | 1.143571 | [0.397244, 2.089306] | 0.089538 | 2.039103 | 0.032374 |
| 43_easy_neural_underharm4_no_guard | scene_uniform | 0.123523 | [0.025479, 0.245068] | -0.050451 | 0.302724 | 0.004415 |
| 43_easy_neural_underharm4_no_guard | joint | 1.143571 | [0.397244, 2.089306] | 0.089538 | 2.039103 | 0.032374 |
| 43_easy_neural_underharm4_no_guard | unary_exact | 1.143571 | [0.397244, 2.089306] | 0.089538 | 2.039103 | 0.032374 |
| 43_easy_neural_underharm4_no_guard | joint_exact | 1.143571 | [0.397244, 2.089306] | 0.089538 | 2.039103 | 0.032374 |
| 43_easy_neural_underharm4_source_zero_guard | independent | 1.077518 | [0.333143, 2.025891] | -0.080387 | 0.302724 | 0.017005 |
| 43_easy_neural_underharm4_source_zero_guard | scene_uniform | 0.123523 | [0.025479, 0.245068] | -0.050451 | 0.302724 | 0.004415 |
| 43_easy_neural_underharm4_source_zero_guard | joint | 1.077518 | [0.333143, 2.025891] | -0.080387 | 0.302724 | 0.017005 |
| 43_easy_neural_underharm4_source_zero_guard | unary_exact | 1.077518 | [0.333143, 2.025891] | -0.080387 | 0.302724 | 0.017005 |
| 43_easy_neural_underharm4_source_zero_guard | joint_exact | 1.077518 | [0.333143, 2.025891] | -0.080387 | 0.302724 | 0.017005 |

## Paired Factorial Contrasts

| Contrast | Rule | ADE gain (%) | Conditional CI |
|---|---|---:|---|
| 17_ridge_no_guard_easy_vs_all | pointwise | -0.336458 | [-0.883275, 0.112263] |
| 17_ridge_no_guard_easy_vs_all | joint | -1.335289 | [-2.820295, -0.203885] |
| 17_ridge_source_zero_guard_easy_vs_all | pointwise | -0.033054 | [-0.285866, 0.180976] |
| 17_ridge_source_zero_guard_easy_vs_all | joint | -0.527350 | [-1.371793, 0.079028] |
| 17_neural_underharm4_no_guard_easy_vs_all | pointwise | 0.089835 | [-0.129911, 0.280352] |
| 17_neural_underharm4_no_guard_easy_vs_all | joint | 0.494952 | [-0.423874, 1.486204] |
| 17_neural_underharm4_source_zero_guard_easy_vs_all | pointwise | 0.198116 | [0.087725, 0.320360] |
| 17_neural_underharm4_source_zero_guard_easy_vs_all | joint | 0.673439 | [0.153975, 1.513092] |
| 29_ridge_no_guard_easy_vs_all | pointwise | -0.165509 | [-0.573312, 0.164794] |
| 29_ridge_no_guard_easy_vs_all | joint | -1.036455 | [-2.383404, 0.017441] |
| 29_ridge_source_zero_guard_easy_vs_all | pointwise | 0.047414 | [-0.145479, 0.213488] |
| 29_ridge_source_zero_guard_easy_vs_all | joint | -0.487599 | [-1.181980, 0.077661] |
| 29_neural_underharm4_no_guard_easy_vs_all | pointwise | 0.119496 | [0.003416, 0.247649] |
| 29_neural_underharm4_no_guard_easy_vs_all | joint | 0.126057 | [-0.604624, 0.879742] |
| 29_neural_underharm4_source_zero_guard_easy_vs_all | pointwise | 0.148279 | [0.050968, 0.265131] |
| 29_neural_underharm4_source_zero_guard_easy_vs_all | joint | 0.498171 | [0.037526, 1.033468] |
| 43_ridge_no_guard_easy_vs_all | pointwise | -0.248058 | [-0.818962, 0.230372] |
| 43_ridge_no_guard_easy_vs_all | joint | -1.423787 | [-2.895424, -0.236033] |
| 43_ridge_source_zero_guard_easy_vs_all | pointwise | 0.094786 | [-0.178335, 0.353737] |
| 43_ridge_source_zero_guard_easy_vs_all | joint | -0.515527 | [-1.478996, 0.135209] |
| 43_neural_underharm4_no_guard_easy_vs_all | pointwise | 0.327590 | [0.002051, 0.704498] |
| 43_neural_underharm4_no_guard_easy_vs_all | joint | 0.304653 | [-0.829771, 1.457388] |
| 43_neural_underharm4_source_zero_guard_easy_vs_all | pointwise | 0.410938 | [0.133342, 0.755449] |
| 43_neural_underharm4_source_zero_guard_easy_vs_all | joint | 0.847166 | [0.209001, 1.697097] |

## Matched Intervention Contrasts

| Policy | Contrast | ADE gain (%) | Conditional CI |
|---|---|---:|---|
| 17_all_ridge_no_guard | joint_vs_independent | -0.007387 | [-0.021290, 0.001396] |
| 17_all_ridge_no_guard | joint_exact_vs_independent | -0.007574 | [-0.022470, 0.002182] |
| 17_all_ridge_no_guard | joint_exact_vs_unary | 0.001395 | [0.000000, 0.004185] |
| 17_all_ridge_source_zero_guard | joint_vs_independent | 0.000750 | [-0.000145, 0.002271] |
| 17_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_all_neural_underharm4_no_guard | joint_vs_independent | 0.001399 | [0.000000, 0.003489] |
| 17_all_neural_underharm4_no_guard | joint_exact_vs_independent | 0.001144 | [0.000000, 0.003431] |
| 17_all_neural_underharm4_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 17_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000690 | [0.000000, 0.002069] |
| 17_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_easy_ridge_no_guard | joint_vs_independent | -0.001987 | [-0.005962, 0.000000] |
| 17_easy_ridge_no_guard | joint_exact_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_easy_ridge_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 17_easy_ridge_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 17_easy_neural_underharm4_no_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_easy_neural_underharm4_no_guard | joint_exact_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_easy_neural_underharm4_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 17_easy_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 17_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 17_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_all_ridge_no_guard | joint_vs_independent | -0.028116 | [-0.084347, 0.000000] |
| 29_all_ridge_no_guard | joint_exact_vs_independent | -0.032025 | [-0.096075, 0.000000] |
| 29_all_ridge_no_guard | joint_exact_vs_unary | -0.029261 | [-0.087784, 0.000000] |
| 29_all_ridge_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_all_neural_underharm4_no_guard | joint_vs_independent | 0.000988 | [0.000000, 0.002965] |
| 29_all_neural_underharm4_no_guard | joint_exact_vs_independent | undefined | undefined |
| 29_all_neural_underharm4_no_guard | joint_exact_vs_unary | undefined | undefined |
| 29_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_easy_ridge_no_guard | joint_vs_independent | -0.003208 | [-0.009623, 0.000000] |
| 29_easy_ridge_no_guard | joint_exact_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_easy_ridge_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 29_easy_ridge_source_zero_guard | joint_vs_independent | -0.003208 | [-0.009623, 0.000000] |
| 29_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 29_easy_neural_underharm4_no_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_easy_neural_underharm4_no_guard | joint_exact_vs_independent | undefined | undefined |
| 29_easy_neural_underharm4_no_guard | joint_exact_vs_unary | undefined | undefined |
| 29_easy_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 29_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 29_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_all_ridge_no_guard | joint_vs_independent | -0.028368 | [-0.085105, 0.000000] |
| 43_all_ridge_no_guard | joint_exact_vs_independent | -0.023643 | [-0.070930, 0.000000] |
| 43_all_ridge_no_guard | joint_exact_vs_unary | 0.007792 | [0.000000, 0.023377] |
| 43_all_ridge_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 43_all_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_all_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_all_neural_underharm4_no_guard | joint_vs_independent | -0.006295 | [-0.018884, 0.000000] |
| 43_all_neural_underharm4_no_guard | joint_exact_vs_independent | -0.014307 | [-0.042920, 0.000000] |
| 43_all_neural_underharm4_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 43_all_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 43_all_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_all_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_easy_ridge_no_guard | joint_vs_independent | -0.006041 | [-0.019971, 0.003248] |
| 43_easy_ridge_no_guard | joint_exact_vs_independent | -0.015528 | [-0.051956, 0.005984] |
| 43_easy_ridge_no_guard | joint_exact_vs_unary | 0.000000 | [0.000000, 0.000000] |
| 43_easy_ridge_source_zero_guard | joint_vs_independent | 0.001624 | [0.000000, 0.004872] |
| 43_easy_ridge_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_easy_ridge_source_zero_guard | joint_exact_vs_unary | undefined | undefined |
| 43_easy_neural_underharm4_no_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 43_easy_neural_underharm4_no_guard | joint_exact_vs_independent | undefined | undefined |
| 43_easy_neural_underharm4_no_guard | joint_exact_vs_unary | undefined | undefined |
| 43_easy_neural_underharm4_source_zero_guard | joint_vs_independent | 0.000000 | [0.000000, 0.000000] |
| 43_easy_neural_underharm4_source_zero_guard | joint_exact_vs_independent | undefined | undefined |
| 43_easy_neural_underharm4_source_zero_guard | joint_exact_vs_unary | undefined | undefined |

Undefined locality contrasts are retained, never zero-filled or recalculated after dropping unsupported localities.
All intervals use 3,000 locality resamples conditional on these development sources and fitted models. They are not independent calibration or confirmation.
Positive-easy mean/worst <=2% and zero-CV added harm0 are unchanged. A zero-switch policy is fallback, not neural contribution.
No deployment, metric/seconds/physical-safety claim, Stage5C or SMC.
