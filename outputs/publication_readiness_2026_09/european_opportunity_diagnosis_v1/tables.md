# Candidate Opportunity and Rejection Tables

Fresh arithmetic on cached_verified source models. Oracle uses future labels only for diagnosis.
No new fitting, threshold choice, reserved readout or deployment.

## Candidate Opportunity

| Seed | Candidate | Raw ADE gain vs CV (%) | Hindsight oracle gain (%) | Oracle conditional locality CI |
|---|---|---:|---:|---|
| 17 | neural | 2.449067 | 17.363789 | [14.173584, 20.699953] |
| 17 | single_lower_fold | 4.124797 | 18.047374 | [16.083693, 19.658409] |
| 17 | single_upper_fold | 1.429576 | 21.618611 | [18.646619, 24.295589] |
| 17 | damping097 | 3.975483 | 8.733475 | [8.322765, 9.111789] |
| 29 | neural | 1.870192 | 17.243128 | [13.924318, 20.636333] |
| 29 | single_lower_fold | 4.083927 | 18.072764 | [16.132126, 19.628138] |
| 29 | single_upper_fold | 1.388888 | 21.523135 | [18.586065, 24.125060] |
| 29 | damping097 | 3.975483 | 8.733475 | [8.322765, 9.111789] |
| 43 | neural | 2.153530 | 17.156434 | [13.955867, 20.410001] |
| 43 | single_lower_fold | 3.420427 | 17.846939 | [16.035797, 19.322108] |
| 43 | single_upper_fold | 1.693674 | 21.363034 | [18.412105, 24.028768] |
| 43 | damping097 | 3.975483 | 8.733475 | [8.322765, 9.111789] |

## Conserved Policy Decomposition

Contributions are percentage points of the same locality CV error denominator. Oracle is not a learned result.
Gate priority: support, motion, utility, event mass, risk. Sequential attribution is not an intervention effect.
| Policy | Subset | Oracle | Captured | Switched harm | Net | Support missed | Motion missed | Utility missed | Zero-mass missed | Risk missed | Gross capture (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17_neural_all_ridge_no_guard | all | 17.363789 | 0.927745 | 0.321538 | 0.606207 | 0.000000 | 0.000000 | 10.202271 | 0.023889 | 6.209884 | 5.342985 |
| 17_neural_all_ridge_no_guard | easy | 14.889270 | 0.309031 | 0.300278 | 0.008753 | 0.000000 | 0.000000 | 11.889666 | 0.135535 | 2.555038 | 2.075525 |
| 17_neural_all_ridge_no_guard | hard | 17.941537 | 1.008811 | 0.183352 | 0.825459 | 0.000000 | 0.000000 | 9.852783 | 0.005408 | 7.074536 | 5.622766 |
| 17_neural_all_ridge_source_zero_guard | all | 17.363789 | 0.321484 | 0.049674 | 0.271810 | 5.351730 | 0.000000 | 8.145290 | 0.006124 | 3.539161 | 1.851464 |
| 17_neural_all_ridge_source_zero_guard | easy | 14.889270 | 0.208739 | 0.149828 | 0.058912 | 4.718666 | 0.000000 | 8.333409 | 0.053750 | 1.574706 | 1.401943 |
| 17_neural_all_ridge_source_zero_guard | hard | 17.941537 | 0.312736 | 0.017338 | 0.295398 | 5.740226 | 0.000000 | 8.045501 | 0.001321 | 3.841753 | 1.743081 |
| 17_neural_all_neural_underharm4_no_guard | all | 17.363789 | 0.158690 | 0.010871 | 0.147819 | 0.000000 | 0.000000 | 10.202271 | 0.000000 | 7.002828 | 0.913912 |
| 17_neural_all_neural_underharm4_no_guard | easy | 14.889270 | 0.046419 | 0.005798 | 0.040621 | 0.000000 | 0.000000 | 11.889666 | 0.000000 | 2.953185 | 0.311762 |
| 17_neural_all_neural_underharm4_no_guard | hard | 17.941537 | 0.154265 | 0.005799 | 0.148466 | 0.000000 | 0.000000 | 9.852783 | 0.000000 | 7.934489 | 0.859819 |
| 17_neural_all_neural_underharm4_source_zero_guard | all | 17.363789 | 0.016405 | 0.000404 | 0.016001 | 5.351730 | 0.000000 | 8.145290 | 0.000000 | 3.850364 | 0.094481 |
| 17_neural_all_neural_underharm4_source_zero_guard | easy | 14.889270 | 0.013800 | 0.000001 | 0.013799 | 4.718666 | 0.000000 | 8.333409 | 0.000000 | 1.823395 | 0.092684 |
| 17_neural_all_neural_underharm4_source_zero_guard | hard | 17.941537 | 0.016770 | 0.000326 | 0.016444 | 5.740226 | 0.000000 | 8.045501 | 0.000000 | 4.139040 | 0.093472 |
| 17_neural_easy_ridge_no_guard | all | 17.363789 | 0.389687 | 0.111912 | 0.277774 | 0.000000 | 0.000000 | 10.202271 | 0.159110 | 6.612721 | 2.244248 |
| 17_neural_easy_ridge_no_guard | easy | 14.889270 | 0.223447 | 0.030606 | 0.192841 | 0.000000 | 0.000000 | 11.889666 | 0.011796 | 2.764361 | 1.500726 |
| 17_neural_easy_ridge_no_guard | hard | 17.941537 | 0.422450 | 0.109770 | 0.312680 | 0.000000 | 0.000000 | 9.852783 | 0.157900 | 7.508404 | 2.354592 |
| 17_neural_easy_ridge_source_zero_guard | all | 17.363789 | 0.254331 | 0.014237 | 0.240094 | 5.351730 | 0.000000 | 8.145290 | 0.080827 | 3.531612 | 1.464720 |
| 17_neural_easy_ridge_source_zero_guard | easy | 14.889270 | 0.100596 | 0.007250 | 0.093345 | 4.718666 | 0.000000 | 8.333409 | 0.004834 | 1.731765 | 0.675625 |
| 17_neural_easy_ridge_source_zero_guard | hard | 17.941537 | 0.305124 | 0.009226 | 0.295898 | 5.740226 | 0.000000 | 8.045501 | 0.092782 | 3.757904 | 1.700658 |
| 17_neural_easy_neural_underharm4_no_guard | all | 17.363789 | 0.250542 | 0.012357 | 0.238185 | 0.000000 | 0.000000 | 10.202271 | 0.000000 | 6.910975 | 1.442902 |
| 17_neural_easy_neural_underharm4_no_guard | easy | 14.889270 | 0.253005 | 0.026367 | 0.226638 | 0.000000 | 0.000000 | 11.889666 | 0.000000 | 2.746599 | 1.699247 |
| 17_neural_easy_neural_underharm4_no_guard | hard | 17.941537 | 0.253621 | 0.006208 | 0.247413 | 0.000000 | 0.000000 | 9.852783 | 0.000000 | 7.835133 | 1.413598 |
| 17_neural_easy_neural_underharm4_source_zero_guard | all | 17.363789 | 0.225862 | 0.011810 | 0.214052 | 5.351730 | 0.000000 | 8.145290 | 0.000000 | 3.640907 | 1.300765 |
| 17_neural_easy_neural_underharm4_source_zero_guard | easy | 14.889270 | 0.180858 | 0.022352 | 0.158505 | 4.718666 | 0.000000 | 8.333409 | 0.000000 | 1.656337 | 1.214684 |
| 17_neural_easy_neural_underharm4_source_zero_guard | hard | 17.941537 | 0.250487 | 0.005966 | 0.244522 | 5.740226 | 0.000000 | 8.045501 | 0.000000 | 3.905323 | 1.396132 |
| 17_damping097_all_ridge_no_guard | all | 8.733475 | 2.026184 | 0.256377 | 1.769807 | 0.000000 | 0.000000 | 2.695875 | 0.016769 | 3.994646 | 23.200205 |
| 17_damping097_all_ridge_no_guard | easy | 6.147387 | 0.674252 | 0.438223 | 0.236029 | 0.000000 | 0.000000 | 2.978216 | 0.091389 | 2.403530 | 10.968107 |
| 17_damping097_all_ridge_no_guard | hard | 8.618661 | 2.010992 | 0.117405 | 1.893587 | 0.000000 | 0.000000 | 2.626799 | 0.004064 | 3.976805 | 23.333002 |
| 17_damping097_all_ridge_source_zero_guard | all | 8.733475 | 0.796279 | 0.061942 | 0.734338 | 3.028899 | 0.000000 | 1.949589 | 0.002905 | 2.955802 | 9.117551 |
| 17_damping097_all_ridge_source_zero_guard | easy | 6.147387 | 0.219039 | 0.163100 | 0.055939 | 1.899982 | 0.000000 | 2.158389 | 0.027713 | 1.842264 | 3.563119 |
| 17_damping097_all_ridge_source_zero_guard | hard | 8.618661 | 0.780407 | 0.027367 | 0.753040 | 2.860470 | 0.000000 | 1.941526 | 0.000722 | 3.035535 | 9.054857 |
| 17_damping097_all_neural_underharm4_no_guard | all | 8.733475 | 0.843279 | 0.128866 | 0.714413 | 0.000000 | 0.000000 | 2.695875 | 0.000000 | 5.194321 | 9.655704 |
| 17_damping097_all_neural_underharm4_no_guard | easy | 6.147387 | 0.034163 | 0.069812 | -0.035649 | 0.000000 | 0.000000 | 2.978216 | 0.000000 | 3.135008 | 0.555736 |
| 17_damping097_all_neural_underharm4_no_guard | hard | 8.618661 | 0.956883 | 0.075724 | 0.881160 | 0.000000 | 0.000000 | 2.626799 | 0.000000 | 5.034978 | 11.102462 |
| 17_damping097_all_neural_underharm4_source_zero_guard | all | 8.733475 | 0.244732 | 0.007406 | 0.237326 | 3.028899 | 0.000000 | 1.949589 | 0.000000 | 3.510254 | 2.802233 |
| 17_damping097_all_neural_underharm4_source_zero_guard | easy | 6.147387 | 0.017331 | 0.011566 | 0.005765 | 1.899982 | 0.000000 | 2.158389 | 0.000000 | 2.071685 | 0.281927 |
| 17_damping097_all_neural_underharm4_source_zero_guard | hard | 8.618661 | 0.278345 | 0.003005 | 0.275340 | 2.860470 | 0.000000 | 1.941526 | 0.000000 | 3.538320 | 3.229557 |
| 17_damping097_easy_ridge_no_guard | all | 8.733475 | 2.074143 | 0.418015 | 1.656127 | 0.000000 | 0.000000 | 2.695875 | 0.120464 | 3.842993 | 23.749342 |
| 17_damping097_easy_ridge_no_guard | easy | 6.147387 | 1.397364 | 0.653879 | 0.743485 | 0.000000 | 0.000000 | 2.978216 | 0.010916 | 1.760891 | 22.731029 |
| 17_damping097_easy_ridge_no_guard | hard | 8.618661 | 1.918437 | 0.187105 | 1.731331 | 0.000000 | 0.000000 | 2.626799 | 0.126541 | 3.946883 | 22.259105 |
| 17_damping097_easy_ridge_source_zero_guard | all | 8.733475 | 1.151169 | 0.084643 | 1.066526 | 3.028899 | 0.000000 | 1.949589 | 0.069635 | 2.534182 | 13.181109 |
| 17_damping097_easy_ridge_source_zero_guard | easy | 6.147387 | 0.754085 | 0.161217 | 0.592867 | 1.899982 | 0.000000 | 2.158389 | 0.002648 | 1.332284 | 12.266750 |
| 17_damping097_easy_ridge_source_zero_guard | hard | 8.618661 | 1.144454 | 0.053885 | 1.090569 | 2.860470 | 0.000000 | 1.941526 | 0.079737 | 2.592473 | 13.278790 |
| 17_damping097_easy_neural_underharm4_no_guard | all | 8.733475 | 2.036177 | 0.225402 | 1.810775 | 0.000000 | 0.000000 | 2.695875 | 0.000000 | 4.001422 | 23.314625 |
| 17_damping097_easy_neural_underharm4_no_guard | easy | 6.147387 | 1.628771 | 0.340032 | 1.288739 | 0.000000 | 0.000000 | 2.978216 | 0.000000 | 1.540401 | 26.495331 |
| 17_damping097_easy_neural_underharm4_no_guard | hard | 8.618661 | 1.926891 | 0.154933 | 1.771958 | 0.000000 | 0.000000 | 2.626799 | 0.000000 | 4.064970 | 22.357203 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | all | 8.733475 | 2.007177 | 0.224448 | 1.782729 | 3.028899 | 0.000000 | 1.949589 | 0.000000 | 1.747809 | 22.982568 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | easy | 6.147387 | 1.459660 | 0.336394 | 1.123266 | 1.899982 | 0.000000 | 2.158389 | 0.000000 | 0.629357 | 23.744390 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | hard | 8.618661 | 1.923142 | 0.154198 | 1.768943 | 2.860470 | 0.000000 | 1.941526 | 0.000000 | 1.893523 | 22.313695 |
| 29_neural_all_ridge_no_guard | all | 17.243128 | 0.747322 | 0.267269 | 0.480054 | 0.000000 | 0.000000 | 10.538949 | 0.015381 | 5.941475 | 4.334031 |
| 29_neural_all_ridge_no_guard | easy | 14.854704 | 0.175300 | 0.245739 | -0.070439 | 0.000000 | 0.000000 | 13.873351 | 0.079476 | 0.726577 | 1.180101 |
| 29_neural_all_ridge_no_guard | hard | 17.815047 | 0.809162 | 0.153905 | 0.655257 | 0.000000 | 0.000000 | 9.771687 | 0.001829 | 7.232369 | 4.542017 |
| 29_neural_all_ridge_source_zero_guard | all | 17.243128 | 0.265266 | 0.047977 | 0.217289 | 5.480552 | 0.000000 | 8.104149 | 0.003293 | 3.389868 | 1.538385 |
| 29_neural_all_ridge_source_zero_guard | easy | 14.854704 | 0.102511 | 0.119573 | -0.017061 | 4.738647 | 0.000000 | 9.692976 | 0.027413 | 0.293157 | 0.690094 |
| 29_neural_all_ridge_source_zero_guard | hard | 17.815047 | 0.265554 | 0.022848 | 0.242706 | 5.926679 | 0.000000 | 7.626610 | 0.000290 | 3.995914 | 1.490616 |
| 29_neural_all_neural_underharm4_no_guard | all | 17.243128 | 0.047911 | 0.001079 | 0.046832 | 0.000000 | 0.000000 | 10.538949 | 0.000000 | 6.656267 | 0.277858 |
| 29_neural_all_neural_underharm4_no_guard | easy | 14.854704 | 0.008493 | 0.000097 | 0.008396 | 0.000000 | 0.000000 | 13.873351 | 0.000000 | 0.972860 | 0.057171 |
| 29_neural_all_neural_underharm4_no_guard | hard | 17.815047 | 0.036853 | 0.000802 | 0.036051 | 0.000000 | 0.000000 | 9.771687 | 0.000000 | 8.006507 | 0.206863 |
| 29_neural_all_neural_underharm4_source_zero_guard | all | 17.243128 | 0.018309 | 0.000432 | 0.017877 | 5.480552 | 0.000000 | 8.104149 | 0.000000 | 3.640117 | 0.106182 |
| 29_neural_all_neural_underharm4_source_zero_guard | easy | 14.854704 | 0.003094 | 0.000000 | 0.003094 | 4.738647 | 0.000000 | 9.692976 | 0.000000 | 0.419987 | 0.020828 |
| 29_neural_all_neural_underharm4_source_zero_guard | hard | 17.815047 | 0.020450 | 0.000361 | 0.020089 | 5.926679 | 0.000000 | 7.626610 | 0.000000 | 4.241308 | 0.114792 |
| 29_neural_easy_ridge_no_guard | all | 17.243128 | 0.344438 | 0.025553 | 0.318885 | 0.000000 | 0.000000 | 10.538949 | 0.107956 | 6.251785 | 1.997537 |
| 29_neural_easy_ridge_no_guard | easy | 14.854704 | 0.096910 | 0.008412 | 0.088498 | 0.000000 | 0.000000 | 13.873351 | 0.007170 | 0.877273 | 0.652384 |
| 29_neural_easy_ridge_no_guard | hard | 17.815047 | 0.380245 | 0.022671 | 0.357574 | 0.000000 | 0.000000 | 9.771687 | 0.112771 | 7.550344 | 2.134404 |
| 29_neural_easy_ridge_source_zero_guard | all | 17.243128 | 0.276155 | 0.010944 | 0.265211 | 5.480552 | 0.000000 | 8.104149 | 0.058419 | 3.323852 | 1.601538 |
| 29_neural_easy_ridge_source_zero_guard | easy | 14.854704 | 0.050446 | 0.005972 | 0.044474 | 4.738647 | 0.000000 | 9.692976 | 0.003317 | 0.369318 | 0.339594 |
| 29_neural_easy_ridge_source_zero_guard | hard | 17.815047 | 0.328884 | 0.009486 | 0.319398 | 5.926679 | 0.000000 | 7.626610 | 0.066734 | 3.866140 | 1.846103 |
| 29_neural_easy_neural_underharm4_no_guard | all | 17.243128 | 0.179370 | 0.013046 | 0.166325 | 0.000000 | 0.000000 | 10.538949 | 0.000000 | 6.524808 | 1.040242 |
| 29_neural_easy_neural_underharm4_no_guard | easy | 14.854704 | 0.098222 | 0.008131 | 0.090090 | 0.000000 | 0.000000 | 13.873351 | 0.000000 | 0.883131 | 0.661215 |
| 29_neural_easy_neural_underharm4_no_guard | hard | 17.815047 | 0.201168 | 0.012580 | 0.188589 | 0.000000 | 0.000000 | 9.771687 | 0.000000 | 7.842192 | 1.129204 |
| 29_neural_easy_neural_underharm4_source_zero_guard | all | 17.243128 | 0.179157 | 0.013046 | 0.166112 | 5.480552 | 0.000000 | 8.104149 | 0.000000 | 3.479269 | 1.039007 |
| 29_neural_easy_neural_underharm4_source_zero_guard | easy | 14.854704 | 0.097883 | 0.008131 | 0.089751 | 4.738647 | 0.000000 | 9.692976 | 0.000000 | 0.325198 | 0.658935 |
| 29_neural_easy_neural_underharm4_source_zero_guard | hard | 17.815047 | 0.201168 | 0.012580 | 0.188589 | 5.926679 | 0.000000 | 7.626610 | 0.000000 | 4.060590 | 1.129204 |
| 29_damping097_all_ridge_no_guard | all | 8.733475 | 1.981631 | 0.248614 | 1.733018 | 0.000000 | 0.000000 | 2.575986 | 0.020199 | 4.155658 | 22.690069 |
| 29_damping097_all_ridge_no_guard | easy | 6.147387 | 0.545406 | 0.434888 | 0.110518 | 0.000000 | 0.000000 | 3.284303 | 0.101380 | 2.216298 | 8.872161 |
| 29_damping097_all_ridge_no_guard | hard | 8.618661 | 1.984340 | 0.109315 | 1.875025 | 0.000000 | 0.000000 | 2.431585 | 0.005348 | 4.197388 | 23.023765 |
| 29_damping097_all_ridge_source_zero_guard | all | 8.733475 | 0.791954 | 0.061584 | 0.730370 | 3.028899 | 0.000000 | 1.806552 | 0.002187 | 3.103883 | 9.068026 |
| 29_damping097_all_ridge_source_zero_guard | easy | 6.147387 | 0.188504 | 0.163669 | 0.024835 | 1.899982 | 0.000000 | 2.379920 | 0.024271 | 1.654711 | 3.066415 |
| 29_damping097_all_ridge_source_zero_guard | hard | 8.618661 | 0.780062 | 0.027310 | 0.752752 | 2.860470 | 0.000000 | 1.726252 | 0.000341 | 3.251535 | 9.050848 |
| 29_damping097_all_neural_underharm4_no_guard | all | 8.733475 | 0.708410 | 0.094598 | 0.613811 | 0.000000 | 0.000000 | 2.575986 | 0.000000 | 5.449079 | 8.111432 |
| 29_damping097_all_neural_underharm4_no_guard | easy | 6.147387 | 0.016515 | 0.038090 | -0.021575 | 0.000000 | 0.000000 | 3.284303 | 0.000000 | 2.846570 | 0.268652 |
| 29_damping097_all_neural_underharm4_no_guard | hard | 8.618661 | 0.829079 | 0.063464 | 0.765615 | 0.000000 | 0.000000 | 2.431585 | 0.000000 | 5.357997 | 9.619579 |
| 29_damping097_all_neural_underharm4_source_zero_guard | all | 8.733475 | 0.268355 | 0.010301 | 0.258053 | 3.028899 | 0.000000 | 1.806552 | 0.000000 | 3.629669 | 3.072714 |
| 29_damping097_all_neural_underharm4_source_zero_guard | easy | 6.147387 | 0.011322 | 0.013939 | -0.002618 | 1.899982 | 0.000000 | 2.379920 | 0.000000 | 1.856164 | 0.184175 |
| 29_damping097_all_neural_underharm4_source_zero_guard | hard | 8.618661 | 0.325755 | 0.007941 | 0.317815 | 2.860470 | 0.000000 | 1.726252 | 0.000000 | 3.706183 | 3.779652 |
| 29_damping097_easy_ridge_no_guard | all | 8.733475 | 2.062244 | 0.427517 | 1.634727 | 0.000000 | 0.000000 | 2.575986 | 0.117669 | 3.977576 | 23.613099 |
| 29_damping097_easy_ridge_no_guard | easy | 6.147387 | 1.215680 | 0.663865 | 0.551815 | 0.000000 | 0.000000 | 3.284303 | 0.010887 | 1.636517 | 19.775557 |
| 29_damping097_easy_ridge_no_guard | hard | 8.618661 | 1.930120 | 0.185594 | 1.744526 | 0.000000 | 0.000000 | 2.431585 | 0.123439 | 4.133517 | 22.394662 |
| 29_damping097_easy_ridge_source_zero_guard | all | 8.733475 | 1.175660 | 0.089442 | 1.086219 | 3.028899 | 0.000000 | 1.806552 | 0.068681 | 2.653682 | 13.461543 |
| 29_damping097_easy_ridge_source_zero_guard | easy | 6.147387 | 0.674935 | 0.151047 | 0.523888 | 1.899982 | 0.000000 | 2.379920 | 0.002648 | 1.189903 | 10.979217 |
| 29_damping097_easy_ridge_source_zero_guard | hard | 8.618661 | 1.179314 | 0.059105 | 1.120209 | 2.860470 | 0.000000 | 1.726252 | 0.078080 | 2.774544 | 13.683267 |
| 29_damping097_easy_neural_underharm4_no_guard | all | 8.733475 | 2.213272 | 0.274465 | 1.938807 | 0.000000 | 0.000000 | 2.575986 | 0.000000 | 3.944216 | 25.342403 |
| 29_damping097_easy_neural_underharm4_no_guard | easy | 6.147387 | 1.550277 | 0.330849 | 1.219427 | 0.000000 | 0.000000 | 3.284303 | 0.000000 | 1.312808 | 25.218468 |
| 29_damping097_easy_neural_underharm4_no_guard | hard | 8.618661 | 2.127290 | 0.203163 | 1.924126 | 0.000000 | 0.000000 | 2.431585 | 0.000000 | 4.059786 | 24.682369 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | all | 8.733475 | 2.171445 | 0.272297 | 1.899148 | 3.028899 | 0.000000 | 1.806552 | 0.000000 | 1.726579 | 24.863468 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | easy | 6.147387 | 1.343666 | 0.321745 | 1.021921 | 1.899982 | 0.000000 | 2.379920 | 0.000000 | 0.523820 | 21.857508 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | hard | 8.618661 | 2.118371 | 0.201532 | 1.916839 | 2.860470 | 0.000000 | 1.726252 | 0.000000 | 1.913567 | 24.578890 |
| 43_neural_all_ridge_no_guard | all | 17.156434 | 0.836231 | 0.229996 | 0.606235 | 0.000000 | 0.000000 | 10.168943 | 0.014758 | 6.136502 | 4.874155 |
| 43_neural_all_ridge_no_guard | easy | 14.894507 | 0.239832 | 0.266274 | -0.026442 | 0.000000 | 0.000000 | 13.599191 | 0.091666 | 0.963819 | 1.610203 |
| 43_neural_all_ridge_no_guard | hard | 17.664870 | 0.898730 | 0.143512 | 0.755218 | 0.000000 | 0.000000 | 9.576157 | 0.001122 | 7.188861 | 5.087668 |
| 43_neural_all_ridge_source_zero_guard | all | 17.156434 | 0.294840 | 0.052843 | 0.241997 | 5.388225 | 0.000000 | 7.623848 | 0.004843 | 3.844677 | 1.718539 |
| 43_neural_all_ridge_source_zero_guard | easy | 14.894507 | 0.200121 | 0.134572 | 0.065549 | 4.756571 | 0.000000 | 9.224120 | 0.037267 | 0.676428 | 1.343590 |
| 43_neural_all_ridge_source_zero_guard | hard | 17.664870 | 0.291320 | 0.021686 | 0.269635 | 5.779673 | 0.000000 | 7.258735 | 0.001042 | 4.334099 | 1.649151 |
| 43_neural_all_neural_underharm4_no_guard | all | 17.156434 | 0.110659 | 0.007379 | 0.103280 | 0.000000 | 0.000000 | 10.168943 | 0.000000 | 6.876832 | 0.645001 |
| 43_neural_all_neural_underharm4_no_guard | easy | 14.894507 | 0.016591 | 0.000921 | 0.015670 | 0.000000 | 0.000000 | 13.599191 | 0.000000 | 1.278725 | 0.111388 |
| 43_neural_all_neural_underharm4_no_guard | hard | 17.664870 | 0.098180 | 0.005410 | 0.092770 | 0.000000 | 0.000000 | 9.576157 | 0.000000 | 7.990533 | 0.555791 |
| 43_neural_all_neural_underharm4_source_zero_guard | all | 17.156434 | 0.017256 | 0.000554 | 0.016701 | 5.388225 | 0.000000 | 7.623848 | 0.000000 | 4.127104 | 0.100579 |
| 43_neural_all_neural_underharm4_source_zero_guard | easy | 14.894507 | 0.006184 | 0.000000 | 0.006184 | 4.756571 | 0.000000 | 9.224120 | 0.000000 | 0.907632 | 0.041521 |
| 43_neural_all_neural_underharm4_source_zero_guard | hard | 17.664870 | 0.012210 | 0.000329 | 0.011881 | 5.779673 | 0.000000 | 7.258735 | 0.000000 | 4.614252 | 0.069119 |
| 43_neural_easy_ridge_no_guard | all | 17.156434 | 0.418107 | 0.051893 | 0.366214 | 0.000000 | 0.000000 | 10.168943 | 0.093436 | 6.475948 | 2.437028 |
| 43_neural_easy_ridge_no_guard | easy | 14.894507 | 0.083081 | 0.006373 | 0.076708 | 0.000000 | 0.000000 | 13.599191 | 0.007897 | 1.204338 | 0.557797 |
| 43_neural_easy_ridge_no_guard | hard | 17.664870 | 0.482362 | 0.053856 | 0.428506 | 0.000000 | 0.000000 | 9.576157 | 0.108609 | 7.497742 | 2.730627 |
| 43_neural_easy_ridge_source_zero_guard | all | 17.156434 | 0.358534 | 0.020797 | 0.337737 | 5.388225 | 0.000000 | 7.623848 | 0.051811 | 3.734015 | 2.089792 |
| 43_neural_easy_ridge_source_zero_guard | easy | 14.894507 | 0.073327 | 0.005513 | 0.067814 | 4.756571 | 0.000000 | 9.224120 | 0.004962 | 0.835528 | 0.492307 |
| 43_neural_easy_ridge_source_zero_guard | hard | 17.664870 | 0.433402 | 0.019804 | 0.413598 | 5.779673 | 0.000000 | 7.258735 | 0.061125 | 4.131935 | 2.453466 |
| 43_neural_easy_neural_underharm4_no_guard | all | 17.156434 | 0.476862 | 0.045828 | 0.431034 | 0.000000 | 0.000000 | 10.168943 | 0.000000 | 6.510629 | 2.779493 |
| 43_neural_easy_neural_underharm4_no_guard | easy | 14.894507 | 0.193554 | 0.120320 | 0.073234 | 0.000000 | 0.000000 | 13.599191 | 0.000000 | 1.101762 | 1.299498 |
| 43_neural_easy_neural_underharm4_no_guard | hard | 17.664870 | 0.524012 | 0.021241 | 0.502771 | 0.000000 | 0.000000 | 9.576157 | 0.000000 | 7.564701 | 2.966406 |
| 43_neural_easy_neural_underharm4_source_zero_guard | all | 17.156434 | 0.473355 | 0.045828 | 0.427527 | 5.388225 | 0.000000 | 7.623848 | 0.000000 | 3.671005 | 2.759053 |
| 43_neural_easy_neural_underharm4_source_zero_guard | easy | 14.894507 | 0.193554 | 0.120320 | 0.073234 | 4.756571 | 0.000000 | 9.224120 | 0.000000 | 0.720262 | 1.299498 |
| 43_neural_easy_neural_underharm4_source_zero_guard | hard | 17.664870 | 0.524012 | 0.021241 | 0.502771 | 5.779673 | 0.000000 | 7.258735 | 0.000000 | 4.102450 | 2.966406 |
| 43_damping097_all_ridge_no_guard | all | 8.733475 | 1.993437 | 0.250012 | 1.743425 | 0.000000 | 0.000000 | 2.868845 | 0.015431 | 3.855762 | 22.825242 |
| 43_damping097_all_ridge_no_guard | easy | 6.147387 | 0.554478 | 0.426518 | 0.127960 | 0.000000 | 0.000000 | 3.656014 | 0.072806 | 1.864089 | 9.019739 |
| 43_damping097_all_ridge_no_guard | hard | 8.618661 | 1.992804 | 0.110838 | 1.881967 | 0.000000 | 0.000000 | 2.615300 | 0.004156 | 4.006400 | 23.121974 |
| 43_damping097_all_ridge_source_zero_guard | all | 8.733475 | 0.795916 | 0.061584 | 0.734332 | 3.028899 | 0.000000 | 1.943584 | 0.002159 | 2.962916 | 9.113394 |
| 43_damping097_all_ridge_source_zero_guard | easy | 6.147387 | 0.205451 | 0.163061 | 0.042390 | 1.899982 | 0.000000 | 2.479409 | 0.022795 | 1.539751 | 3.342083 |
| 43_damping097_all_ridge_source_zero_guard | hard | 8.618661 | 0.781940 | 0.027041 | 0.754899 | 2.860470 | 0.000000 | 1.871943 | 0.000338 | 3.103969 | 9.072644 |
| 43_damping097_all_neural_underharm4_no_guard | all | 8.733475 | 0.698738 | 0.077841 | 0.620897 | 0.000000 | 0.000000 | 2.868845 | 0.000000 | 5.165891 | 8.000691 |
| 43_damping097_all_neural_underharm4_no_guard | easy | 6.147387 | 0.042071 | 0.045618 | -0.003547 | 0.000000 | 0.000000 | 3.656014 | 0.000000 | 2.449302 | 0.684378 |
| 43_damping097_all_neural_underharm4_no_guard | hard | 8.618661 | 0.793040 | 0.052122 | 0.740917 | 0.000000 | 0.000000 | 2.615300 | 0.000000 | 5.210321 | 9.201426 |
| 43_damping097_all_neural_underharm4_source_zero_guard | all | 8.733475 | 0.369512 | 0.013250 | 0.356262 | 3.028899 | 0.000000 | 1.943584 | 0.000000 | 3.391479 | 4.230987 |
| 43_damping097_all_neural_underharm4_source_zero_guard | easy | 6.147387 | 0.032553 | 0.021922 | 0.010632 | 1.899982 | 0.000000 | 2.479409 | 0.000000 | 1.735443 | 0.529548 |
| 43_damping097_all_neural_underharm4_source_zero_guard | hard | 8.618661 | 0.416508 | 0.005223 | 0.411285 | 2.860470 | 0.000000 | 1.871943 | 0.000000 | 3.469739 | 4.832634 |
| 43_damping097_easy_ridge_no_guard | all | 8.733475 | 1.969588 | 0.421387 | 1.548200 | 0.000000 | 0.000000 | 2.868845 | 0.117102 | 3.777940 | 22.552166 |
| 43_damping097_easy_ridge_no_guard | easy | 6.147387 | 1.148558 | 0.630891 | 0.517667 | 0.000000 | 0.000000 | 3.656014 | 0.010859 | 1.331956 | 18.683685 |
| 43_damping097_easy_ridge_no_guard | hard | 8.618661 | 1.920031 | 0.187636 | 1.732395 | 0.000000 | 0.000000 | 2.615300 | 0.125250 | 3.958079 | 22.277605 |
| 43_damping097_easy_ridge_source_zero_guard | all | 8.733475 | 1.169581 | 0.083820 | 1.085762 | 3.028899 | 0.000000 | 1.943584 | 0.070823 | 2.520587 | 13.391935 |
| 43_damping097_easy_ridge_source_zero_guard | easy | 6.147387 | 0.734992 | 0.163891 | 0.571100 | 1.899982 | 0.000000 | 2.479409 | 0.002648 | 1.030357 | 11.956162 |
| 43_damping097_easy_ridge_source_zero_guard | hard | 8.618661 | 1.168423 | 0.050853 | 1.117570 | 2.860470 | 0.000000 | 1.871943 | 0.080474 | 2.637350 | 13.556902 |
| 43_damping097_easy_neural_underharm4_no_guard | all | 8.733475 | 2.136321 | 0.238839 | 1.897481 | 0.000000 | 0.000000 | 2.868845 | 0.000000 | 3.728309 | 24.461291 |
| 43_damping097_easy_neural_underharm4_no_guard | easy | 6.147387 | 1.545341 | 0.306863 | 1.238478 | 0.000000 | 0.000000 | 3.656014 | 0.000000 | 0.946033 | 25.138173 |
| 43_damping097_easy_neural_underharm4_no_guard | hard | 8.618661 | 2.052299 | 0.173855 | 1.878444 | 0.000000 | 0.000000 | 2.615300 | 0.000000 | 3.951061 | 23.812273 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | all | 8.733475 | 2.103663 | 0.237547 | 1.866116 | 3.028899 | 0.000000 | 1.943584 | 0.000000 | 1.657328 | 24.087355 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | easy | 6.147387 | 1.426372 | 0.304287 | 1.122085 | 1.899982 | 0.000000 | 2.479409 | 0.000000 | 0.341625 | 23.202893 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | hard | 8.618661 | 2.040889 | 0.172646 | 1.868243 | 2.860470 | 0.000000 | 1.871943 | 0.000000 | 1.845359 | 23.679883 |

## Producer Population Comparison

Positive means the eight-locality producer beats the named four-locality producer on the same held rows.
Both ordered controls are retained; no better-of-two selection. This is not isolated randomized sample-size evidence.
| Seed | Four-locality control | Eight-locality gain (%) | Conditional locality CI |
|---|---|---:|---|
| 17 | single_lower_fold | -1.843165 | [-3.400304, -0.416987] |
| 17 | single_upper_fold | 0.525346 | [-2.331167, 2.742026] |
| 29 | single_lower_fold | -2.344727 | [-3.663156, -1.074960] |
| 29 | single_upper_fold | 0.087852 | [-2.629008, 2.049170] |
| 43 | single_lower_fold | -1.426592 | [-2.791855, -0.206975] |
| 43 | single_upper_fold | 0.127657 | [-2.570974, 2.040601] |

## Fixed Risk Bands

Counts retain unsupported future rows; precision denominators use supported rows only. Not a threshold search.
| Policy | Risk band | Indexed | Known | Beneficial | Harmful | Positive utility | Positive-utility beneficial | Switched (all rows) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 17_neural_all_ridge_no_guard | zero_mass | 6709 | 6572 | 5146 | 1426 | 1115 | 880 | 0 |
| 17_neural_all_ridge_no_guard | le_002 | 17010 | 16624 | 10610 | 6014 | 12394 | 7174 | 12700 |
| 17_neural_all_ridge_no_guard | 002_to_01 | 143322 | 140426 | 105229 | 35197 | 61287 | 44828 | 0 |
| 17_neural_all_ridge_no_guard | 01_to_05 | 151221 | 147612 | 97575 | 50037 | 36728 | 28224 | 0 |
| 17_neural_all_ridge_no_guard | 05_to_1 | 434 | 425 | 255 | 170 | 26 | 20 | 0 |
| 17_neural_all_ridge_no_guard | gt_1 | 273 | 263 | 175 | 88 | 16 | 13 | 0 |
| 17_neural_all_ridge_source_zero_guard | zero_mass | 6709 | 6572 | 5146 | 1426 | 1115 | 880 | 0 |
| 17_neural_all_ridge_source_zero_guard | le_002 | 17010 | 16624 | 10610 | 6014 | 12394 | 7174 | 2069 |
| 17_neural_all_ridge_source_zero_guard | 002_to_01 | 143322 | 140426 | 105229 | 35197 | 61287 | 44828 | 0 |
| 17_neural_all_ridge_source_zero_guard | 01_to_05 | 151221 | 147612 | 97575 | 50037 | 36728 | 28224 | 0 |
| 17_neural_all_ridge_source_zero_guard | 05_to_1 | 434 | 425 | 255 | 170 | 26 | 20 | 0 |
| 17_neural_all_ridge_source_zero_guard | gt_1 | 273 | 263 | 175 | 88 | 16 | 13 | 0 |
| 17_neural_all_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_neural_all_neural_underharm4_no_guard | le_002 | 1012 | 992 | 745 | 247 | 915 | 671 | 935 |
| 17_neural_all_neural_underharm4_no_guard | 002_to_01 | 13341 | 13035 | 8533 | 4502 | 12270 | 7858 | 0 |
| 17_neural_all_neural_underharm4_no_guard | 01_to_05 | 246805 | 241320 | 167117 | 74203 | 92418 | 67613 | 0 |
| 17_neural_all_neural_underharm4_no_guard | 05_to_1 | 48760 | 47672 | 35449 | 12223 | 5578 | 4673 | 0 |
| 17_neural_all_neural_underharm4_no_guard | gt_1 | 9051 | 8903 | 7146 | 1757 | 385 | 324 | 0 |
| 17_neural_all_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_neural_all_neural_underharm4_source_zero_guard | le_002 | 1012 | 992 | 745 | 247 | 915 | 671 | 259 |
| 17_neural_all_neural_underharm4_source_zero_guard | 002_to_01 | 13341 | 13035 | 8533 | 4502 | 12270 | 7858 | 0 |
| 17_neural_all_neural_underharm4_source_zero_guard | 01_to_05 | 246805 | 241320 | 167117 | 74203 | 92418 | 67613 | 0 |
| 17_neural_all_neural_underharm4_source_zero_guard | 05_to_1 | 48760 | 47672 | 35449 | 12223 | 5578 | 4673 | 0 |
| 17_neural_all_neural_underharm4_source_zero_guard | gt_1 | 9051 | 8903 | 7146 | 1757 | 385 | 324 | 0 |
| 17_neural_easy_ridge_no_guard | zero_mass | 2216 | 2025 | 1146 | 879 | 500 | 369 | 0 |
| 17_neural_easy_ridge_no_guard | le_002 | 24233 | 23696 | 17878 | 5818 | 8570 | 7153 | 8692 |
| 17_neural_easy_ridge_no_guard | 002_to_01 | 94753 | 93014 | 71743 | 21271 | 37160 | 29093 | 0 |
| 17_neural_easy_ridge_no_guard | 01_to_05 | 164305 | 160881 | 107326 | 53555 | 59959 | 40503 | 0 |
| 17_neural_easy_ridge_no_guard | 05_to_1 | 27746 | 26960 | 17496 | 9464 | 4760 | 3559 | 0 |
| 17_neural_easy_ridge_no_guard | gt_1 | 5716 | 5346 | 3401 | 1945 | 617 | 462 | 0 |
| 17_neural_easy_ridge_source_zero_guard | zero_mass | 2216 | 2025 | 1146 | 879 | 500 | 369 | 0 |
| 17_neural_easy_ridge_source_zero_guard | le_002 | 24233 | 23696 | 17878 | 5818 | 8570 | 7153 | 3113 |
| 17_neural_easy_ridge_source_zero_guard | 002_to_01 | 94753 | 93014 | 71743 | 21271 | 37160 | 29093 | 0 |
| 17_neural_easy_ridge_source_zero_guard | 01_to_05 | 164305 | 160881 | 107326 | 53555 | 59959 | 40503 | 0 |
| 17_neural_easy_ridge_source_zero_guard | 05_to_1 | 27746 | 26960 | 17496 | 9464 | 4760 | 3559 | 0 |
| 17_neural_easy_ridge_source_zero_guard | gt_1 | 5716 | 5346 | 3401 | 1945 | 617 | 462 | 0 |
| 17_neural_easy_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_neural_easy_neural_underharm4_no_guard | le_002 | 9234 | 9058 | 7350 | 1708 | 3892 | 3290 | 3973 |
| 17_neural_easy_neural_underharm4_no_guard | 002_to_01 | 37062 | 36547 | 31566 | 4981 | 14777 | 13065 | 0 |
| 17_neural_easy_neural_underharm4_no_guard | 01_to_05 | 96175 | 94551 | 74475 | 20076 | 34560 | 28349 | 0 |
| 17_neural_easy_neural_underharm4_no_guard | 05_to_1 | 63476 | 62160 | 43132 | 19028 | 18958 | 13585 | 0 |
| 17_neural_easy_neural_underharm4_no_guard | gt_1 | 113022 | 109606 | 62467 | 47139 | 39379 | 22850 | 0 |
| 17_neural_easy_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_neural_easy_neural_underharm4_source_zero_guard | le_002 | 9234 | 9058 | 7350 | 1708 | 3892 | 3290 | 1755 |
| 17_neural_easy_neural_underharm4_source_zero_guard | 002_to_01 | 37062 | 36547 | 31566 | 4981 | 14777 | 13065 | 0 |
| 17_neural_easy_neural_underharm4_source_zero_guard | 01_to_05 | 96175 | 94551 | 74475 | 20076 | 34560 | 28349 | 0 |
| 17_neural_easy_neural_underharm4_source_zero_guard | 05_to_1 | 63476 | 62160 | 43132 | 19028 | 18958 | 13585 | 0 |
| 17_neural_easy_neural_underharm4_source_zero_guard | gt_1 | 113022 | 109606 | 62467 | 47139 | 39379 | 22850 | 0 |
| 17_damping097_all_ridge_no_guard | zero_mass | 7811 | 7638 | 6197 | 1316 | 2051 | 1615 | 0 |
| 17_damping097_all_ridge_no_guard | le_002 | 118053 | 116227 | 98907 | 15972 | 97345 | 82573 | 98894 |
| 17_damping097_all_ridge_no_guard | 002_to_01 | 187411 | 182550 | 127121 | 51531 | 85615 | 66452 | 0 |
| 17_damping097_all_ridge_no_guard | 01_to_05 | 5277 | 5101 | 2672 | 2301 | 517 | 407 | 0 |
| 17_damping097_all_ridge_no_guard | 05_to_1 | 241 | 236 | 172 | 61 | 38 | 30 | 0 |
| 17_damping097_all_ridge_no_guard | gt_1 | 176 | 170 | 127 | 39 | 40 | 30 | 0 |
| 17_damping097_all_ridge_source_zero_guard | zero_mass | 7811 | 7638 | 6197 | 1316 | 2051 | 1615 | 0 |
| 17_damping097_all_ridge_source_zero_guard | le_002 | 118053 | 116227 | 98907 | 15972 | 97345 | 82573 | 23686 |
| 17_damping097_all_ridge_source_zero_guard | 002_to_01 | 187411 | 182550 | 127121 | 51531 | 85615 | 66452 | 0 |
| 17_damping097_all_ridge_source_zero_guard | 01_to_05 | 5277 | 5101 | 2672 | 2301 | 517 | 407 | 0 |
| 17_damping097_all_ridge_source_zero_guard | 05_to_1 | 241 | 236 | 172 | 61 | 38 | 30 | 0 |
| 17_damping097_all_ridge_source_zero_guard | gt_1 | 176 | 170 | 127 | 39 | 40 | 30 | 0 |
| 17_damping097_all_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_damping097_all_neural_underharm4_no_guard | le_002 | 17700 | 17107 | 10229 | 6290 | 13085 | 9040 | 13549 |
| 17_damping097_all_neural_underharm4_no_guard | 002_to_01 | 185104 | 180798 | 138138 | 39315 | 127554 | 104071 | 0 |
| 17_damping097_all_neural_underharm4_no_guard | 01_to_05 | 108954 | 106883 | 80842 | 24524 | 43732 | 36968 | 0 |
| 17_damping097_all_neural_underharm4_no_guard | 05_to_1 | 5459 | 5401 | 4531 | 830 | 984 | 808 | 0 |
| 17_damping097_all_neural_underharm4_no_guard | gt_1 | 1752 | 1733 | 1456 | 261 | 251 | 220 | 0 |
| 17_damping097_all_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_damping097_all_neural_underharm4_source_zero_guard | le_002 | 17700 | 17107 | 10229 | 6290 | 13085 | 9040 | 2616 |
| 17_damping097_all_neural_underharm4_source_zero_guard | 002_to_01 | 185104 | 180798 | 138138 | 39315 | 127554 | 104071 | 0 |
| 17_damping097_all_neural_underharm4_source_zero_guard | 01_to_05 | 108954 | 106883 | 80842 | 24524 | 43732 | 36968 | 0 |
| 17_damping097_all_neural_underharm4_source_zero_guard | 05_to_1 | 5459 | 5401 | 4531 | 830 | 984 | 808 | 0 |
| 17_damping097_all_neural_underharm4_source_zero_guard | gt_1 | 1752 | 1733 | 1456 | 261 | 251 | 220 | 0 |
| 17_damping097_easy_ridge_no_guard | zero_mass | 1243 | 1123 | 807 | 242 | 684 | 515 | 0 |
| 17_damping097_easy_ridge_no_guard | le_002 | 169110 | 166224 | 133761 | 30315 | 106525 | 90260 | 108251 |
| 17_damping097_easy_ridge_no_guard | 002_to_01 | 122405 | 119402 | 85192 | 31767 | 66605 | 50956 | 0 |
| 17_damping097_easy_ridge_no_guard | 01_to_05 | 24757 | 23856 | 14806 | 8320 | 11334 | 9034 | 0 |
| 17_damping097_easy_ridge_no_guard | 05_to_1 | 1096 | 987 | 432 | 482 | 309 | 238 | 0 |
| 17_damping097_easy_ridge_no_guard | gt_1 | 358 | 330 | 198 | 94 | 149 | 104 | 0 |
| 17_damping097_easy_ridge_source_zero_guard | zero_mass | 1243 | 1123 | 807 | 242 | 684 | 515 | 0 |
| 17_damping097_easy_ridge_source_zero_guard | le_002 | 169110 | 166224 | 133761 | 30315 | 106525 | 90260 | 35951 |
| 17_damping097_easy_ridge_source_zero_guard | 002_to_01 | 122405 | 119402 | 85192 | 31767 | 66605 | 50956 | 0 |
| 17_damping097_easy_ridge_source_zero_guard | 01_to_05 | 24757 | 23856 | 14806 | 8320 | 11334 | 9034 | 0 |
| 17_damping097_easy_ridge_source_zero_guard | 05_to_1 | 1096 | 987 | 432 | 482 | 309 | 238 | 0 |
| 17_damping097_easy_ridge_source_zero_guard | gt_1 | 358 | 330 | 198 | 94 | 149 | 104 | 0 |
| 17_damping097_easy_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_damping097_easy_neural_underharm4_no_guard | le_002 | 120639 | 118674 | 93926 | 23173 | 66080 | 56018 | 67075 |
| 17_damping097_easy_neural_underharm4_no_guard | 002_to_01 | 116761 | 114786 | 96024 | 17329 | 74013 | 64011 | 0 |
| 17_damping097_easy_neural_underharm4_no_guard | 01_to_05 | 40455 | 39016 | 23719 | 14150 | 21405 | 15077 | 0 |
| 17_damping097_easy_neural_underharm4_no_guard | 05_to_1 | 9734 | 9323 | 4993 | 3980 | 5053 | 3425 | 0 |
| 17_damping097_easy_neural_underharm4_no_guard | gt_1 | 31380 | 30123 | 16534 | 12588 | 19055 | 12576 | 0 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | le_002 | 120639 | 118674 | 93926 | 23173 | 66080 | 56018 | 50581 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | 002_to_01 | 116761 | 114786 | 96024 | 17329 | 74013 | 64011 | 0 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | 01_to_05 | 40455 | 39016 | 23719 | 14150 | 21405 | 15077 | 0 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | 05_to_1 | 9734 | 9323 | 4993 | 3980 | 5053 | 3425 | 0 |
| 17_damping097_easy_neural_underharm4_source_zero_guard | gt_1 | 31380 | 30123 | 16534 | 12588 | 19055 | 12576 | 0 |
| 29_neural_all_ridge_no_guard | zero_mass | 6829 | 6683 | 5261 | 1422 | 445 | 365 | 0 |
| 29_neural_all_ridge_no_guard | le_002 | 13007 | 12711 | 7935 | 4776 | 8443 | 4602 | 8649 |
| 29_neural_all_ridge_no_guard | 002_to_01 | 138889 | 136067 | 102981 | 33086 | 33077 | 21483 | 0 |
| 29_neural_all_ridge_no_guard | 01_to_05 | 159133 | 155375 | 101789 | 53586 | 21173 | 17038 | 0 |
| 29_neural_all_ridge_no_guard | 05_to_1 | 674 | 659 | 405 | 254 | 14 | 12 | 0 |
| 29_neural_all_ridge_no_guard | gt_1 | 437 | 427 | 289 | 138 | 12 | 11 | 0 |
| 29_neural_all_ridge_source_zero_guard | zero_mass | 6829 | 6683 | 5261 | 1422 | 445 | 365 | 0 |
| 29_neural_all_ridge_source_zero_guard | le_002 | 13007 | 12711 | 7935 | 4776 | 8443 | 4602 | 1346 |
| 29_neural_all_ridge_source_zero_guard | 002_to_01 | 138889 | 136067 | 102981 | 33086 | 33077 | 21483 | 0 |
| 29_neural_all_ridge_source_zero_guard | 01_to_05 | 159133 | 155375 | 101789 | 53586 | 21173 | 17038 | 0 |
| 29_neural_all_ridge_source_zero_guard | 05_to_1 | 674 | 659 | 405 | 254 | 14 | 12 | 0 |
| 29_neural_all_ridge_source_zero_guard | gt_1 | 437 | 427 | 289 | 138 | 12 | 11 | 0 |
| 29_neural_all_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_neural_all_neural_underharm4_no_guard | le_002 | 343 | 335 | 299 | 36 | 288 | 254 | 295 |
| 29_neural_all_neural_underharm4_no_guard | 002_to_01 | 9875 | 9633 | 5952 | 3681 | 9045 | 5609 | 0 |
| 29_neural_all_neural_underharm4_no_guard | 01_to_05 | 231429 | 226071 | 155911 | 70160 | 51034 | 35261 | 0 |
| 29_neural_all_neural_underharm4_no_guard | 05_to_1 | 64348 | 63108 | 46412 | 16696 | 2448 | 2087 | 0 |
| 29_neural_all_neural_underharm4_no_guard | gt_1 | 12974 | 12775 | 10086 | 2689 | 349 | 300 | 0 |
| 29_neural_all_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_neural_all_neural_underharm4_source_zero_guard | le_002 | 343 | 335 | 299 | 36 | 288 | 254 | 191 |
| 29_neural_all_neural_underharm4_source_zero_guard | 002_to_01 | 9875 | 9633 | 5952 | 3681 | 9045 | 5609 | 0 |
| 29_neural_all_neural_underharm4_source_zero_guard | 01_to_05 | 231429 | 226071 | 155911 | 70160 | 51034 | 35261 | 0 |
| 29_neural_all_neural_underharm4_source_zero_guard | 05_to_1 | 64348 | 63108 | 46412 | 16696 | 2448 | 2087 | 0 |
| 29_neural_all_neural_underharm4_source_zero_guard | gt_1 | 12974 | 12775 | 10086 | 2689 | 349 | 300 | 0 |
| 29_neural_easy_ridge_no_guard | zero_mass | 2163 | 1971 | 1178 | 793 | 383 | 307 | 0 |
| 29_neural_easy_ridge_no_guard | le_002 | 19578 | 19124 | 14384 | 4740 | 3484 | 2910 | 3534 |
| 29_neural_easy_ridge_no_guard | 002_to_01 | 92591 | 90896 | 70525 | 20371 | 14935 | 11108 | 0 |
| 29_neural_easy_ridge_no_guard | 01_to_05 | 169135 | 165633 | 110567 | 55066 | 41057 | 26681 | 0 |
| 29_neural_easy_ridge_no_guard | 05_to_1 | 29733 | 28907 | 18644 | 10263 | 2747 | 2077 | 0 |
| 29_neural_easy_ridge_no_guard | gt_1 | 5769 | 5391 | 3362 | 2029 | 558 | 428 | 0 |
| 29_neural_easy_ridge_source_zero_guard | zero_mass | 2163 | 1971 | 1178 | 793 | 383 | 307 | 0 |
| 29_neural_easy_ridge_source_zero_guard | le_002 | 19578 | 19124 | 14384 | 4740 | 3484 | 2910 | 2414 |
| 29_neural_easy_ridge_source_zero_guard | 002_to_01 | 92591 | 90896 | 70525 | 20371 | 14935 | 11108 | 0 |
| 29_neural_easy_ridge_source_zero_guard | 01_to_05 | 169135 | 165633 | 110567 | 55066 | 41057 | 26681 | 0 |
| 29_neural_easy_ridge_source_zero_guard | 05_to_1 | 29733 | 28907 | 18644 | 10263 | 2747 | 2077 | 0 |
| 29_neural_easy_ridge_source_zero_guard | gt_1 | 5769 | 5391 | 3362 | 2029 | 558 | 428 | 0 |
| 29_neural_easy_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_neural_easy_neural_underharm4_no_guard | le_002 | 3155 | 3065 | 2199 | 866 | 911 | 781 | 939 |
| 29_neural_easy_neural_underharm4_no_guard | 002_to_01 | 27769 | 27420 | 24055 | 3365 | 2722 | 2360 | 0 |
| 29_neural_easy_neural_underharm4_no_guard | 01_to_05 | 117143 | 115149 | 92200 | 22949 | 19144 | 14844 | 0 |
| 29_neural_easy_neural_underharm4_no_guard | 05_to_1 | 59695 | 58467 | 39792 | 18675 | 11870 | 8774 | 0 |
| 29_neural_easy_neural_underharm4_no_guard | gt_1 | 111207 | 107821 | 60414 | 47407 | 28517 | 16752 | 0 |
| 29_neural_easy_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_neural_easy_neural_underharm4_source_zero_guard | le_002 | 3155 | 3065 | 2199 | 866 | 911 | 781 | 929 |
| 29_neural_easy_neural_underharm4_source_zero_guard | 002_to_01 | 27769 | 27420 | 24055 | 3365 | 2722 | 2360 | 0 |
| 29_neural_easy_neural_underharm4_source_zero_guard | 01_to_05 | 117143 | 115149 | 92200 | 22949 | 19144 | 14844 | 0 |
| 29_neural_easy_neural_underharm4_source_zero_guard | 05_to_1 | 59695 | 58467 | 39792 | 18675 | 11870 | 8774 | 0 |
| 29_neural_easy_neural_underharm4_source_zero_guard | gt_1 | 111207 | 107821 | 60414 | 47407 | 28517 | 16752 | 0 |
| 29_damping097_all_ridge_no_guard | zero_mass | 7811 | 7638 | 6197 | 1316 | 3859 | 3116 | 0 |
| 29_damping097_all_ridge_no_guard | le_002 | 118053 | 116227 | 98907 | 15972 | 80732 | 67282 | 82099 |
| 29_damping097_all_ridge_no_guard | 002_to_01 | 187411 | 182550 | 127121 | 51531 | 85565 | 65576 | 0 |
| 29_damping097_all_ridge_no_guard | 01_to_05 | 5277 | 5101 | 2672 | 2301 | 912 | 698 | 0 |
| 29_damping097_all_ridge_no_guard | 05_to_1 | 241 | 236 | 172 | 61 | 75 | 59 | 0 |
| 29_damping097_all_ridge_no_guard | gt_1 | 176 | 170 | 127 | 39 | 60 | 47 | 0 |
| 29_damping097_all_ridge_source_zero_guard | zero_mass | 7811 | 7638 | 6197 | 1316 | 3859 | 3116 | 0 |
| 29_damping097_all_ridge_source_zero_guard | le_002 | 118053 | 116227 | 98907 | 15972 | 80732 | 67282 | 22623 |
| 29_damping097_all_ridge_source_zero_guard | 002_to_01 | 187411 | 182550 | 127121 | 51531 | 85565 | 65576 | 0 |
| 29_damping097_all_ridge_source_zero_guard | 01_to_05 | 5277 | 5101 | 2672 | 2301 | 912 | 698 | 0 |
| 29_damping097_all_ridge_source_zero_guard | 05_to_1 | 241 | 236 | 172 | 61 | 75 | 59 | 0 |
| 29_damping097_all_ridge_source_zero_guard | gt_1 | 176 | 170 | 127 | 39 | 60 | 47 | 0 |
| 29_damping097_all_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_damping097_all_neural_underharm4_no_guard | le_002 | 12760 | 12384 | 8582 | 3429 | 11331 | 8265 | 11658 |
| 29_damping097_all_neural_underharm4_no_guard | 002_to_01 | 182022 | 177697 | 133454 | 40838 | 121960 | 97140 | 0 |
| 29_damping097_all_neural_underharm4_no_guard | 01_to_05 | 118030 | 115750 | 88073 | 25999 | 37151 | 30728 | 0 |
| 29_damping097_all_neural_underharm4_no_guard | 05_to_1 | 4877 | 4827 | 4013 | 775 | 573 | 485 | 0 |
| 29_damping097_all_neural_underharm4_no_guard | gt_1 | 1280 | 1264 | 1074 | 179 | 188 | 160 | 0 |
| 29_damping097_all_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_damping097_all_neural_underharm4_source_zero_guard | le_002 | 12760 | 12384 | 8582 | 3429 | 11331 | 8265 | 4515 |
| 29_damping097_all_neural_underharm4_source_zero_guard | 002_to_01 | 182022 | 177697 | 133454 | 40838 | 121960 | 97140 | 0 |
| 29_damping097_all_neural_underharm4_source_zero_guard | 01_to_05 | 118030 | 115750 | 88073 | 25999 | 37151 | 30728 | 0 |
| 29_damping097_all_neural_underharm4_source_zero_guard | 05_to_1 | 4877 | 4827 | 4013 | 775 | 573 | 485 | 0 |
| 29_damping097_all_neural_underharm4_source_zero_guard | gt_1 | 1280 | 1264 | 1074 | 179 | 188 | 160 | 0 |
| 29_damping097_easy_ridge_no_guard | zero_mass | 1243 | 1123 | 807 | 242 | 648 | 498 | 0 |
| 29_damping097_easy_ridge_no_guard | le_002 | 169110 | 166224 | 133761 | 30315 | 91457 | 76178 | 93004 |
| 29_damping097_easy_ridge_no_guard | 002_to_01 | 122405 | 119402 | 85192 | 31767 | 68446 | 51691 | 0 |
| 29_damping097_easy_ridge_no_guard | 01_to_05 | 24757 | 23856 | 14806 | 8320 | 10183 | 8069 | 0 |
| 29_damping097_easy_ridge_no_guard | 05_to_1 | 1096 | 987 | 432 | 482 | 317 | 239 | 0 |
| 29_damping097_easy_ridge_no_guard | gt_1 | 358 | 330 | 198 | 94 | 152 | 103 | 0 |
| 29_damping097_easy_ridge_source_zero_guard | zero_mass | 1243 | 1123 | 807 | 242 | 648 | 498 | 0 |
| 29_damping097_easy_ridge_source_zero_guard | le_002 | 169110 | 166224 | 133761 | 30315 | 91457 | 76178 | 34694 |
| 29_damping097_easy_ridge_source_zero_guard | 002_to_01 | 122405 | 119402 | 85192 | 31767 | 68446 | 51691 | 0 |
| 29_damping097_easy_ridge_source_zero_guard | 01_to_05 | 24757 | 23856 | 14806 | 8320 | 10183 | 8069 | 0 |
| 29_damping097_easy_ridge_source_zero_guard | 05_to_1 | 1096 | 987 | 432 | 482 | 317 | 239 | 0 |
| 29_damping097_easy_ridge_source_zero_guard | gt_1 | 358 | 330 | 198 | 94 | 152 | 103 | 0 |
| 29_damping097_easy_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_damping097_easy_neural_underharm4_no_guard | le_002 | 134485 | 132370 | 106226 | 24454 | 69024 | 58868 | 70065 |
| 29_damping097_easy_neural_underharm4_no_guard | 002_to_01 | 107911 | 105874 | 86736 | 17619 | 59864 | 49550 | 0 |
| 29_damping097_easy_neural_underharm4_no_guard | 01_to_05 | 40982 | 39542 | 24351 | 14075 | 22100 | 15437 | 0 |
| 29_damping097_easy_neural_underharm4_no_guard | 05_to_1 | 9880 | 9502 | 5066 | 4133 | 5553 | 3662 | 0 |
| 29_damping097_easy_neural_underharm4_no_guard | gt_1 | 25711 | 24634 | 12817 | 10939 | 14662 | 9261 | 0 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | le_002 | 134485 | 132370 | 106226 | 24454 | 69024 | 58868 | 49122 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | 002_to_01 | 107911 | 105874 | 86736 | 17619 | 59864 | 49550 | 0 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | 01_to_05 | 40982 | 39542 | 24351 | 14075 | 22100 | 15437 | 0 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | 05_to_1 | 9880 | 9502 | 5066 | 4133 | 5553 | 3662 | 0 |
| 29_damping097_easy_neural_underharm4_source_zero_guard | gt_1 | 25711 | 24634 | 12817 | 10939 | 14662 | 9261 | 0 |
| 43_neural_all_ridge_no_guard | zero_mass | 6981 | 6829 | 5412 | 1417 | 624 | 525 | 0 |
| 43_neural_all_ridge_no_guard | le_002 | 12818 | 12530 | 7934 | 4596 | 7937 | 4377 | 8133 |
| 43_neural_all_ridge_no_guard | 002_to_01 | 144146 | 141091 | 105821 | 35270 | 31839 | 20642 | 0 |
| 43_neural_all_ridge_no_guard | 01_to_05 | 154031 | 150500 | 99465 | 51035 | 26324 | 20624 | 0 |
| 43_neural_all_ridge_no_guard | 05_to_1 | 621 | 608 | 384 | 224 | 17 | 15 | 0 |
| 43_neural_all_ridge_no_guard | gt_1 | 372 | 364 | 238 | 126 | 10 | 9 | 0 |
| 43_neural_all_ridge_source_zero_guard | zero_mass | 6981 | 6829 | 5412 | 1417 | 624 | 525 | 0 |
| 43_neural_all_ridge_source_zero_guard | le_002 | 12818 | 12530 | 7934 | 4596 | 7937 | 4377 | 1675 |
| 43_neural_all_ridge_source_zero_guard | 002_to_01 | 144146 | 141091 | 105821 | 35270 | 31839 | 20642 | 0 |
| 43_neural_all_ridge_source_zero_guard | 01_to_05 | 154031 | 150500 | 99465 | 51035 | 26324 | 20624 | 0 |
| 43_neural_all_ridge_source_zero_guard | 05_to_1 | 621 | 608 | 384 | 224 | 17 | 15 | 0 |
| 43_neural_all_ridge_source_zero_guard | gt_1 | 372 | 364 | 238 | 126 | 10 | 9 | 0 |
| 43_neural_all_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_neural_all_neural_underharm4_no_guard | le_002 | 735 | 721 | 566 | 155 | 609 | 461 | 618 |
| 43_neural_all_neural_underharm4_no_guard | 002_to_01 | 15818 | 15451 | 9804 | 5647 | 14011 | 8613 | 0 |
| 43_neural_all_neural_underharm4_no_guard | 01_to_05 | 250852 | 245177 | 170365 | 74812 | 50871 | 36036 | 0 |
| 43_neural_all_neural_underharm4_no_guard | 05_to_1 | 43935 | 43049 | 32303 | 10746 | 1221 | 1048 | 0 |
| 43_neural_all_neural_underharm4_no_guard | gt_1 | 7629 | 7524 | 6216 | 1308 | 39 | 34 | 0 |
| 43_neural_all_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_neural_all_neural_underharm4_source_zero_guard | le_002 | 735 | 721 | 566 | 155 | 609 | 461 | 204 |
| 43_neural_all_neural_underharm4_source_zero_guard | 002_to_01 | 15818 | 15451 | 9804 | 5647 | 14011 | 8613 | 0 |
| 43_neural_all_neural_underharm4_source_zero_guard | 01_to_05 | 250852 | 245177 | 170365 | 74812 | 50871 | 36036 | 0 |
| 43_neural_all_neural_underharm4_source_zero_guard | 05_to_1 | 43935 | 43049 | 32303 | 10746 | 1221 | 1048 | 0 |
| 43_neural_all_neural_underharm4_source_zero_guard | gt_1 | 7629 | 7524 | 6216 | 1308 | 39 | 34 | 0 |
| 43_neural_easy_ridge_no_guard | zero_mass | 2244 | 2034 | 1200 | 834 | 360 | 270 | 0 |
| 43_neural_easy_ridge_no_guard | le_002 | 17508 | 17101 | 12869 | 4232 | 3695 | 3051 | 3746 |
| 43_neural_easy_ridge_no_guard | 002_to_01 | 97972 | 96165 | 74382 | 21783 | 14203 | 10328 | 0 |
| 43_neural_easy_ridge_no_guard | 01_to_05 | 166925 | 163505 | 109793 | 53712 | 44324 | 29433 | 0 |
| 43_neural_easy_ridge_no_guard | 05_to_1 | 28271 | 27466 | 17444 | 10022 | 3597 | 2678 | 0 |
| 43_neural_easy_ridge_no_guard | gt_1 | 6049 | 5651 | 3566 | 2085 | 572 | 432 | 0 |
| 43_neural_easy_ridge_source_zero_guard | zero_mass | 2244 | 2034 | 1200 | 834 | 360 | 270 | 0 |
| 43_neural_easy_ridge_source_zero_guard | le_002 | 17508 | 17101 | 12869 | 4232 | 3695 | 3051 | 2578 |
| 43_neural_easy_ridge_source_zero_guard | 002_to_01 | 97972 | 96165 | 74382 | 21783 | 14203 | 10328 | 0 |
| 43_neural_easy_ridge_source_zero_guard | 01_to_05 | 166925 | 163505 | 109793 | 53712 | 44324 | 29433 | 0 |
| 43_neural_easy_ridge_source_zero_guard | 05_to_1 | 28271 | 27466 | 17444 | 10022 | 3597 | 2678 | 0 |
| 43_neural_easy_ridge_source_zero_guard | gt_1 | 6049 | 5651 | 3566 | 2085 | 572 | 432 | 0 |
| 43_neural_easy_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_neural_easy_neural_underharm4_no_guard | le_002 | 4355 | 4231 | 3016 | 1215 | 1657 | 1338 | 1705 |
| 43_neural_easy_neural_underharm4_no_guard | 002_to_01 | 28371 | 27978 | 24220 | 3758 | 2956 | 2474 | 0 |
| 43_neural_easy_neural_underharm4_no_guard | 01_to_05 | 122386 | 120250 | 95550 | 24700 | 20913 | 16723 | 0 |
| 43_neural_easy_neural_underharm4_no_guard | 05_to_1 | 63686 | 62388 | 42913 | 19475 | 13380 | 10000 | 0 |
| 43_neural_easy_neural_underharm4_no_guard | gt_1 | 100171 | 97075 | 53555 | 43520 | 27845 | 15657 | 0 |
| 43_neural_easy_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_neural_easy_neural_underharm4_source_zero_guard | le_002 | 4355 | 4231 | 3016 | 1215 | 1657 | 1338 | 1701 |
| 43_neural_easy_neural_underharm4_source_zero_guard | 002_to_01 | 28371 | 27978 | 24220 | 3758 | 2956 | 2474 | 0 |
| 43_neural_easy_neural_underharm4_source_zero_guard | 01_to_05 | 122386 | 120250 | 95550 | 24700 | 20913 | 16723 | 0 |
| 43_neural_easy_neural_underharm4_source_zero_guard | 05_to_1 | 63686 | 62388 | 42913 | 19475 | 13380 | 10000 | 0 |
| 43_neural_easy_neural_underharm4_source_zero_guard | gt_1 | 100171 | 97075 | 53555 | 43520 | 27845 | 15657 | 0 |
| 43_damping097_all_ridge_no_guard | zero_mass | 7811 | 7638 | 6197 | 1316 | 1847 | 1445 | 0 |
| 43_damping097_all_ridge_no_guard | le_002 | 118053 | 116227 | 98907 | 15972 | 84226 | 70639 | 85632 |
| 43_damping097_all_ridge_no_guard | 002_to_01 | 187411 | 182550 | 127121 | 51531 | 72736 | 55175 | 0 |
| 43_damping097_all_ridge_no_guard | 01_to_05 | 5277 | 5101 | 2672 | 2301 | 321 | 228 | 0 |
| 43_damping097_all_ridge_no_guard | 05_to_1 | 241 | 236 | 172 | 61 | 29 | 23 | 0 |
| 43_damping097_all_ridge_no_guard | gt_1 | 176 | 170 | 127 | 39 | 17 | 13 | 0 |
| 43_damping097_all_ridge_source_zero_guard | zero_mass | 7811 | 7638 | 6197 | 1316 | 1847 | 1445 | 0 |
| 43_damping097_all_ridge_source_zero_guard | le_002 | 118053 | 116227 | 98907 | 15972 | 84226 | 70639 | 23679 |
| 43_damping097_all_ridge_source_zero_guard | 002_to_01 | 187411 | 182550 | 127121 | 51531 | 72736 | 55175 | 0 |
| 43_damping097_all_ridge_source_zero_guard | 01_to_05 | 5277 | 5101 | 2672 | 2301 | 321 | 228 | 0 |
| 43_damping097_all_ridge_source_zero_guard | 05_to_1 | 241 | 236 | 172 | 61 | 29 | 23 | 0 |
| 43_damping097_all_ridge_source_zero_guard | gt_1 | 176 | 170 | 127 | 39 | 17 | 13 | 0 |
| 43_damping097_all_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_damping097_all_neural_underharm4_no_guard | le_002 | 11460 | 11135 | 7435 | 3337 | 9616 | 6853 | 9887 |
| 43_damping097_all_neural_underharm4_no_guard | 002_to_01 | 199460 | 194936 | 148446 | 42966 | 121803 | 97366 | 0 |
| 43_damping097_all_neural_underharm4_no_guard | 01_to_05 | 103996 | 101858 | 76080 | 24206 | 27175 | 22815 | 0 |
| 43_damping097_all_neural_underharm4_no_guard | 05_to_1 | 2969 | 2927 | 2415 | 480 | 417 | 355 | 0 |
| 43_damping097_all_neural_underharm4_no_guard | gt_1 | 1084 | 1066 | 820 | 231 | 165 | 134 | 0 |
| 43_damping097_all_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_damping097_all_neural_underharm4_source_zero_guard | le_002 | 11460 | 11135 | 7435 | 3337 | 9616 | 6853 | 3960 |
| 43_damping097_all_neural_underharm4_source_zero_guard | 002_to_01 | 199460 | 194936 | 148446 | 42966 | 121803 | 97366 | 0 |
| 43_damping097_all_neural_underharm4_source_zero_guard | 01_to_05 | 103996 | 101858 | 76080 | 24206 | 27175 | 22815 | 0 |
| 43_damping097_all_neural_underharm4_source_zero_guard | 05_to_1 | 2969 | 2927 | 2415 | 480 | 417 | 355 | 0 |
| 43_damping097_all_neural_underharm4_source_zero_guard | gt_1 | 1084 | 1066 | 820 | 231 | 165 | 134 | 0 |
| 43_damping097_easy_ridge_no_guard | zero_mass | 1243 | 1123 | 807 | 242 | 666 | 511 | 0 |
| 43_damping097_easy_ridge_no_guard | le_002 | 169110 | 166224 | 133761 | 30315 | 91066 | 75998 | 92611 |
| 43_damping097_easy_ridge_no_guard | 002_to_01 | 122405 | 119402 | 85192 | 31767 | 57766 | 43347 | 0 |
| 43_damping097_easy_ridge_no_guard | 01_to_05 | 24757 | 23856 | 14806 | 8320 | 9183 | 7300 | 0 |
| 43_damping097_easy_ridge_no_guard | 05_to_1 | 1096 | 987 | 432 | 482 | 329 | 252 | 0 |
| 43_damping097_easy_ridge_no_guard | gt_1 | 358 | 330 | 198 | 94 | 166 | 115 | 0 |
| 43_damping097_easy_ridge_source_zero_guard | zero_mass | 1243 | 1123 | 807 | 242 | 666 | 511 | 0 |
| 43_damping097_easy_ridge_source_zero_guard | le_002 | 169110 | 166224 | 133761 | 30315 | 91066 | 75998 | 36857 |
| 43_damping097_easy_ridge_source_zero_guard | 002_to_01 | 122405 | 119402 | 85192 | 31767 | 57766 | 43347 | 0 |
| 43_damping097_easy_ridge_source_zero_guard | 01_to_05 | 24757 | 23856 | 14806 | 8320 | 9183 | 7300 | 0 |
| 43_damping097_easy_ridge_source_zero_guard | 05_to_1 | 1096 | 987 | 432 | 482 | 329 | 252 | 0 |
| 43_damping097_easy_ridge_source_zero_guard | gt_1 | 358 | 330 | 198 | 94 | 166 | 115 | 0 |
| 43_damping097_easy_neural_underharm4_no_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_damping097_easy_neural_underharm4_no_guard | le_002 | 117204 | 115235 | 90708 | 22961 | 58910 | 49389 | 59885 |
| 43_damping097_easy_neural_underharm4_no_guard | 002_to_01 | 124669 | 122561 | 101991 | 19005 | 59021 | 50468 | 0 |
| 43_damping097_easy_neural_underharm4_no_guard | 01_to_05 | 37707 | 36413 | 22584 | 12801 | 18427 | 13324 | 0 |
| 43_damping097_easy_neural_underharm4_no_guard | 05_to_1 | 9311 | 8889 | 4828 | 3770 | 4867 | 3299 | 0 |
| 43_damping097_easy_neural_underharm4_no_guard | gt_1 | 30078 | 28824 | 15085 | 12683 | 17951 | 11043 | 0 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | zero_mass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | le_002 | 117204 | 115235 | 90708 | 22961 | 58910 | 49389 | 51541 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | 002_to_01 | 124669 | 122561 | 101991 | 19005 | 59021 | 50468 | 0 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | 01_to_05 | 37707 | 36413 | 22584 | 12801 | 18427 | 13324 | 0 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | 05_to_1 | 9311 | 8889 | 4828 | 3770 | 4867 | 3299 | 0 |
| 43_damping097_easy_neural_underharm4_source_zero_guard | gt_1 | 30078 | 28824 | 15085 | 12683 | 17951 | 11043 | 0 |

Per-locality contributions and conditional intervals are retained in summary_metrics.json.
Unknown and zero-reference cases remain explicit. These are source-only image-pixel obs8/pred12 raw-stride12 results;
not t50, seconds, metric, true 3D, foundation, physical safety or submission-ready evidence. Stage5C/SMC off.
