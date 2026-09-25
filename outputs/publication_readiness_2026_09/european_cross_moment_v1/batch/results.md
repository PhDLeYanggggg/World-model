# Cross-Moment batch: Every Registered View

Batch mode changes pair weighting; fitting mode changes only its normalizer.
Control: {'product': 'supported_pair_control', 'hurdle': 'cross_moment_batch'}.
Treatment is cross-moment ranking. All count controls are offline diagnostics.

| View | ADE gain vs CV (%) | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Predicted violations |
|---|---:|---:|---:|---:|---|---:|---:|
| neural_fold0_seed17_all_control_original | 2.097745 | 3.183188 | 1.724884 | -4.078708 | 2/4 | 30.007519 | 0 |
| neural_fold0_seed17_all_treatment_original | 2.226082 | 3.317301 | 1.936496 | -2.759284 | 3/4 | 28.589237 | 0 |
| neural_fold0_seed17_all_control_common | 2.097745 | 3.183188 | 1.724884 | -4.078708 | 2/4 | 30.007519 | 0 |
| neural_fold0_seed17_all_treatment_common | 2.226082 | 3.317301 | 1.936496 | -2.759284 | 3/4 | 28.589237 | 0 |
| neural_fold0_seed17_all_treatment_at_control | 2.299888 | 3.435897 | 2.037849 | -2.759284 | 3/4 | 30.007519 | 2961 |
| neural_fold0_seed17_all_control_at_treatment | 2.016531 | 3.051119 | 1.544773 | -4.078708 | 2/4 | 28.589237 | 94 |
| neural_fold0_seed17_easy_control_original | 0.426666 | 0.673281 | 0.162768 | -1.923476 | 2/4 | 12.484541 | 0 |
| neural_fold0_seed17_easy_treatment_original | 0.332789 | 0.544375 | 0.067299 | -1.762257 | 2/4 | 11.779605 | 0 |
| neural_fold0_seed17_easy_control_common | 0.426666 | 0.673281 | 0.162768 | -1.923476 | 2/4 | 12.484541 | 0 |
| neural_fold0_seed17_easy_treatment_common | 0.332789 | 0.544375 | 0.067299 | -1.762257 | 2/4 | 11.779605 | 0 |
| neural_fold0_seed17_easy_treatment_at_control | 0.339727 | 0.553679 | 0.069104 | -1.960704 | 2/4 | 12.484541 | 1465 |
| neural_fold0_seed17_easy_control_at_treatment | 0.418617 | 0.664489 | 0.158168 | -1.751485 | 2/4 | 11.779605 | 40 |
| damping097_fold0_seed17_all_control_original | 3.354003 | 5.325797 | 3.286328 | -0.753575 | 0/4 | 63.203823 | 0 |
| damping097_fold0_seed17_all_treatment_original | 4.375189 | 6.859494 | 4.540174 | 1.585499 | 0/4 | 67.610044 | 0 |
| damping097_fold0_seed17_all_control_common | 3.354003 | 5.325797 | 3.286328 | -0.753575 | 0/4 | 63.203823 | 0 |
| damping097_fold0_seed17_all_treatment_common | 4.375189 | 6.859494 | 4.540174 | 1.585499 | 0/4 | 67.610044 | 0 |
| damping097_fold0_seed17_all_treatment_at_control | 3.995334 | 6.289391 | 4.034549 | -0.601852 | 0/4 | 63.203823 | 0 |
| damping097_fold0_seed17_all_control_at_treatment | 3.800537 | 5.968143 | 3.879321 | 0.727317 | 0/4 | 67.610044 | 8907 |
| damping097_fold0_seed17_easy_control_original | 0.962367 | 1.574708 | 0.557170 | -0.733333 | 0/4 | 46.563375 | 0 |
| damping097_fold0_seed17_easy_treatment_original | 0.638494 | 1.054715 | 0.207404 | -0.740510 | 0/4 | 44.288781 | 0 |
| damping097_fold0_seed17_easy_control_common | 0.962367 | 1.574708 | 0.557170 | -0.733333 | 0/4 | 46.563375 | 0 |
| damping097_fold0_seed17_easy_treatment_common | 0.638494 | 1.054715 | 0.207404 | -0.740510 | 0/4 | 44.288781 | 0 |
| damping097_fold0_seed17_easy_treatment_at_control | 0.788916 | 1.300344 | 0.339983 | -0.733333 | 0/4 | 46.563375 | 4598 |
| damping097_fold0_seed17_easy_control_at_treatment | 0.766972 | 1.255457 | 0.377600 | -0.733333 | 0/4 | 44.288781 | 0 |
| neural_fold0_seed29_all_control_original | 1.857938 | 2.852020 | 1.143010 | -2.415837 | 3/4 | 36.416748 | 0 |
| neural_fold0_seed29_all_treatment_original | 2.111865 | 3.294770 | 1.591448 | -1.655855 | 3/4 | 34.654656 | 0 |
| neural_fold0_seed29_all_control_common | 1.857938 | 2.852020 | 1.143010 | -2.415837 | 3/4 | 36.416748 | 0 |
| neural_fold0_seed29_all_treatment_common | 2.111865 | 3.294770 | 1.591448 | -1.655855 | 3/4 | 34.654656 | 0 |
| neural_fold0_seed29_all_treatment_at_control | 2.170999 | 3.398249 | 1.635720 | -1.908461 | 4/4 | 36.416748 | 3732 |
| neural_fold0_seed29_all_control_at_treatment | 1.808967 | 2.793298 | 1.109829 | -3.191831 | 3/4 | 34.654656 | 170 |
| neural_fold0_seed29_easy_control_original | 0.460089 | 0.784847 | 0.144303 | -2.940380 | 2/4 | 18.982814 | 0 |
| neural_fold0_seed29_easy_treatment_original | 0.310448 | 0.533308 | 0.017749 | -2.776698 | 2/4 | 16.831399 | 0 |
| neural_fold0_seed29_easy_control_common | 0.460089 | 0.784847 | 0.144303 | -2.940380 | 2/4 | 18.982814 | 0 |
| neural_fold0_seed29_easy_treatment_common | 0.310448 | 0.533308 | 0.017749 | -2.776698 | 2/4 | 16.831399 | 0 |
| neural_fold0_seed29_easy_treatment_at_control | 0.357717 | 0.594828 | 0.029560 | -2.887159 | 2/4 | 18.982814 | 4349 |
| neural_fold0_seed29_easy_control_at_treatment | 0.402793 | 0.700812 | 0.126039 | -2.790456 | 2/4 | 16.831399 | 0 |
| damping097_fold0_seed29_all_control_original | 3.335169 | 5.315657 | 3.253602 | 0.181579 | 0/4 | 60.395951 | 0 |
| damping097_fold0_seed29_all_treatment_original | 4.494621 | 7.066909 | 4.714663 | 1.634624 | 0/4 | 65.113334 | 0 |
| damping097_fold0_seed29_all_control_common | 3.335169 | 5.315657 | 3.253602 | 0.181579 | 0/4 | 60.395951 | 0 |
| damping097_fold0_seed29_all_treatment_common | 4.494621 | 7.066909 | 4.714663 | 1.634624 | 0/4 | 65.113334 | 0 |
| damping097_fold0_seed29_all_treatment_at_control | 3.994667 | 6.242796 | 4.138368 | 0.770199 | 0/4 | 60.395951 | 0 |
| damping097_fold0_seed29_all_control_at_treatment | 3.954658 | 6.275302 | 4.101898 | 1.589437 | 0/4 | 65.113334 | 9536 |
| damping097_fold0_seed29_easy_control_original | 0.704842 | 1.163311 | 0.282014 | -1.310473 | 0/4 | 43.730274 | 0 |
| damping097_fold0_seed29_easy_treatment_original | 0.505857 | 0.839834 | 0.080289 | -1.263071 | 0/4 | 41.227133 | 0 |
| damping097_fold0_seed29_easy_control_common | 0.704842 | 1.163311 | 0.282014 | -1.310473 | 0/4 | 43.730274 | 0 |
| damping097_fold0_seed29_easy_treatment_common | 0.505857 | 0.839834 | 0.080289 | -1.263071 | 0/4 | 41.227133 | 0 |
| damping097_fold0_seed29_easy_treatment_at_control | 0.626359 | 1.028493 | 0.150074 | -1.310473 | 0/4 | 43.730274 | 5060 |
| damping097_fold0_seed29_easy_control_at_treatment | 0.571676 | 0.951808 | 0.186277 | -1.265092 | 0/4 | 41.227133 | 0 |
| neural_fold0_seed43_all_control_original | 2.235334 | 3.470064 | 1.740238 | -2.998448 | 3/4 | 38.705688 | 0 |
| neural_fold0_seed43_all_treatment_original | 2.006631 | 3.075791 | 1.407716 | -2.837189 | 4/4 | 36.300496 | 0 |
| neural_fold0_seed43_all_control_common | 2.235334 | 3.470064 | 1.740238 | -2.998448 | 3/4 | 38.705688 | 0 |
| neural_fold0_seed43_all_treatment_common | 2.006631 | 3.075791 | 1.407716 | -2.837189 | 4/4 | 36.300496 | 0 |
| neural_fold0_seed43_all_treatment_at_control | 2.154330 | 3.315222 | 1.650947 | -2.981475 | 4/4 | 38.705688 | 4933 |
| neural_fold0_seed43_all_control_at_treatment | 2.068901 | 3.218448 | 1.570179 | -4.178408 | 3/4 | 36.300496 | 71 |
| neural_fold0_seed43_easy_control_original | 0.451061 | 0.747138 | 0.127183 | -3.183055 | 2/4 | 19.365211 | 0 |
| neural_fold0_seed43_easy_treatment_original | 0.373928 | 0.620363 | 0.044111 | -2.893174 | 2/4 | 19.611073 | 0 |
| neural_fold0_seed43_easy_control_common | 0.451061 | 0.747138 | 0.127183 | -3.183055 | 2/4 | 19.365211 | 0 |
| neural_fold0_seed43_easy_treatment_common | 0.373928 | 0.620363 | 0.044111 | -2.893174 | 2/4 | 19.611073 | 0 |
| neural_fold0_seed43_easy_treatment_at_control | 0.371395 | 0.617134 | 0.042932 | -2.860470 | 2/4 | 19.365211 | 48 |
| neural_fold0_seed43_easy_control_at_treatment | 0.453629 | 0.749828 | 0.126684 | -3.106277 | 2/4 | 19.611073 | 545 |
| damping097_fold0_seed43_all_control_original | 3.740667 | 5.895224 | 3.778473 | -0.242480 | 0/4 | 66.171480 | 0 |
| damping097_fold0_seed43_all_treatment_original | 4.630725 | 7.229988 | 4.824467 | 1.860211 | 0/4 | 70.688018 | 0 |
| damping097_fold0_seed43_all_control_common | 3.740667 | 5.895224 | 3.778473 | -0.242480 | 0/4 | 66.171480 | 0 |
| damping097_fold0_seed43_all_treatment_common | 4.630725 | 7.229988 | 4.824467 | 1.860211 | 0/4 | 70.688018 | 0 |
| damping097_fold0_seed43_all_treatment_at_control | 4.262565 | 6.676488 | 4.425578 | 0.517998 | 0/4 | 66.171480 | 0 |
| damping097_fold0_seed43_all_control_at_treatment | 4.239910 | 6.627793 | 4.393728 | 1.912056 | 0/4 | 70.688018 | 9130 |
| damping097_fold0_seed43_easy_control_original | 1.025782 | 1.669471 | 0.721406 | -0.738833 | 0/4 | 47.498837 | 0 |
| damping097_fold0_seed43_easy_treatment_original | 0.701725 | 1.188686 | 0.282024 | -0.958676 | 0/4 | 45.698159 | 0 |
| damping097_fold0_seed43_easy_control_common | 1.025782 | 1.669471 | 0.721406 | -0.738833 | 0/4 | 47.498837 | 0 |
| damping097_fold0_seed43_easy_treatment_common | 0.701725 | 1.188686 | 0.282024 | -0.958676 | 0/4 | 45.698159 | 0 |
| damping097_fold0_seed43_easy_treatment_at_control | 0.838530 | 1.392601 | 0.398792 | -0.746011 | 0/4 | 47.498837 | 3640 |
| damping097_fold0_seed43_easy_control_at_treatment | 0.861466 | 1.402119 | 0.568202 | -0.738833 | 0/4 | 45.698159 | 0 |
| neural_fold1_seed17_all_control_original | 2.516139 | 3.773795 | 1.283281 | -4.168900 | 3/4 | 50.231532 | 0 |
| neural_fold1_seed17_all_treatment_original | 3.281879 | 4.789355 | 2.331359 | -3.662656 | 4/4 | 53.522444 | 0 |
| neural_fold1_seed17_all_control_common | 2.516139 | 3.773795 | 1.283281 | -4.168900 | 3/4 | 50.231532 | 0 |
| neural_fold1_seed17_all_treatment_common | 3.281879 | 4.789355 | 2.331359 | -3.662656 | 4/4 | 53.522444 | 0 |
| neural_fold1_seed17_all_treatment_at_control | 2.719389 | 3.993303 | 1.847944 | -3.767097 | 4/4 | 50.231532 | 14 |
| neural_fold1_seed17_all_control_at_treatment | 3.075086 | 4.587869 | 1.763980 | -3.945082 | 3/4 | 53.522444 | 9729 |
| neural_fold1_seed17_easy_control_original | 0.451026 | 0.751748 | 0.100511 | -2.613569 | 0/4 | 16.111745 | 0 |
| neural_fold1_seed17_easy_treatment_original | 0.378773 | 0.661296 | 0.056568 | -2.817163 | 1/4 | 14.179542 | 0 |
| neural_fold1_seed17_easy_control_common | 0.451026 | 0.751748 | 0.100511 | -2.613569 | 0/4 | 16.111745 | 0 |
| neural_fold1_seed17_easy_treatment_common | 0.378773 | 0.661296 | 0.056568 | -2.817163 | 1/4 | 14.179542 | 0 |
| neural_fold1_seed17_easy_treatment_at_control | 0.402489 | 0.697195 | 0.059579 | -2.698992 | 1/4 | 16.111745 | 5874 |
| neural_fold1_seed17_easy_control_at_treatment | 0.436804 | 0.736350 | 0.097003 | -2.653720 | 0/4 | 14.179542 | 170 |
| damping097_fold1_seed17_all_control_original | 3.671691 | 5.930314 | 3.329866 | -1.168639 | 0/4 | 63.127229 | 0 |
| damping097_fold1_seed17_all_treatment_original | 4.366782 | 6.889977 | 4.339656 | -0.564990 | 0/4 | 66.114625 | 0 |
| damping097_fold1_seed17_all_control_common | 3.671691 | 5.930314 | 3.329866 | -1.168639 | 0/4 | 63.127229 | 0 |
| damping097_fold1_seed17_all_treatment_common | 4.366782 | 6.889977 | 4.339656 | -0.564990 | 0/4 | 66.114625 | 0 |
| damping097_fold1_seed17_all_treatment_at_control | 4.116332 | 6.544953 | 3.968714 | -1.064975 | 0/4 | 63.127229 | 0 |
| damping097_fold1_seed17_all_control_at_treatment | 3.963306 | 6.320635 | 3.946336 | -0.901258 | 0/4 | 66.114625 | 8819 |
| damping097_fold1_seed17_easy_control_original | 2.015405 | 3.297610 | 1.782652 | 1.142704 | 0/4 | 55.587097 | 0 |
| damping097_fold1_seed17_easy_treatment_original | 0.808908 | 1.358741 | 0.222619 | -0.309701 | 0/4 | 39.620334 | 0 |
| damping097_fold1_seed17_easy_control_common | 2.015405 | 3.297610 | 1.782652 | 1.142704 | 0/4 | 55.587097 | 0 |
| damping097_fold1_seed17_easy_treatment_common | 0.808908 | 1.358741 | 0.222619 | -0.309701 | 0/4 | 39.620334 | 0 |
| damping097_fold1_seed17_easy_treatment_at_control | 2.040999 | 3.196575 | 1.733594 | 1.136791 | 0/4 | 55.587097 | 47135 |
| damping097_fold1_seed17_easy_control_at_treatment | 0.881296 | 1.419764 | 0.302904 | -0.251065 | 0/4 | 39.620334 | 0 |
| neural_fold1_seed29_all_control_original | 2.504319 | 3.838100 | 1.388208 | -4.525943 | 3/4 | 40.858448 | 0 |
| neural_fold1_seed29_all_treatment_original | 3.297682 | 4.873784 | 2.601430 | -4.329509 | 3/4 | 43.407846 | 0 |
| neural_fold1_seed29_all_control_common | 2.504319 | 3.838100 | 1.388208 | -4.525943 | 3/4 | 40.858448 | 0 |
| neural_fold1_seed29_all_treatment_common | 3.297682 | 4.873784 | 2.601430 | -4.329509 | 3/4 | 43.407846 | 0 |
| neural_fold1_seed29_all_treatment_at_control | 2.783840 | 4.061301 | 1.996571 | -4.322909 | 3/4 | 40.858448 | 10 |
| neural_fold1_seed29_all_control_at_treatment | 2.911138 | 4.457284 | 1.769811 | -4.525943 | 3/4 | 43.407846 | 7536 |
| neural_fold1_seed29_easy_control_original | 0.572578 | 0.905352 | 0.273998 | -2.189646 | 1/4 | 14.595521 | 0 |
| neural_fold1_seed29_easy_treatment_original | 0.471294 | 0.837976 | 0.077317 | -3.178346 | 1/4 | 14.324186 | 0 |
| neural_fold1_seed29_easy_control_common | 0.572578 | 0.905352 | 0.273998 | -2.189646 | 1/4 | 14.595521 | 0 |
| neural_fold1_seed29_easy_treatment_common | 0.471294 | 0.837976 | 0.077317 | -3.178346 | 1/4 | 14.324186 | 0 |
| neural_fold1_seed29_easy_treatment_at_control | 0.525423 | 0.927660 | 0.087439 | -3.189160 | 1/4 | 14.595521 | 1424 |
| neural_fold1_seed29_easy_control_at_treatment | 0.573656 | 0.985862 | 0.225995 | -1.952531 | 1/4 | 14.324186 | 623 |
| damping097_fold1_seed29_all_control_original | 3.874346 | 6.168474 | 3.697191 | -0.593639 | 0/4 | 62.465321 | 0 |
| damping097_fold1_seed29_all_treatment_original | 4.384809 | 7.004164 | 4.694431 | -0.438749 | 0/4 | 65.520804 | 0 |
| damping097_fold1_seed29_all_control_common | 3.874346 | 6.168474 | 3.697191 | -0.593639 | 0/4 | 62.465321 | 0 |
| damping097_fold1_seed29_all_treatment_common | 4.384809 | 7.004164 | 4.694431 | -0.438749 | 0/4 | 65.520804 | 0 |
| damping097_fold1_seed29_all_treatment_at_control | 4.189328 | 6.616852 | 4.317802 | -0.509114 | 0/4 | 62.465321 | 0 |
| damping097_fold1_seed29_all_control_at_treatment | 4.055233 | 6.444005 | 4.078121 | -0.454010 | 0/4 | 65.520804 | 9020 |
| damping097_fold1_seed29_easy_control_original | 2.025984 | 3.124833 | 2.063009 | 2.467059 | 0/4 | 54.655208 | 0 |
| damping097_fold1_seed29_easy_treatment_original | 1.039105 | 1.645581 | 0.496558 | -1.230417 | 0/4 | 42.061333 | 0 |
| damping097_fold1_seed29_easy_control_common | 2.025984 | 3.124833 | 2.063009 | 2.467059 | 0/4 | 54.655208 | 0 |
| damping097_fold1_seed29_easy_treatment_common | 1.039105 | 1.645581 | 0.496558 | -1.230417 | 0/4 | 42.061333 | 0 |
| damping097_fold1_seed29_easy_treatment_at_control | 1.894884 | 2.863944 | 1.747634 | 1.805786 | 0/4 | 54.655208 | 37178 |
| damping097_fold1_seed29_easy_control_at_treatment | 1.071494 | 1.717412 | 0.517306 | -0.824660 | 0/4 | 42.061333 | 0 |
| neural_fold1_seed43_all_control_original | 2.955752 | 4.558475 | 1.795116 | -2.789988 | 4/4 | 53.130176 | 0 |
| neural_fold1_seed43_all_treatment_original | 3.475632 | 5.163616 | 2.616718 | -2.785595 | 4/4 | 51.435772 | 0 |
| neural_fold1_seed43_all_control_common | 2.955752 | 4.558475 | 1.795116 | -2.789988 | 4/4 | 53.130176 | 0 |
| neural_fold1_seed43_all_treatment_common | 3.475632 | 5.163616 | 2.616718 | -2.785595 | 4/4 | 51.435772 | 0 |
| neural_fold1_seed43_all_treatment_at_control | 3.374446 | 4.969645 | 2.580217 | -2.758709 | 4/4 | 53.130176 | 5519 |
| neural_fold1_seed43_all_control_at_treatment | 3.120597 | 4.834037 | 1.879124 | -2.789988 | 4/4 | 51.435772 | 517 |
| neural_fold1_seed43_easy_control_original | 0.527596 | 0.807165 | 0.249171 | -1.666304 | 1/4 | 18.567649 | 0 |
| neural_fold1_seed43_easy_treatment_original | 0.457482 | 0.804443 | 0.095600 | -2.492550 | 1/4 | 16.656787 | 0 |
| neural_fold1_seed43_easy_control_common | 0.527596 | 0.807165 | 0.249171 | -1.666304 | 1/4 | 18.567649 | 0 |
| neural_fold1_seed43_easy_treatment_common | 0.457482 | 0.804443 | 0.095600 | -2.492550 | 1/4 | 16.656787 | 0 |
| neural_fold1_seed43_easy_treatment_at_control | 0.564039 | 0.949639 | 0.197432 | -2.661048 | 1/4 | 18.567649 | 5641 |
| neural_fold1_seed43_easy_control_at_treatment | 0.518874 | 0.871309 | 0.234999 | -2.515960 | 1/4 | 16.656787 | 0 |
| damping097_fold1_seed43_all_control_original | 3.764070 | 6.096085 | 3.581467 | 0.376509 | 0/4 | 65.411389 | 0 |
| damping097_fold1_seed43_all_treatment_original | 4.506847 | 7.184891 | 4.882285 | 0.252048 | 0/4 | 65.708469 | 0 |
| damping097_fold1_seed43_all_control_common | 3.764070 | 6.096085 | 3.581467 | 0.376509 | 0/4 | 65.411389 | 0 |
| damping097_fold1_seed43_all_treatment_common | 4.506847 | 7.184891 | 4.882285 | 0.252048 | 0/4 | 65.708469 | 0 |
| damping097_fold1_seed43_all_treatment_at_control | 4.300610 | 6.835721 | 4.325510 | -0.674697 | 0/4 | 65.411389 | 40 |
| damping097_fold1_seed43_all_control_at_treatment | 3.921115 | 6.364599 | 3.939214 | 1.338690 | 0/4 | 65.708469 | 917 |
| damping097_fold1_seed43_easy_control_original | 2.316209 | 3.753573 | 2.367646 | 2.048852 | 0/4 | 57.931892 | 0 |
| damping097_fold1_seed43_easy_treatment_original | 0.813741 | 1.371460 | 0.208664 | -1.211075 | 0/4 | 39.629819 | 0 |
| damping097_fold1_seed43_easy_control_common | 2.316209 | 3.753573 | 2.367646 | 2.048852 | 0/4 | 57.931892 | 0 |
| damping097_fold1_seed43_easy_treatment_common | 0.813741 | 1.371460 | 0.208664 | -1.211075 | 0/4 | 39.629819 | 0 |
| damping097_fold1_seed43_easy_treatment_at_control | 2.128734 | 3.322205 | 1.815444 | 2.106157 | 0/4 | 57.931892 | 54029 |
| damping097_fold1_seed43_easy_control_at_treatment | 0.890534 | 1.518573 | 0.394901 | -0.709614 | 0/4 | 39.629819 | 0 |
| neural_fold2_seed17_all_control_original | 0.950495 | 1.403498 | 1.038842 | -0.114752 | 0/0 | 9.508127 | 0 |
| neural_fold2_seed17_all_treatment_original | 0.613641 | 0.874707 | 0.599564 | 0.131986 | 0/0 | 10.290572 | 0 |
| neural_fold2_seed17_all_control_common | 0.950495 | 1.403498 | 1.038842 | -0.114752 | 0/0 | 9.508127 | 0 |
| neural_fold2_seed17_all_treatment_common | 0.613641 | 0.874707 | 0.599564 | 0.131986 | 0/0 | 10.290572 | 0 |
| neural_fold2_seed17_all_treatment_at_control | 0.539380 | 0.770053 | 0.533252 | 0.042071 | 0/0 | 9.508127 | 0 |
| neural_fold2_seed17_all_control_at_treatment | 1.020865 | 1.512896 | 1.108597 | 0.120176 | 0/0 | 10.290572 | 1100 |
| neural_fold2_seed17_easy_control_original | 0.115339 | 0.181770 | 0.052162 | 0.060609 | 0/0 | 0.519259 | 0 |
| neural_fold2_seed17_easy_treatment_original | 0.144482 | 0.153073 | 0.089705 | 0.173793 | 0/0 | 0.482271 | 0 |
| neural_fold2_seed17_easy_control_common | 0.115339 | 0.181770 | 0.052162 | 0.060609 | 0/0 | 0.519259 | 0 |
| neural_fold2_seed17_easy_treatment_common | 0.144482 | 0.153073 | 0.089705 | 0.173793 | 0/0 | 0.482271 | 0 |
| neural_fold2_seed17_easy_treatment_at_control | 0.145759 | 0.155269 | 0.089764 | 0.175464 | 0/0 | 0.519259 | 57 |
| neural_fold2_seed17_easy_control_at_treatment | 0.098043 | 0.152937 | 0.052162 | 0.065759 | 0/0 | 0.482271 | 5 |
| damping097_fold2_seed17_all_control_original | 2.725395 | 4.368773 | 2.180938 | -2.897101 | 0/0 | 61.070527 | 0 |
| damping097_fold2_seed17_all_treatment_original | 3.419451 | 5.460161 | 2.988070 | -2.843555 | 0/0 | 64.849024 | 0 |
| damping097_fold2_seed17_all_control_common | 2.725395 | 4.368773 | 2.180938 | -2.897101 | 0/0 | 61.070527 | 0 |
| damping097_fold2_seed17_all_treatment_common | 3.419451 | 5.460161 | 2.988070 | -2.843555 | 0/0 | 64.849024 | 0 |
| damping097_fold2_seed17_all_treatment_at_control | 3.046140 | 4.876499 | 2.590207 | -2.911855 | 0/0 | 61.070527 | 0 |
| damping097_fold2_seed17_all_control_at_treatment | 3.112586 | 4.988895 | 2.646383 | -2.629616 | 0/0 | 64.849024 | 5312 |
| damping097_fold2_seed17_easy_control_original | 0.716912 | 1.151618 | 0.315631 | -3.487691 | 0/0 | 36.205143 | 0 |
| damping097_fold2_seed17_easy_treatment_original | 0.466020 | 0.762282 | 0.169469 | -3.380052 | 0/0 | 28.257638 | 0 |
| damping097_fold2_seed17_easy_control_common | 0.716912 | 1.151618 | 0.315631 | -3.487691 | 0/0 | 36.205143 | 0 |
| damping097_fold2_seed17_easy_treatment_common | 0.466020 | 0.762282 | 0.169469 | -3.380052 | 0/0 | 28.257638 | 0 |
| damping097_fold2_seed17_easy_treatment_at_control | 0.661711 | 1.069614 | 0.257939 | -3.515820 | 0/0 | 36.205143 | 11173 |
| damping097_fold2_seed17_easy_control_at_treatment | 0.503646 | 0.819928 | 0.214973 | -3.314841 | 0/0 | 28.257638 | 0 |
| neural_fold2_seed29_all_control_original | 1.175740 | 1.657428 | 1.217576 | -0.325611 | 0/0 | 7.736956 | 0 |
| neural_fold2_seed29_all_treatment_original | 0.634182 | 0.817347 | 0.571158 | -0.122335 | 0/0 | 7.679340 | 0 |
| neural_fold2_seed29_all_control_common | 1.175740 | 1.657428 | 1.217576 | -0.325611 | 0/0 | 7.736956 | 0 |
| neural_fold2_seed29_all_treatment_common | 0.634182 | 0.817347 | 0.571158 | -0.122335 | 0/0 | 7.679340 | 0 |
| neural_fold2_seed29_all_treatment_at_control | 0.602080 | 0.801861 | 0.534561 | -0.208126 | 0/0 | 7.736956 | 704 |
| neural_fold2_seed29_all_control_at_treatment | 1.247341 | 1.791999 | 1.309498 | -0.405330 | 0/0 | 7.679340 | 623 |
| neural_fold2_seed29_easy_control_original | 0.308107 | 0.410344 | 0.166315 | -0.219386 | 0/0 | 0.820856 | 0 |
| neural_fold2_seed29_easy_treatment_original | 0.276658 | 0.317228 | 0.106861 | -0.223220 | 0/0 | 0.826546 | 0 |
| neural_fold2_seed29_easy_control_common | 0.308107 | 0.410344 | 0.166315 | -0.219386 | 0/0 | 0.820856 | 0 |
| neural_fold2_seed29_easy_treatment_common | 0.276658 | 0.317228 | 0.106861 | -0.223220 | 0/0 | 0.826546 | 0 |
| neural_fold2_seed29_easy_treatment_at_control | 0.270731 | 0.314221 | 0.094184 | -0.223903 | 0/0 | 0.820856 | 23 |
| neural_fold2_seed29_easy_control_at_treatment | 0.303808 | 0.410749 | 0.158145 | -0.221258 | 0/0 | 0.826546 | 31 |
| damping097_fold2_seed29_all_control_original | 2.976640 | 4.774715 | 2.512350 | -2.180797 | 0/0 | 57.085749 | 0 |
| damping097_fold2_seed29_all_treatment_original | 3.577220 | 5.670895 | 3.290416 | -2.190022 | 0/0 | 59.351282 | 0 |
| damping097_fold2_seed29_all_control_common | 2.976640 | 4.774715 | 2.512350 | -2.180797 | 0/0 | 57.085749 | 0 |
| damping097_fold2_seed29_all_treatment_common | 3.577220 | 5.670895 | 3.290416 | -2.190022 | 0/0 | 59.351282 | 0 |
| damping097_fold2_seed29_all_treatment_at_control | 3.323031 | 5.275305 | 2.991941 | -2.190022 | 0/0 | 57.085749 | 0 |
| damping097_fold2_seed29_all_control_at_treatment | 3.212780 | 5.125622 | 2.794926 | -2.017099 | 0/0 | 59.351282 | 3185 |
| damping097_fold2_seed29_easy_control_original | 0.506624 | 0.829873 | 0.135439 | -2.880705 | 0/0 | 30.348188 | 0 |
| damping097_fold2_seed29_easy_treatment_original | 0.425814 | 0.703574 | 0.127930 | -2.666981 | 0/0 | 26.482199 | 0 |
| damping097_fold2_seed29_easy_control_common | 0.506624 | 0.829873 | 0.135439 | -2.880705 | 0/0 | 30.348188 | 0 |
| damping097_fold2_seed29_easy_treatment_common | 0.425814 | 0.703574 | 0.127930 | -2.666981 | 0/0 | 26.482199 | 0 |
| damping097_fold2_seed29_easy_treatment_at_control | 0.526429 | 0.859292 | 0.155134 | -2.705416 | 0/0 | 30.348188 | 5438 |
| damping097_fold2_seed29_easy_control_at_treatment | 0.414512 | 0.683419 | 0.108341 | -2.872812 | 0/0 | 26.482199 | 3 |
| neural_fold2_seed43_all_control_original | 0.726629 | 1.118725 | 0.778022 | -0.124823 | 0/0 | 9.463314 | 0 |
| neural_fold2_seed43_all_treatment_original | 0.561045 | 0.839171 | 0.531454 | -0.487873 | 0/0 | 9.839599 | 0 |
| neural_fold2_seed43_all_control_common | 0.726629 | 1.118725 | 0.778022 | -0.124823 | 0/0 | 9.463314 | 0 |
| neural_fold2_seed43_all_treatment_common | 0.561045 | 0.839171 | 0.531454 | -0.487873 | 0/0 | 9.839599 | 0 |
| neural_fold2_seed43_all_treatment_at_control | 0.516894 | 0.763617 | 0.486032 | -0.438653 | 0/0 | 9.463314 | 9 |
| neural_fold2_seed43_all_control_at_treatment | 0.774353 | 1.239621 | 0.813626 | 0.214437 | 0/0 | 9.839599 | 538 |
| neural_fold2_seed43_easy_control_original | 0.115464 | 0.187522 | 0.037463 | -0.035136 | 0/0 | 0.544155 | 0 |
| neural_fold2_seed43_easy_treatment_original | 0.157530 | 0.165736 | 0.076813 | -0.033643 | 0/0 | 0.428211 | 0 |
| neural_fold2_seed43_easy_control_common | 0.115464 | 0.187522 | 0.037463 | -0.035136 | 0/0 | 0.544155 | 0 |
| neural_fold2_seed43_easy_treatment_common | 0.157530 | 0.165736 | 0.076813 | -0.033643 | 0/0 | 0.428211 | 0 |
| neural_fold2_seed43_easy_treatment_at_control | 0.159033 | 0.167476 | 0.078714 | -0.034533 | 0/0 | 0.544155 | 172 |
| neural_fold2_seed43_easy_control_at_treatment | 0.110851 | 0.166930 | 0.046456 | -0.031634 | 0/0 | 0.428211 | 9 |
| damping097_fold2_seed43_all_control_original | 2.846330 | 4.563553 | 2.354107 | -2.247175 | 0/0 | 58.999893 | 0 |
| damping097_fold2_seed43_all_treatment_original | 3.452928 | 5.468897 | 3.134996 | -2.233540 | 0/0 | 60.789558 | 0 |
| damping097_fold2_seed43_all_control_common | 2.846330 | 4.563553 | 2.354107 | -2.247175 | 0/0 | 58.999893 | 0 |
| damping097_fold2_seed43_all_treatment_common | 3.452928 | 5.468897 | 3.134996 | -2.233540 | 0/0 | 60.789558 | 0 |
| damping097_fold2_seed43_all_treatment_at_control | 3.292378 | 5.241991 | 2.946260 | -2.247349 | 0/0 | 58.999893 | 0 |
| damping097_fold2_seed43_all_control_at_treatment | 3.058395 | 4.884697 | 2.596564 | -2.223953 | 0/0 | 60.789558 | 2516 |
| damping097_fold2_seed43_easy_control_original | 0.371469 | 0.601697 | 0.101390 | -2.801148 | 0/0 | 26.888359 | 0 |
| damping097_fold2_seed43_easy_treatment_original | 0.318491 | 0.514958 | 0.086261 | -2.672612 | 0/0 | 23.088523 | 0 |
| damping097_fold2_seed43_easy_control_common | 0.371469 | 0.601697 | 0.101390 | -2.801148 | 0/0 | 26.888359 | 0 |
| damping097_fold2_seed43_easy_treatment_common | 0.318491 | 0.514958 | 0.086261 | -2.672612 | 0/0 | 23.088523 | 0 |
| damping097_fold2_seed43_easy_treatment_at_control | 0.378451 | 0.614402 | 0.102361 | -2.717724 | 0/0 | 26.888359 | 5353 |
| damping097_fold2_seed43_easy_control_at_treatment | 0.317540 | 0.519951 | 0.091240 | -2.727885 | 0/0 | 23.088523 | 11 |

## Ranking and Coverage Components

Both anchors retained. Positive ranks favor the new ordering. Units: pp of CV-normalized ADE.
Exact accounting, not unique causal mediation. Common-pool and full-anchor effects both retained.

| Group | Subset | Component | Mean (pp) | Conditional 95% CI (pp) |
|---|---|---|---:|---|
| neural_fold0_seed17_all | all | full_total | 0.128337 | [-0.121370, 0.365832] |
| neural_fold0_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | all | total | 0.128337 | [-0.121370, 0.365832] |
| neural_fold0_seed17_all | all | ranking_at_control_count | 0.202144 | [-0.006397, 0.408672] |
| neural_fold0_seed17_all | all | ranking_at_treatment_count | 0.209551 | [0.032870, 0.388776] |
| neural_fold0_seed17_all | all | coverage_with_treatment_ranking | -0.073806 | [-0.188690, 0.025306] |
| neural_fold0_seed17_all | all | coverage_with_control_ranking | -0.081213 | [-0.202348, 0.023450] |
| neural_fold0_seed17_all | easy | full_total | -0.247562 | [-0.626659, 0.083854] |
| neural_fold0_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | easy | total | -0.247562 | [-0.626659, 0.083854] |
| neural_fold0_seed17_all | easy | ranking_at_control_count | -0.249114 | [-0.628067, 0.077685] |
| neural_fold0_seed17_all | easy | ranking_at_treatment_count | -0.284061 | [-0.641463, 0.012915] |
| neural_fold0_seed17_all | easy | coverage_with_treatment_ranking | 0.001552 | [-0.017377, 0.023075] |
| neural_fold0_seed17_all | easy | coverage_with_control_ranking | 0.036499 | [-0.003921, 0.082929] |
| neural_fold0_seed17_all | hard | full_total | 0.211612 | [-0.032763, 0.453632] |
| neural_fold0_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | hard | total | 0.211612 | [-0.032763, 0.453632] |
| neural_fold0_seed17_all | hard | ranking_at_control_count | 0.312965 | [0.128290, 0.508243] |
| neural_fold0_seed17_all | hard | ranking_at_treatment_count | 0.391723 | [0.201856, 0.575022] |
| neural_fold0_seed17_all | hard | coverage_with_treatment_ranking | -0.101353 | [-0.254812, 0.028652] |
| neural_fold0_seed17_all | hard | coverage_with_control_ranking | -0.180111 | [-0.383646, -0.002059] |
| neural_fold0_seed17_easy | all | full_total | -0.093877 | [-0.171284, -0.034175] |
| neural_fold0_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | all | total | -0.093877 | [-0.171284, -0.034175] |
| neural_fold0_seed17_easy | all | ranking_at_control_count | -0.086940 | [-0.166065, -0.029680] |
| neural_fold0_seed17_easy | all | ranking_at_treatment_count | -0.085828 | [-0.164592, -0.028347] |
| neural_fold0_seed17_easy | all | coverage_with_treatment_ranking | -0.006937 | [-0.018291, 0.002408] |
| neural_fold0_seed17_easy | all | coverage_with_control_ranking | -0.008049 | [-0.017542, -0.000313] |
| neural_fold0_seed17_easy | easy | full_total | -0.058626 | [-0.164102, 0.054164] |
| neural_fold0_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | easy | total | -0.058626 | [-0.164102, 0.054164] |
| neural_fold0_seed17_easy | easy | ranking_at_control_count | 0.015052 | [-0.018454, 0.045847] |
| neural_fold0_seed17_easy | easy | ranking_at_treatment_count | 0.016707 | [0.000565, 0.033690] |
| neural_fold0_seed17_easy | easy | coverage_with_treatment_ranking | -0.073678 | [-0.196896, 0.056198] |
| neural_fold0_seed17_easy | easy | coverage_with_control_ranking | -0.075333 | [-0.180987, 0.034846] |
| neural_fold0_seed17_easy | hard | full_total | -0.095469 | [-0.193388, -0.013470] |
| neural_fold0_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | hard | total | -0.095469 | [-0.193388, -0.013470] |
| neural_fold0_seed17_easy | hard | ranking_at_control_count | -0.093665 | [-0.191905, -0.012491] |
| neural_fold0_seed17_easy | hard | ranking_at_treatment_count | -0.090869 | [-0.185166, -0.013300] |
| neural_fold0_seed17_easy | hard | coverage_with_treatment_ranking | -0.001805 | [-0.004416, 0.000067] |
| neural_fold0_seed17_easy | hard | coverage_with_control_ranking | -0.004601 | [-0.013063, 0.000000] |
| damping097_fold0_seed17_all | all | full_total | 1.021186 | [0.648899, 1.312959] |
| damping097_fold0_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | all | total | 1.021186 | [0.648899, 1.312959] |
| damping097_fold0_seed17_all | all | ranking_at_control_count | 0.641331 | [0.349466, 0.907819] |
| damping097_fold0_seed17_all | all | ranking_at_treatment_count | 0.574652 | [0.432118, 0.714572] |
| damping097_fold0_seed17_all | all | coverage_with_treatment_ranking | 0.379855 | [0.244829, 0.517360] |
| damping097_fold0_seed17_all | all | coverage_with_control_ranking | 0.446534 | [0.193514, 0.628777] |
| damping097_fold0_seed17_all | easy | full_total | -0.412163 | [-1.023014, -0.006527] |
| damping097_fold0_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | easy | total | -0.412163 | [-1.023014, -0.006527] |
| damping097_fold0_seed17_all | easy | ranking_at_control_count | -0.013343 | [-0.086082, 0.049759] |
| damping097_fold0_seed17_all | easy | ranking_at_treatment_count | -0.106386 | [-0.339024, 0.059291] |
| damping097_fold0_seed17_all | easy | coverage_with_treatment_ranking | -0.398821 | [-0.974735, -0.020801] |
| damping097_fold0_seed17_all | easy | coverage_with_control_ranking | -0.305777 | [-0.712689, 0.008178] |
| damping097_fold0_seed17_all | hard | full_total | 1.253846 | [0.793211, 1.664098] |
| damping097_fold0_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | hard | total | 1.253846 | [0.793211, 1.664098] |
| damping097_fold0_seed17_all | hard | ranking_at_control_count | 0.748221 | [0.270224, 1.153626] |
| damping097_fold0_seed17_all | hard | ranking_at_treatment_count | 0.660853 | [0.389493, 0.917506] |
| damping097_fold0_seed17_all | hard | coverage_with_treatment_ranking | 0.505625 | [0.391789, 0.615374] |
| damping097_fold0_seed17_all | hard | coverage_with_control_ranking | 0.592993 | [0.355373, 0.784143] |
| damping097_fold0_seed17_easy | all | full_total | -0.323873 | [-0.442150, -0.226528] |
| damping097_fold0_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | all | total | -0.323873 | [-0.442150, -0.226528] |
| damping097_fold0_seed17_easy | all | ranking_at_control_count | -0.173451 | [-0.259161, -0.103767] |
| damping097_fold0_seed17_easy | all | ranking_at_treatment_count | -0.128477 | [-0.197072, -0.069976] |
| damping097_fold0_seed17_easy | all | coverage_with_treatment_ranking | -0.150422 | [-0.205771, -0.102282] |
| damping097_fold0_seed17_easy | all | coverage_with_control_ranking | -0.195395 | [-0.285580, -0.118703] |
| damping097_fold0_seed17_easy | easy | full_total | 0.012827 | [-0.008950, 0.037514] |
| damping097_fold0_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | easy | total | 0.012827 | [-0.008950, 0.037514] |
| damping097_fold0_seed17_easy | easy | ranking_at_control_count | -0.013784 | [-0.041125, 0.018639] |
| damping097_fold0_seed17_easy | easy | ranking_at_treatment_count | -0.026413 | [-0.053520, -0.006434] |
| damping097_fold0_seed17_easy | easy | coverage_with_treatment_ranking | 0.026611 | [-0.007487, 0.060614] |
| damping097_fold0_seed17_easy | easy | coverage_with_control_ranking | 0.039241 | [0.008761, 0.073924] |
| damping097_fold0_seed17_easy | hard | full_total | -0.349766 | [-0.503725, -0.233750] |
| damping097_fold0_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | hard | total | -0.349766 | [-0.503725, -0.233750] |
| damping097_fold0_seed17_easy | hard | ranking_at_control_count | -0.217187 | [-0.343091, -0.106377] |
| damping097_fold0_seed17_easy | hard | ranking_at_treatment_count | -0.170196 | [-0.248772, -0.097866] |
| damping097_fold0_seed17_easy | hard | coverage_with_treatment_ranking | -0.132579 | [-0.192608, -0.078908] |
| damping097_fold0_seed17_easy | hard | coverage_with_control_ranking | -0.179570 | [-0.303248, -0.090092] |
| neural_fold0_seed29_all | all | full_total | 0.253927 | [0.035531, 0.465172] |
| neural_fold0_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | all | total | 0.253927 | [0.035531, 0.465172] |
| neural_fold0_seed29_all | all | ranking_at_control_count | 0.313061 | [0.215549, 0.411481] |
| neural_fold0_seed29_all | all | ranking_at_treatment_count | 0.302898 | [0.185022, 0.412952] |
| neural_fold0_seed29_all | all | coverage_with_treatment_ranking | -0.059134 | [-0.196361, 0.067404] |
| neural_fold0_seed29_all | all | coverage_with_control_ranking | -0.048970 | [-0.233801, 0.111804] |
| neural_fold0_seed29_all | easy | full_total | -0.388192 | [-0.602495, -0.181315] |
| neural_fold0_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | easy | total | -0.388192 | [-0.602495, -0.181315] |
| neural_fold0_seed29_all | easy | ranking_at_control_count | -0.319833 | [-0.498329, -0.156389] |
| neural_fold0_seed29_all | easy | ranking_at_treatment_count | -0.458200 | [-0.827488, -0.147602] |
| neural_fold0_seed29_all | easy | coverage_with_treatment_ranking | -0.068359 | [-0.135802, -0.003506] |
| neural_fold0_seed29_all | easy | coverage_with_control_ranking | 0.070008 | [-0.067374, 0.286060] |
| neural_fold0_seed29_all | hard | full_total | 0.448437 | [0.257090, 0.681104] |
| neural_fold0_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | hard | total | 0.448437 | [0.257090, 0.681104] |
| neural_fold0_seed29_all | hard | ranking_at_control_count | 0.492710 | [0.401245, 0.600681] |
| neural_fold0_seed29_all | hard | ranking_at_treatment_count | 0.481619 | [0.295002, 0.650930] |
| neural_fold0_seed29_all | hard | coverage_with_treatment_ranking | -0.044273 | [-0.194296, 0.096384] |
| neural_fold0_seed29_all | hard | coverage_with_control_ranking | -0.033181 | [-0.236787, 0.144854] |
| neural_fold0_seed29_easy | all | full_total | -0.149641 | [-0.267115, -0.027589] |
| neural_fold0_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | all | total | -0.149641 | [-0.267115, -0.027589] |
| neural_fold0_seed29_easy | all | ranking_at_control_count | -0.102372 | [-0.198369, 0.000331] |
| neural_fold0_seed29_easy | all | ranking_at_treatment_count | -0.092345 | [-0.195632, 0.017129] |
| neural_fold0_seed29_easy | all | coverage_with_treatment_ranking | -0.047269 | [-0.074921, -0.022024] |
| neural_fold0_seed29_easy | all | coverage_with_control_ranking | -0.057296 | [-0.088018, -0.029920] |
| neural_fold0_seed29_easy | easy | full_total | -0.460020 | [-0.608182, -0.307546] |
| neural_fold0_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | easy | total | -0.460020 | [-0.608182, -0.307546] |
| neural_fold0_seed29_easy | easy | ranking_at_control_count | 0.283755 | [0.167545, 0.387129] |
| neural_fold0_seed29_easy | easy | ranking_at_treatment_count | 0.102985 | [0.014645, 0.189364] |
| neural_fold0_seed29_easy | easy | coverage_with_treatment_ranking | -0.743775 | [-0.937234, -0.512932] |
| neural_fold0_seed29_easy | easy | coverage_with_control_ranking | -0.563006 | [-0.722049, -0.395415] |
| neural_fold0_seed29_easy | hard | full_total | -0.126554 | [-0.250056, -0.019504] |
| neural_fold0_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | hard | total | -0.126554 | [-0.250056, -0.019504] |
| neural_fold0_seed29_easy | hard | ranking_at_control_count | -0.114742 | [-0.227764, -0.018815] |
| neural_fold0_seed29_easy | hard | ranking_at_treatment_count | -0.108291 | [-0.212989, -0.014092] |
| neural_fold0_seed29_easy | hard | coverage_with_treatment_ranking | -0.011811 | [-0.026191, -0.000245] |
| neural_fold0_seed29_easy | hard | coverage_with_control_ranking | -0.018263 | [-0.039956, -0.002484] |
| damping097_fold0_seed29_all | all | full_total | 1.159451 | [0.864113, 1.487577] |
| damping097_fold0_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | all | total | 1.159451 | [0.864113, 1.487577] |
| damping097_fold0_seed29_all | all | ranking_at_control_count | 0.659497 | [0.468137, 0.883529] |
| damping097_fold0_seed29_all | all | ranking_at_treatment_count | 0.539963 | [0.390277, 0.733596] |
| damping097_fold0_seed29_all | all | coverage_with_treatment_ranking | 0.499954 | [0.365097, 0.631024] |
| damping097_fold0_seed29_all | all | coverage_with_control_ranking | 0.619488 | [0.445930, 0.774075] |
| damping097_fold0_seed29_all | easy | full_total | -0.280006 | [-0.671724, 0.002341] |
| damping097_fold0_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | easy | total | -0.280006 | [-0.671724, 0.002341] |
| damping097_fold0_seed29_all | easy | ranking_at_control_count | -0.023599 | [-0.211926, 0.113561] |
| damping097_fold0_seed29_all | easy | ranking_at_treatment_count | 0.024815 | [-0.043238, 0.120830] |
| damping097_fold0_seed29_all | easy | coverage_with_treatment_ranking | -0.256408 | [-0.499694, -0.057227] |
| damping097_fold0_seed29_all | easy | coverage_with_control_ranking | -0.304821 | [-0.695276, -0.009030] |
| damping097_fold0_seed29_all | hard | full_total | 1.461061 | [1.139470, 1.844407] |
| damping097_fold0_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | hard | total | 1.461061 | [1.139470, 1.844407] |
| damping097_fold0_seed29_all | hard | ranking_at_control_count | 0.884766 | [0.644170, 1.148951] |
| damping097_fold0_seed29_all | hard | ranking_at_treatment_count | 0.612764 | [0.339954, 0.890886] |
| damping097_fold0_seed29_all | hard | coverage_with_treatment_ranking | 0.576294 | [0.328852, 0.773633] |
| damping097_fold0_seed29_all | hard | coverage_with_control_ranking | 0.848297 | [0.704342, 0.987632] |
| damping097_fold0_seed29_easy | all | full_total | -0.198985 | [-0.273410, -0.126743] |
| damping097_fold0_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | all | total | -0.198985 | [-0.273410, -0.126743] |
| damping097_fold0_seed29_easy | all | ranking_at_control_count | -0.078484 | [-0.121036, -0.043894] |
| damping097_fold0_seed29_easy | all | ranking_at_treatment_count | -0.065819 | [-0.102620, -0.033621] |
| damping097_fold0_seed29_easy | all | coverage_with_treatment_ranking | -0.120502 | [-0.172118, -0.070666] |
| damping097_fold0_seed29_easy | all | coverage_with_control_ranking | -0.133166 | [-0.177492, -0.084120] |
| damping097_fold0_seed29_easy | easy | full_total | 0.009447 | [-0.023048, 0.042670] |
| damping097_fold0_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | easy | total | 0.009447 | [-0.023048, 0.042670] |
| damping097_fold0_seed29_easy | easy | ranking_at_control_count | 0.010228 | [-0.011060, 0.033760] |
| damping097_fold0_seed29_easy | easy | ranking_at_treatment_count | 0.015592 | [-0.007703, 0.052505] |
| damping097_fold0_seed29_easy | easy | coverage_with_treatment_ranking | -0.000781 | [-0.021617, 0.018807] |
| damping097_fold0_seed29_easy | easy | coverage_with_control_ranking | -0.006145 | [-0.032242, 0.025039] |
| damping097_fold0_seed29_easy | hard | full_total | -0.201724 | [-0.278475, -0.133429] |
| damping097_fold0_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | hard | total | -0.201724 | [-0.278475, -0.133429] |
| damping097_fold0_seed29_easy | hard | ranking_at_control_count | -0.131940 | [-0.190408, -0.079003] |
| damping097_fold0_seed29_easy | hard | ranking_at_treatment_count | -0.105987 | [-0.162672, -0.051767] |
| damping097_fold0_seed29_easy | hard | coverage_with_treatment_ranking | -0.069785 | [-0.112826, -0.028869] |
| damping097_fold0_seed29_easy | hard | coverage_with_control_ranking | -0.095737 | [-0.134133, -0.056261] |
| neural_fold0_seed43_all | all | full_total | -0.228702 | [-0.448472, -0.012578] |
| neural_fold0_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | all | total | -0.228702 | [-0.448472, -0.012578] |
| neural_fold0_seed43_all | all | ranking_at_control_count | -0.081004 | [-0.181638, 0.028022] |
| neural_fold0_seed43_all | all | ranking_at_treatment_count | -0.062270 | [-0.150058, 0.058080] |
| neural_fold0_seed43_all | all | coverage_with_treatment_ranking | -0.147699 | [-0.305501, 0.005920] |
| neural_fold0_seed43_all | all | coverage_with_control_ranking | -0.166433 | [-0.345251, 0.011839] |
| neural_fold0_seed43_all | easy | full_total | -0.112912 | [-0.253801, 0.087523] |
| neural_fold0_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | easy | total | -0.112912 | [-0.253801, 0.087523] |
| neural_fold0_seed43_all | easy | ranking_at_control_count | -0.085449 | [-0.236297, 0.130893] |
| neural_fold0_seed43_all | easy | ranking_at_treatment_count | -0.238477 | [-0.629095, 0.095511] |
| neural_fold0_seed43_all | easy | coverage_with_treatment_ranking | -0.027463 | [-0.068337, 0.009879] |
| neural_fold0_seed43_all | easy | coverage_with_control_ranking | 0.125565 | [-0.048873, 0.437050] |
| neural_fold0_seed43_all | hard | full_total | -0.332522 | [-0.628499, -0.061107] |
| neural_fold0_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | hard | total | -0.332522 | [-0.628499, -0.061107] |
| neural_fold0_seed43_all | hard | ranking_at_control_count | -0.089290 | [-0.276844, 0.102485] |
| neural_fold0_seed43_all | hard | ranking_at_treatment_count | -0.162463 | [-0.339268, 0.016133] |
| neural_fold0_seed43_all | hard | coverage_with_treatment_ranking | -0.243231 | [-0.461906, -0.028691] |
| neural_fold0_seed43_all | hard | coverage_with_control_ranking | -0.170059 | [-0.410342, 0.069554] |
| neural_fold0_seed43_easy | all | full_total | -0.077133 | [-0.171584, -0.013145] |
| neural_fold0_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | all | total | -0.077133 | [-0.171584, -0.013145] |
| neural_fold0_seed43_easy | all | ranking_at_control_count | -0.079666 | [-0.174610, -0.019956] |
| neural_fold0_seed43_easy | all | ranking_at_treatment_count | -0.079701 | [-0.174013, -0.020722] |
| neural_fold0_seed43_easy | all | coverage_with_treatment_ranking | 0.002533 | [-0.006151, 0.010996] |
| neural_fold0_seed43_easy | all | coverage_with_control_ranking | 0.002568 | [-0.006519, 0.012464] |
| neural_fold0_seed43_easy | easy | full_total | -0.120389 | [-0.267778, 0.053451] |
| neural_fold0_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | easy | total | -0.120389 | [-0.267778, 0.053451] |
| neural_fold0_seed43_easy | easy | ranking_at_control_count | -0.205065 | [-0.332613, -0.052612] |
| neural_fold0_seed43_easy | easy | ranking_at_treatment_count | -0.186173 | [-0.323697, -0.022584] |
| neural_fold0_seed43_easy | easy | coverage_with_treatment_ranking | 0.084676 | [-0.070515, 0.275052] |
| neural_fold0_seed43_easy | easy | coverage_with_control_ranking | 0.065784 | [-0.049687, 0.195499] |
| neural_fold0_seed43_easy | hard | full_total | -0.083072 | [-0.195368, -0.010458] |
| neural_fold0_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | hard | total | -0.083072 | [-0.195368, -0.010458] |
| neural_fold0_seed43_easy | hard | ranking_at_control_count | -0.084251 | [-0.196647, -0.011434] |
| neural_fold0_seed43_easy | hard | ranking_at_treatment_count | -0.082573 | [-0.196365, -0.008901] |
| neural_fold0_seed43_easy | hard | coverage_with_treatment_ranking | 0.001179 | [0.000317, 0.002080] |
| neural_fold0_seed43_easy | hard | coverage_with_control_ranking | -0.000499 | [-0.004698, 0.002742] |
| damping097_fold0_seed43_all | all | full_total | 0.890057 | [0.644066, 1.124177] |
| damping097_fold0_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | all | total | 0.890057 | [0.644066, 1.124177] |
| damping097_fold0_seed43_all | all | ranking_at_control_count | 0.521898 | [0.292906, 0.735143] |
| damping097_fold0_seed43_all | all | ranking_at_treatment_count | 0.390815 | [0.140648, 0.605840] |
| damping097_fold0_seed43_all | all | coverage_with_treatment_ranking | 0.368160 | [0.254123, 0.489847] |
| damping097_fold0_seed43_all | all | coverage_with_control_ranking | 0.499242 | [0.377350, 0.637608] |
| damping097_fold0_seed43_all | easy | full_total | -0.425454 | [-0.953386, -0.076544] |
| damping097_fold0_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | easy | total | -0.425454 | [-0.953386, -0.076544] |
| damping097_fold0_seed43_all | easy | ranking_at_control_count | -0.002768 | [-0.270089, 0.233837] |
| damping097_fold0_seed43_all | easy | ranking_at_treatment_count | 0.041532 | [-0.024056, 0.106702] |
| damping097_fold0_seed43_all | easy | coverage_with_treatment_ranking | -0.422687 | [-0.803891, -0.113769] |
| damping097_fold0_seed43_all | easy | coverage_with_control_ranking | -0.466986 | [-1.006214, -0.097089] |
| damping097_fold0_seed43_all | hard | full_total | 1.045994 | [0.669807, 1.387604] |
| damping097_fold0_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | hard | total | 1.045994 | [0.669807, 1.387604] |
| damping097_fold0_seed43_all | hard | ranking_at_control_count | 0.647105 | [0.404648, 0.890545] |
| damping097_fold0_seed43_all | hard | ranking_at_treatment_count | 0.430740 | [0.086678, 0.725726] |
| damping097_fold0_seed43_all | hard | coverage_with_treatment_ranking | 0.398889 | [0.196814, 0.576419] |
| damping097_fold0_seed43_all | hard | coverage_with_control_ranking | 0.615255 | [0.476751, 0.762046] |
| damping097_fold0_seed43_easy | all | full_total | -0.324057 | [-0.399469, -0.249372] |
| damping097_fold0_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | all | total | -0.324057 | [-0.399469, -0.249372] |
| damping097_fold0_seed43_easy | all | ranking_at_control_count | -0.187252 | [-0.220238, -0.150457] |
| damping097_fold0_seed43_easy | all | ranking_at_treatment_count | -0.159741 | [-0.221490, -0.113158] |
| damping097_fold0_seed43_easy | all | coverage_with_treatment_ranking | -0.136805 | [-0.194839, -0.075210] |
| damping097_fold0_seed43_easy | all | coverage_with_control_ranking | -0.164316 | [-0.216079, -0.110221] |
| damping097_fold0_seed43_easy | easy | full_total | 0.055519 | [0.006292, 0.113475] |
| damping097_fold0_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | easy | total | 0.055519 | [0.006292, 0.113475] |
| damping097_fold0_seed43_easy | easy | ranking_at_control_count | 0.003481 | [-0.025340, 0.030771] |
| damping097_fold0_seed43_easy | easy | ranking_at_treatment_count | 0.055620 | [0.002521, 0.116621] |
| damping097_fold0_seed43_easy | easy | coverage_with_treatment_ranking | 0.052038 | [0.014241, 0.103346] |
| damping097_fold0_seed43_easy | easy | coverage_with_control_ranking | -0.000101 | [-0.014925, 0.012763] |
| damping097_fold0_seed43_easy | hard | full_total | -0.439382 | [-0.572675, -0.317484] |
| damping097_fold0_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | hard | total | -0.439382 | [-0.572675, -0.317484] |
| damping097_fold0_seed43_easy | hard | ranking_at_control_count | -0.322614 | [-0.469739, -0.218234] |
| damping097_fold0_seed43_easy | hard | ranking_at_treatment_count | -0.286178 | [-0.451443, -0.164347] |
| damping097_fold0_seed43_easy | hard | coverage_with_treatment_ranking | -0.116768 | [-0.178270, -0.060040] |
| damping097_fold0_seed43_easy | hard | coverage_with_control_ranking | -0.153203 | [-0.221733, -0.084839] |
| neural_fold1_seed17_all | all | full_total | 0.765740 | [0.337213, 1.233526] |
| neural_fold1_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | all | total | 0.765740 | [0.337213, 1.233526] |
| neural_fold1_seed17_all | all | ranking_at_control_count | 0.203250 | [-0.035629, 0.465400] |
| neural_fold1_seed17_all | all | ranking_at_treatment_count | 0.206793 | [-0.046474, 0.453289] |
| neural_fold1_seed17_all | all | coverage_with_treatment_ranking | 0.562490 | [0.290610, 0.828387] |
| neural_fold1_seed17_all | all | coverage_with_control_ranking | 0.558947 | [0.258056, 0.838282] |
| neural_fold1_seed17_all | easy | full_total | -0.704880 | [-1.034793, -0.376760] |
| neural_fold1_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | easy | total | -0.704880 | [-1.034793, -0.376760] |
| neural_fold1_seed17_all | easy | ranking_at_control_count | -0.348039 | [-0.542478, -0.171519] |
| neural_fold1_seed17_all | easy | ranking_at_treatment_count | -0.382601 | [-0.612480, -0.163569] |
| neural_fold1_seed17_all | easy | coverage_with_treatment_ranking | -0.356841 | [-0.642693, -0.088224] |
| neural_fold1_seed17_all | easy | coverage_with_control_ranking | -0.322279 | [-0.609044, -0.104887] |
| neural_fold1_seed17_all | hard | full_total | 1.048078 | [0.461010, 1.620034] |
| neural_fold1_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | hard | total | 1.048078 | [0.461010, 1.620034] |
| neural_fold1_seed17_all | hard | ranking_at_control_count | 0.564663 | [0.237827, 0.868612] |
| neural_fold1_seed17_all | hard | ranking_at_treatment_count | 0.567378 | [0.233688, 0.869678] |
| neural_fold1_seed17_all | hard | coverage_with_treatment_ranking | 0.483415 | [0.169754, 0.806780] |
| neural_fold1_seed17_all | hard | coverage_with_control_ranking | 0.480700 | [0.198110, 0.785140] |
| neural_fold1_seed17_easy | all | full_total | -0.072252 | [-0.145025, -0.006927] |
| neural_fold1_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | all | total | -0.072252 | [-0.145025, -0.006927] |
| neural_fold1_seed17_easy | all | ranking_at_control_count | -0.048537 | [-0.100946, 0.003552] |
| neural_fold1_seed17_easy | all | ranking_at_treatment_count | -0.058031 | [-0.099736, -0.016987] |
| neural_fold1_seed17_easy | all | coverage_with_treatment_ranking | -0.023716 | [-0.052229, 0.002633] |
| neural_fold1_seed17_easy | all | coverage_with_control_ranking | -0.014221 | [-0.054963, 0.020723] |
| neural_fold1_seed17_easy | easy | full_total | -0.063978 | [-0.289402, 0.137973] |
| neural_fold1_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | easy | total | -0.063978 | [-0.289402, 0.137973] |
| neural_fold1_seed17_easy | easy | ranking_at_control_count | 0.090027 | [-0.164322, 0.369470] |
| neural_fold1_seed17_easy | easy | ranking_at_treatment_count | 0.116971 | [-0.109803, 0.348678] |
| neural_fold1_seed17_easy | easy | coverage_with_treatment_ranking | -0.154005 | [-0.533122, 0.224853] |
| neural_fold1_seed17_easy | easy | coverage_with_control_ranking | -0.180950 | [-0.516375, 0.157838] |
| neural_fold1_seed17_easy | hard | full_total | -0.043943 | [-0.120313, 0.022466] |
| neural_fold1_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | hard | total | -0.043943 | [-0.120313, 0.022466] |
| neural_fold1_seed17_easy | hard | ranking_at_control_count | -0.040933 | [-0.111213, 0.022036] |
| neural_fold1_seed17_easy | hard | ranking_at_treatment_count | -0.040435 | [-0.100742, 0.020235] |
| neural_fold1_seed17_easy | hard | coverage_with_treatment_ranking | -0.003011 | [-0.009677, 0.000708] |
| neural_fold1_seed17_easy | hard | coverage_with_control_ranking | -0.003508 | [-0.026210, 0.013438] |
| damping097_fold1_seed17_all | all | full_total | 0.695091 | [0.535034, 0.862053] |
| damping097_fold1_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | all | total | 0.695091 | [0.535034, 0.862053] |
| damping097_fold1_seed17_all | all | ranking_at_control_count | 0.444641 | [0.293654, 0.611761] |
| damping097_fold1_seed17_all | all | ranking_at_treatment_count | 0.403476 | [0.254951, 0.546749] |
| damping097_fold1_seed17_all | all | coverage_with_treatment_ranking | 0.250450 | [0.168771, 0.369206] |
| damping097_fold1_seed17_all | all | coverage_with_control_ranking | 0.291615 | [0.194913, 0.401347] |
| damping097_fold1_seed17_all | easy | full_total | -0.117193 | [-0.388554, 0.106612] |
| damping097_fold1_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | easy | total | -0.117193 | [-0.388554, 0.106612] |
| damping097_fold1_seed17_all | easy | ranking_at_control_count | -0.014060 | [-0.160097, 0.116770] |
| damping097_fold1_seed17_all | easy | ranking_at_treatment_count | -0.087818 | [-0.359574, 0.102502] |
| damping097_fold1_seed17_all | easy | coverage_with_treatment_ranking | -0.103133 | [-0.244934, 0.019446] |
| damping097_fold1_seed17_all | easy | coverage_with_control_ranking | -0.029375 | [-0.116779, 0.044825] |
| damping097_fold1_seed17_all | hard | full_total | 1.009790 | [0.720678, 1.283984] |
| damping097_fold1_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | hard | total | 1.009790 | [0.720678, 1.283984] |
| damping097_fold1_seed17_all | hard | ranking_at_control_count | 0.638848 | [0.422580, 0.869877] |
| damping097_fold1_seed17_all | hard | ranking_at_treatment_count | 0.393319 | [-0.198032, 0.765804] |
| damping097_fold1_seed17_all | hard | coverage_with_treatment_ranking | 0.370942 | [0.211847, 0.580273] |
| damping097_fold1_seed17_all | hard | coverage_with_control_ranking | 0.616471 | [0.300807, 1.028880] |
| damping097_fold1_seed17_easy | all | full_total | -1.206497 | [-1.465559, -0.978564] |
| damping097_fold1_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | all | total | -1.206497 | [-1.465559, -0.978564] |
| damping097_fold1_seed17_easy | all | ranking_at_control_count | 0.025594 | [-0.123100, 0.203262] |
| damping097_fold1_seed17_easy | all | ranking_at_treatment_count | -0.072388 | [-0.118862, -0.030735] |
| damping097_fold1_seed17_easy | all | coverage_with_treatment_ranking | -1.232091 | [-1.455808, -0.995515] |
| damping097_fold1_seed17_easy | all | coverage_with_control_ranking | -1.134109 | [-1.369242, -0.917702] |
| damping097_fold1_seed17_easy | easy | full_total | 0.534368 | [0.179113, 0.924264] |
| damping097_fold1_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | easy | total | 0.534368 | [0.179113, 0.924264] |
| damping097_fold1_seed17_easy | easy | ranking_at_control_count | -0.014881 | [-0.054660, 0.024081] |
| damping097_fold1_seed17_easy | easy | ranking_at_treatment_count | 0.058088 | [-0.006537, 0.138864] |
| damping097_fold1_seed17_easy | easy | coverage_with_treatment_ranking | 0.549249 | [0.220911, 0.901999] |
| damping097_fold1_seed17_easy | easy | coverage_with_control_ranking | 0.476280 | [0.112167, 0.898842] |
| damping097_fold1_seed17_easy | hard | full_total | -1.560033 | [-2.299247, -1.026205] |
| damping097_fold1_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | hard | total | -1.560033 | [-2.299247, -1.026205] |
| damping097_fold1_seed17_easy | hard | ranking_at_control_count | -0.049058 | [-0.180130, 0.118980] |
| damping097_fold1_seed17_easy | hard | ranking_at_treatment_count | -0.080285 | [-0.139523, -0.031251] |
| damping097_fold1_seed17_easy | hard | coverage_with_treatment_ranking | -1.510975 | [-2.295082, -0.934730] |
| damping097_fold1_seed17_easy | hard | coverage_with_control_ranking | -1.479748 | [-2.226569, -0.961812] |
| neural_fold1_seed29_all | all | full_total | 0.793362 | [0.283241, 1.238624] |
| neural_fold1_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | all | total | 0.793362 | [0.283241, 1.238624] |
| neural_fold1_seed29_all | all | ranking_at_control_count | 0.279521 | [-0.142991, 0.630995] |
| neural_fold1_seed29_all | all | ranking_at_treatment_count | 0.386543 | [0.111528, 0.629547] |
| neural_fold1_seed29_all | all | coverage_with_treatment_ranking | 0.513841 | [0.396248, 0.643228] |
| neural_fold1_seed29_all | all | coverage_with_control_ranking | 0.406819 | [0.174165, 0.630239] |
| neural_fold1_seed29_all | easy | full_total | -0.747962 | [-1.199548, -0.346238] |
| neural_fold1_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | easy | total | -0.747962 | [-1.199548, -0.346238] |
| neural_fold1_seed29_all | easy | ranking_at_control_count | -0.593148 | [-0.918782, -0.292453] |
| neural_fold1_seed29_all | easy | ranking_at_treatment_count | -0.492233 | [-0.874142, -0.200974] |
| neural_fold1_seed29_all | easy | coverage_with_treatment_ranking | -0.154814 | [-0.350586, -0.002166] |
| neural_fold1_seed29_all | easy | coverage_with_control_ranking | -0.255729 | [-0.499493, -0.062937] |
| neural_fold1_seed29_all | hard | full_total | 1.213222 | [0.716800, 1.700974] |
| neural_fold1_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | hard | total | 1.213222 | [0.716800, 1.700974] |
| neural_fold1_seed29_all | hard | ranking_at_control_count | 0.608363 | [0.069503, 1.069271] |
| neural_fold1_seed29_all | hard | ranking_at_treatment_count | 0.831619 | [0.468459, 1.150388] |
| neural_fold1_seed29_all | hard | coverage_with_treatment_ranking | 0.604859 | [0.419101, 0.804974] |
| neural_fold1_seed29_all | hard | coverage_with_control_ranking | 0.381603 | [0.189591, 0.603351] |
| neural_fold1_seed29_easy | all | full_total | -0.101284 | [-0.369485, 0.144472] |
| neural_fold1_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | all | total | -0.101284 | [-0.369485, 0.144472] |
| neural_fold1_seed29_easy | all | ranking_at_control_count | -0.047155 | [-0.251284, 0.188083] |
| neural_fold1_seed29_easy | all | ranking_at_treatment_count | -0.102362 | [-0.238489, 0.009428] |
| neural_fold1_seed29_easy | all | coverage_with_treatment_ranking | -0.054128 | [-0.142595, 0.008914] |
| neural_fold1_seed29_easy | all | coverage_with_control_ranking | 0.001078 | [-0.136310, 0.152107] |
| neural_fold1_seed29_easy | easy | full_total | 0.507030 | [0.069128, 0.928809] |
| neural_fold1_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | easy | total | 0.507030 | [0.069128, 0.928809] |
| neural_fold1_seed29_easy | easy | ranking_at_control_count | 0.596184 | [0.233887, 1.057529] |
| neural_fold1_seed29_easy | easy | ranking_at_treatment_count | 0.587437 | [0.250031, 1.022013] |
| neural_fold1_seed29_easy | easy | coverage_with_treatment_ranking | -0.089155 | [-0.473059, 0.254331] |
| neural_fold1_seed29_easy | easy | coverage_with_control_ranking | -0.080408 | [-0.445714, 0.244683] |
| neural_fold1_seed29_easy | hard | full_total | -0.196681 | [-0.439698, -0.023679] |
| neural_fold1_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | hard | total | -0.196681 | [-0.439698, -0.023679] |
| neural_fold1_seed29_easy | hard | ranking_at_control_count | -0.186559 | [-0.415657, -0.021852] |
| neural_fold1_seed29_easy | hard | ranking_at_treatment_count | -0.148678 | [-0.319126, -0.021407] |
| neural_fold1_seed29_easy | hard | coverage_with_treatment_ranking | -0.010122 | [-0.024867, -0.000428] |
| neural_fold1_seed29_easy | hard | coverage_with_control_ranking | -0.048003 | [-0.123675, -0.001279] |
| damping097_fold1_seed29_all | all | full_total | 0.510463 | [0.363634, 0.640136] |
| damping097_fold1_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | all | total | 0.510463 | [0.363634, 0.640136] |
| damping097_fold1_seed29_all | all | ranking_at_control_count | 0.314982 | [0.141468, 0.432384] |
| damping097_fold1_seed29_all | all | ranking_at_treatment_count | 0.329576 | [0.208598, 0.423025] |
| damping097_fold1_seed29_all | all | coverage_with_treatment_ranking | 0.195481 | [0.121335, 0.269925] |
| damping097_fold1_seed29_all | all | coverage_with_control_ranking | 0.180887 | [0.128204, 0.235702] |
| damping097_fold1_seed29_all | easy | full_total | -0.023118 | [-0.115911, 0.058345] |
| damping097_fold1_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | easy | total | -0.023118 | [-0.115911, 0.058345] |
| damping097_fold1_seed29_all | easy | ranking_at_control_count | 0.016846 | [-0.026092, 0.065419] |
| damping097_fold1_seed29_all | easy | ranking_at_treatment_count | 0.069292 | [-0.015046, 0.218489] |
| damping097_fold1_seed29_all | easy | coverage_with_treatment_ranking | -0.039964 | [-0.103576, 0.003038] |
| damping097_fold1_seed29_all | easy | coverage_with_control_ranking | -0.092410 | [-0.209981, 0.005086] |
| damping097_fold1_seed29_all | hard | full_total | 0.997240 | [0.585204, 1.534125] |
| damping097_fold1_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | hard | total | 0.997240 | [0.585204, 1.534125] |
| damping097_fold1_seed29_all | hard | ranking_at_control_count | 0.620611 | [0.281137, 0.981917] |
| damping097_fold1_seed29_all | hard | ranking_at_treatment_count | 0.616310 | [0.373519, 0.928215] |
| damping097_fold1_seed29_all | hard | coverage_with_treatment_ranking | 0.376629 | [0.188558, 0.606851] |
| damping097_fold1_seed29_all | hard | coverage_with_control_ranking | 0.380930 | [0.192255, 0.616442] |
| damping097_fold1_seed29_easy | all | full_total | -0.986879 | [-1.345043, -0.572049] |
| damping097_fold1_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | all | total | -0.986879 | [-1.345043, -0.572049] |
| damping097_fold1_seed29_easy | all | ranking_at_control_count | -0.131100 | [-0.303272, 0.001343] |
| damping097_fold1_seed29_easy | all | ranking_at_treatment_count | -0.032389 | [-0.111066, 0.036905] |
| damping097_fold1_seed29_easy | all | coverage_with_treatment_ranking | -0.855779 | [-1.149924, -0.528424] |
| damping097_fold1_seed29_easy | all | coverage_with_control_ranking | -0.954490 | [-1.249031, -0.598107] |
| damping097_fold1_seed29_easy | easy | full_total | 0.729222 | [0.097902, 1.640182] |
| damping097_fold1_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | easy | total | 0.729222 | [0.097902, 1.640182] |
| damping097_fold1_seed29_easy | easy | ranking_at_control_count | 0.099285 | [-0.005529, 0.269641] |
| damping097_fold1_seed29_easy | easy | ranking_at_treatment_count | 0.084759 | [0.011394, 0.186698] |
| damping097_fold1_seed29_easy | easy | coverage_with_treatment_ranking | 0.629938 | [0.078939, 1.382015] |
| damping097_fold1_seed29_easy | easy | coverage_with_control_ranking | 0.644463 | [0.052377, 1.465438] |
| damping097_fold1_seed29_easy | hard | full_total | -1.566451 | [-2.405082, -0.961254] |
| damping097_fold1_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | hard | total | -1.566451 | [-2.405082, -0.961254] |
| damping097_fold1_seed29_easy | hard | ranking_at_control_count | -0.315375 | [-0.526629, -0.117036] |
| damping097_fold1_seed29_easy | hard | ranking_at_treatment_count | -0.020749 | [-0.136845, 0.108837] |
| damping097_fold1_seed29_easy | hard | coverage_with_treatment_ranking | -1.251076 | [-1.986839, -0.734547] |
| damping097_fold1_seed29_easy | hard | coverage_with_control_ranking | -1.545702 | [-2.457740, -0.935744] |
| neural_fold1_seed43_all | all | full_total | 0.519880 | [0.117146, 0.921262] |
| neural_fold1_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | all | total | 0.519880 | [0.117146, 0.921262] |
| neural_fold1_seed43_all | all | ranking_at_control_count | 0.418695 | [0.154575, 0.732792] |
| neural_fold1_seed43_all | all | ranking_at_treatment_count | 0.355035 | [0.092479, 0.649596] |
| neural_fold1_seed43_all | all | coverage_with_treatment_ranking | 0.101186 | [-0.128547, 0.333935] |
| neural_fold1_seed43_all | all | coverage_with_control_ranking | 0.164846 | [-0.087652, 0.411343] |
| neural_fold1_seed43_all | easy | full_total | -0.186468 | [-0.390525, 0.007425] |
| neural_fold1_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | easy | total | -0.186468 | [-0.390525, 0.007425] |
| neural_fold1_seed43_all | easy | ranking_at_control_count | -0.280602 | [-0.514117, -0.082199] |
| neural_fold1_seed43_all | easy | ranking_at_treatment_count | -0.229481 | [-0.514780, 0.002179] |
| neural_fold1_seed43_all | easy | coverage_with_treatment_ranking | 0.094134 | [-0.113147, 0.317810] |
| neural_fold1_seed43_all | easy | coverage_with_control_ranking | 0.043013 | [-0.132405, 0.288891] |
| neural_fold1_seed43_all | hard | full_total | 0.821602 | [0.396804, 1.276334] |
| neural_fold1_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | hard | total | 0.821602 | [0.396804, 1.276334] |
| neural_fold1_seed43_all | hard | ranking_at_control_count | 0.785101 | [0.409913, 1.187930] |
| neural_fold1_seed43_all | hard | ranking_at_treatment_count | 0.737594 | [0.395239, 1.090669] |
| neural_fold1_seed43_all | hard | coverage_with_treatment_ranking | 0.036501 | [-0.161516, 0.222633] |
| neural_fold1_seed43_all | hard | coverage_with_control_ranking | 0.084008 | [-0.140732, 0.299015] |
| neural_fold1_seed43_easy | all | full_total | -0.070115 | [-0.318432, 0.161612] |
| neural_fold1_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | all | total | -0.070115 | [-0.318432, 0.161612] |
| neural_fold1_seed43_easy | all | ranking_at_control_count | 0.036442 | [-0.153308, 0.210697] |
| neural_fold1_seed43_easy | all | ranking_at_treatment_count | -0.061393 | [-0.213134, 0.072167] |
| neural_fold1_seed43_easy | all | coverage_with_treatment_ranking | -0.106557 | [-0.235799, -0.018968] |
| neural_fold1_seed43_easy | all | coverage_with_control_ranking | -0.008722 | [-0.127737, 0.129646] |
| neural_fold1_seed43_easy | easy | full_total | -0.071646 | [-0.332325, 0.225681] |
| neural_fold1_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | easy | total | -0.071646 | [-0.332325, 0.225681] |
| neural_fold1_seed43_easy | easy | ranking_at_control_count | 0.458102 | [0.219041, 0.719773] |
| neural_fold1_seed43_easy | easy | ranking_at_treatment_count | 0.239553 | [0.052545, 0.463543] |
| neural_fold1_seed43_easy | easy | coverage_with_treatment_ranking | -0.529748 | [-0.783271, -0.309080] |
| neural_fold1_seed43_easy | easy | coverage_with_control_ranking | -0.311199 | [-0.628747, 0.086759] |
| neural_fold1_seed43_easy | hard | full_total | -0.153571 | [-0.402215, 0.021930] |
| neural_fold1_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | hard | total | -0.153571 | [-0.402215, 0.021930] |
| neural_fold1_seed43_easy | hard | ranking_at_control_count | -0.051739 | [-0.291507, 0.111451] |
| neural_fold1_seed43_easy | hard | ranking_at_treatment_count | -0.139399 | [-0.340575, 0.007150] |
| neural_fold1_seed43_easy | hard | coverage_with_treatment_ranking | -0.101832 | [-0.260692, -0.005454] |
| neural_fold1_seed43_easy | hard | coverage_with_control_ranking | -0.014172 | [-0.072054, 0.040300] |
| damping097_fold1_seed43_all | all | full_total | 0.742777 | [0.414670, 1.200520] |
| damping097_fold1_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | all | total | 0.742777 | [0.414670, 1.200520] |
| damping097_fold1_seed43_all | all | ranking_at_control_count | 0.536541 | [0.397141, 0.676605] |
| damping097_fold1_seed43_all | all | ranking_at_treatment_count | 0.585732 | [0.429760, 0.776782] |
| damping097_fold1_seed43_all | all | coverage_with_treatment_ranking | 0.206237 | [-0.031471, 0.606293] |
| damping097_fold1_seed43_all | all | coverage_with_control_ranking | 0.157045 | [-0.015870, 0.428382] |
| damping097_fold1_seed43_all | easy | full_total | 0.023310 | [-0.009682, 0.059726] |
| damping097_fold1_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | easy | total | 0.023310 | [-0.009682, 0.059726] |
| damping097_fold1_seed43_all | easy | ranking_at_control_count | 0.186230 | [-0.015343, 0.550678] |
| damping097_fold1_seed43_all | easy | ranking_at_treatment_count | 0.146463 | [-0.006413, 0.421757] |
| damping097_fold1_seed43_all | easy | coverage_with_treatment_ranking | -0.162921 | [-0.494321, 0.009776] |
| damping097_fold1_seed43_all | easy | coverage_with_control_ranking | -0.123154 | [-0.364013, 0.000000] |
| damping097_fold1_seed43_all | hard | full_total | 1.300817 | [0.668106, 2.094570] |
| damping097_fold1_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | hard | total | 1.300817 | [0.668106, 2.094570] |
| damping097_fold1_seed43_all | hard | ranking_at_control_count | 0.744042 | [0.589157, 0.900241] |
| damping097_fold1_seed43_all | hard | ranking_at_treatment_count | 0.943070 | [0.653620, 1.287135] |
| damping097_fold1_seed43_all | hard | coverage_with_treatment_ranking | 0.556775 | [0.020394, 1.246026] |
| damping097_fold1_seed43_all | hard | coverage_with_control_ranking | 0.357747 | [0.008788, 0.839029] |
| damping097_fold1_seed43_easy | all | full_total | -1.502468 | [-1.864500, -1.180661] |
| damping097_fold1_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | all | total | -1.502468 | [-1.864500, -1.180661] |
| damping097_fold1_seed43_easy | all | ranking_at_control_count | -0.187475 | [-0.357623, -0.020051] |
| damping097_fold1_seed43_easy | all | ranking_at_treatment_count | -0.076793 | [-0.197482, 0.049104] |
| damping097_fold1_seed43_easy | all | coverage_with_treatment_ranking | -1.314993 | [-1.636950, -1.021800] |
| damping097_fold1_seed43_easy | all | coverage_with_control_ranking | -1.425675 | [-1.696016, -1.182773] |
| damping097_fold1_seed43_easy | easy | full_total | 0.790858 | [0.168031, 1.583605] |
| damping097_fold1_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | easy | total | 0.790858 | [0.168031, 1.583605] |
| damping097_fold1_seed43_easy | easy | ranking_at_control_count | -0.074809 | [-0.104094, -0.043924] |
| damping097_fold1_seed43_easy | easy | ranking_at_treatment_count | 0.057089 | [-0.043791, 0.199915] |
| damping097_fold1_seed43_easy | easy | coverage_with_treatment_ranking | 0.865668 | [0.229888, 1.668220] |
| damping097_fold1_seed43_easy | easy | coverage_with_control_ranking | 0.733769 | [0.145976, 1.448811] |
| damping097_fold1_seed43_easy | hard | full_total | -2.158982 | [-2.938893, -1.561289] |
| damping097_fold1_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | hard | total | -2.158982 | [-2.938893, -1.561289] |
| damping097_fold1_seed43_easy | hard | ranking_at_control_count | -0.552203 | [-1.053514, -0.181460] |
| damping097_fold1_seed43_easy | hard | ranking_at_treatment_count | -0.186238 | [-0.318856, -0.067291] |
| damping097_fold1_seed43_easy | hard | coverage_with_treatment_ranking | -1.606780 | [-1.959119, -1.258490] |
| damping097_fold1_seed43_easy | hard | coverage_with_control_ranking | -1.972745 | [-2.798831, -1.388294] |
| neural_fold2_seed17_all | all | full_total | -0.336854 | [-0.547280, -0.135055] |
| neural_fold2_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | all | total | -0.336854 | [-0.547280, -0.135055] |
| neural_fold2_seed17_all | all | ranking_at_control_count | -0.411116 | [-0.618412, -0.227182] |
| neural_fold2_seed17_all | all | ranking_at_treatment_count | -0.407224 | [-0.611847, -0.210135] |
| neural_fold2_seed17_all | all | coverage_with_treatment_ranking | 0.074261 | [0.026192, 0.125329] |
| neural_fold2_seed17_all | all | coverage_with_control_ranking | 0.070370 | [0.040397, 0.097551] |
| neural_fold2_seed17_all | easy | full_total | 0.451821 | [0.062649, 0.943945] |
| neural_fold2_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | easy | total | 0.451821 | [0.062649, 0.943945] |
| neural_fold2_seed17_all | easy | ranking_at_control_count | 0.186728 | [-0.048620, 0.434998] |
| neural_fold2_seed17_all | easy | ranking_at_treatment_count | 0.278778 | [0.030983, 0.545975] |
| neural_fold2_seed17_all | easy | coverage_with_treatment_ranking | 0.265094 | [0.008040, 0.592344] |
| neural_fold2_seed17_all | easy | coverage_with_control_ranking | 0.173043 | [-0.120756, 0.516502] |
| neural_fold2_seed17_all | hard | full_total | -0.439278 | [-0.697909, -0.207369] |
| neural_fold2_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | hard | total | -0.439278 | [-0.697909, -0.207369] |
| neural_fold2_seed17_all | hard | ranking_at_control_count | -0.505589 | [-0.755185, -0.281918] |
| neural_fold2_seed17_all | hard | ranking_at_treatment_count | -0.509033 | [-0.761914, -0.266960] |
| neural_fold2_seed17_all | hard | coverage_with_treatment_ranking | 0.066311 | [0.014452, 0.119694] |
| neural_fold2_seed17_all | hard | coverage_with_control_ranking | 0.069755 | [0.021082, 0.110992] |
| neural_fold2_seed17_easy | all | full_total | 0.029143 | [-0.012654, 0.104809] |
| neural_fold2_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | all | total | 0.029143 | [-0.012654, 0.104809] |
| neural_fold2_seed17_easy | all | ranking_at_control_count | 0.030420 | [-0.012787, 0.109104] |
| neural_fold2_seed17_easy | all | ranking_at_treatment_count | 0.046439 | [-0.012490, 0.156349] |
| neural_fold2_seed17_easy | all | coverage_with_treatment_ranking | -0.001277 | [-0.004788, 0.001109] |
| neural_fold2_seed17_easy | all | coverage_with_control_ranking | -0.017296 | [-0.051604, -0.000050] |
| neural_fold2_seed17_easy | easy | full_total | -0.098427 | [-0.244576, 0.002194] |
| neural_fold2_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | easy | total | -0.098427 | [-0.244576, 0.002194] |
| neural_fold2_seed17_easy | easy | ranking_at_control_count | -0.066654 | [-0.145317, 0.001766] |
| neural_fold2_seed17_easy | easy | ranking_at_treatment_count | -0.056447 | [-0.135089, 0.006038] |
| neural_fold2_seed17_easy | easy | coverage_with_treatment_ranking | -0.031773 | [-0.103977, 0.008457] |
| neural_fold2_seed17_easy | easy | coverage_with_control_ranking | -0.041981 | [-0.110336, 0.001827] |
| neural_fold2_seed17_easy | hard | full_total | 0.037543 | [-0.008606, 0.120372] |
| neural_fold2_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | hard | total | 0.037543 | [-0.008606, 0.120372] |
| neural_fold2_seed17_easy | hard | ranking_at_control_count | 0.037602 | [-0.008447, 0.120413] |
| neural_fold2_seed17_easy | hard | ranking_at_treatment_count | 0.037543 | [-0.008606, 0.120372] |
| neural_fold2_seed17_easy | hard | coverage_with_treatment_ranking | -0.000059 | [-0.000176, 0.000000] |
| neural_fold2_seed17_easy | hard | coverage_with_control_ranking | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | all | full_total | 0.694056 | [0.469282, 0.932842] |
| damping097_fold2_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | all | total | 0.694056 | [0.469282, 0.932842] |
| damping097_fold2_seed17_all | all | ranking_at_control_count | 0.320745 | [0.204066, 0.449699] |
| damping097_fold2_seed17_all | all | ranking_at_treatment_count | 0.306865 | [0.137371, 0.478446] |
| damping097_fold2_seed17_all | all | coverage_with_treatment_ranking | 0.373311 | [0.222997, 0.569953] |
| damping097_fold2_seed17_all | all | coverage_with_control_ranking | 0.387192 | [0.257112, 0.536767] |
| damping097_fold2_seed17_all | easy | full_total | -0.081265 | [-0.189205, -0.005980] |
| damping097_fold2_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | easy | total | -0.081265 | [-0.189205, -0.005980] |
| damping097_fold2_seed17_all | easy | ranking_at_control_count | -0.016624 | [-0.054871, 0.009428] |
| damping097_fold2_seed17_all | easy | ranking_at_treatment_count | 0.032780 | [-0.014014, 0.094396] |
| damping097_fold2_seed17_all | easy | coverage_with_treatment_ranking | -0.064642 | [-0.181061, 0.002747] |
| damping097_fold2_seed17_all | easy | coverage_with_control_ranking | -0.114046 | [-0.277320, 0.004005] |
| damping097_fold2_seed17_all | hard | full_total | 0.807132 | [0.586313, 1.036960] |
| damping097_fold2_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | hard | total | 0.807132 | [0.586313, 1.036960] |
| damping097_fold2_seed17_all | hard | ranking_at_control_count | 0.409269 | [0.276023, 0.548834] |
| damping097_fold2_seed17_all | hard | ranking_at_treatment_count | 0.341688 | [0.168158, 0.550647] |
| damping097_fold2_seed17_all | hard | coverage_with_treatment_ranking | 0.397863 | [0.256041, 0.585227] |
| damping097_fold2_seed17_all | hard | coverage_with_control_ranking | 0.465445 | [0.292879, 0.662780] |
| damping097_fold2_seed17_easy | all | full_total | -0.250892 | [-0.342682, -0.151642] |
| damping097_fold2_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | all | total | -0.250892 | [-0.342682, -0.151642] |
| damping097_fold2_seed17_easy | all | ranking_at_control_count | -0.055200 | [-0.091188, -0.021922] |
| damping097_fold2_seed17_easy | all | ranking_at_treatment_count | -0.037626 | [-0.064989, -0.010073] |
| damping097_fold2_seed17_easy | all | coverage_with_treatment_ranking | -0.195692 | [-0.277555, -0.108649] |
| damping097_fold2_seed17_easy | all | coverage_with_control_ranking | -0.213266 | [-0.288976, -0.131301] |
| damping097_fold2_seed17_easy | easy | full_total | -0.206199 | [-0.265531, -0.149169] |
| damping097_fold2_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | easy | total | -0.206199 | [-0.265531, -0.149169] |
| damping097_fold2_seed17_easy | easy | ranking_at_control_count | 0.021828 | [-0.013255, 0.062057] |
| damping097_fold2_seed17_easy | easy | ranking_at_treatment_count | 0.105413 | [0.037986, 0.174930] |
| damping097_fold2_seed17_easy | easy | coverage_with_treatment_ranking | -0.228028 | [-0.290542, -0.163399] |
| damping097_fold2_seed17_easy | easy | coverage_with_control_ranking | -0.311612 | [-0.412001, -0.210357] |
| damping097_fold2_seed17_easy | hard | full_total | -0.146162 | [-0.235556, -0.063329] |
| damping097_fold2_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | hard | total | -0.146162 | [-0.235556, -0.063329] |
| damping097_fold2_seed17_easy | hard | ranking_at_control_count | -0.057692 | [-0.104138, -0.017692] |
| damping097_fold2_seed17_easy | hard | ranking_at_treatment_count | -0.045504 | [-0.081950, -0.012937] |
| damping097_fold2_seed17_easy | hard | coverage_with_treatment_ranking | -0.088470 | [-0.176706, -0.022893] |
| damping097_fold2_seed17_easy | hard | coverage_with_control_ranking | -0.100657 | [-0.154850, -0.049175] |
| neural_fold2_seed29_all | all | full_total | -0.541558 | [-0.706659, -0.357972] |
| neural_fold2_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | all | total | -0.541558 | [-0.706659, -0.357972] |
| neural_fold2_seed29_all | all | ranking_at_control_count | -0.573660 | [-0.732204, -0.400572] |
| neural_fold2_seed29_all | all | ranking_at_treatment_count | -0.613159 | [-0.780573, -0.431570] |
| neural_fold2_seed29_all | all | coverage_with_treatment_ranking | 0.032102 | [0.000593, 0.058006] |
| neural_fold2_seed29_all | all | coverage_with_control_ranking | 0.071601 | [0.010224, 0.123924] |
| neural_fold2_seed29_all | easy | full_total | 0.400059 | [-0.148328, 0.908361] |
| neural_fold2_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | easy | total | 0.400059 | [-0.148328, 0.908361] |
| neural_fold2_seed29_all | easy | ranking_at_control_count | -0.063522 | [-0.644425, 0.463610] |
| neural_fold2_seed29_all | easy | ranking_at_treatment_count | 0.135976 | [-0.352324, 0.553911] |
| neural_fold2_seed29_all | easy | coverage_with_treatment_ranking | 0.463581 | [0.088305, 0.907741] |
| neural_fold2_seed29_all | easy | coverage_with_control_ranking | 0.264083 | [0.047945, 0.504581] |
| neural_fold2_seed29_all | hard | full_total | -0.646418 | [-0.879871, -0.423901] |
| neural_fold2_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | hard | total | -0.646418 | [-0.879871, -0.423901] |
| neural_fold2_seed29_all | hard | ranking_at_control_count | -0.683015 | [-0.903032, -0.464170] |
| neural_fold2_seed29_all | hard | ranking_at_treatment_count | -0.738340 | [-0.958361, -0.518402] |
| neural_fold2_seed29_all | hard | coverage_with_treatment_ranking | 0.036597 | [-0.010366, 0.096949] |
| neural_fold2_seed29_all | hard | coverage_with_control_ranking | 0.091922 | [0.010032, 0.167839] |
| neural_fold2_seed29_easy | all | full_total | -0.031448 | [-0.091989, 0.009388] |
| neural_fold2_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | all | total | -0.031448 | [-0.091989, 0.009388] |
| neural_fold2_seed29_easy | all | ranking_at_control_count | -0.037375 | [-0.095551, 0.007989] |
| neural_fold2_seed29_easy | all | ranking_at_treatment_count | -0.027150 | [-0.092085, 0.023109] |
| neural_fold2_seed29_easy | all | coverage_with_treatment_ranking | 0.005927 | [-0.003472, 0.021782] |
| neural_fold2_seed29_easy | all | coverage_with_control_ranking | -0.004298 | [-0.015249, 0.002796] |
| neural_fold2_seed29_easy | easy | full_total | 0.041616 | [-0.018183, 0.137977] |
| neural_fold2_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | easy | total | 0.041616 | [-0.018183, 0.137977] |
| neural_fold2_seed29_easy | easy | ranking_at_control_count | 0.037746 | [-0.019972, 0.135076] |
| neural_fold2_seed29_easy | easy | ranking_at_treatment_count | 0.046293 | [-0.008658, 0.140611] |
| neural_fold2_seed29_easy | easy | coverage_with_treatment_ranking | 0.003870 | [-0.007557, 0.016046] |
| neural_fold2_seed29_easy | easy | coverage_with_control_ranking | -0.004677 | [-0.012098, 0.001370] |
| neural_fold2_seed29_easy | hard | full_total | -0.059454 | [-0.131670, -0.002548] |
| neural_fold2_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | hard | total | -0.059454 | [-0.131670, -0.002548] |
| neural_fold2_seed29_easy | hard | ranking_at_control_count | -0.072131 | [-0.143954, -0.012919] |
| neural_fold2_seed29_easy | hard | ranking_at_treatment_count | -0.051284 | [-0.120561, -0.001349] |
| neural_fold2_seed29_easy | hard | coverage_with_treatment_ranking | 0.012677 | [0.000000, 0.038028] |
| neural_fold2_seed29_easy | hard | coverage_with_control_ranking | -0.008170 | [-0.023862, 0.000128] |
| damping097_fold2_seed29_all | all | full_total | 0.600580 | [0.354442, 0.891439] |
| damping097_fold2_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | all | total | 0.600580 | [0.354442, 0.891439] |
| damping097_fold2_seed29_all | all | ranking_at_control_count | 0.346391 | [0.202645, 0.510804] |
| damping097_fold2_seed29_all | all | ranking_at_treatment_count | 0.364441 | [0.163723, 0.626446] |
| damping097_fold2_seed29_all | all | coverage_with_treatment_ranking | 0.254189 | [0.146821, 0.384577] |
| damping097_fold2_seed29_all | all | coverage_with_control_ranking | 0.236140 | [0.157295, 0.319134] |
| damping097_fold2_seed29_all | easy | full_total | -0.033500 | [-0.076645, -0.004066] |
| damping097_fold2_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | easy | total | -0.033500 | [-0.076645, -0.004066] |
| damping097_fold2_seed29_all | easy | ranking_at_control_count | 0.037274 | [-0.006478, 0.088164] |
| damping097_fold2_seed29_all | easy | ranking_at_treatment_count | 0.054623 | [-0.000548, 0.127509] |
| damping097_fold2_seed29_all | easy | coverage_with_treatment_ranking | -0.070774 | [-0.147031, -0.001773] |
| damping097_fold2_seed29_all | easy | coverage_with_control_ranking | -0.088124 | [-0.198291, -0.006298] |
| damping097_fold2_seed29_all | hard | full_total | 0.778066 | [0.439756, 1.224924] |
| damping097_fold2_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | hard | total | 0.778066 | [0.439756, 1.224924] |
| damping097_fold2_seed29_all | hard | ranking_at_control_count | 0.479591 | [0.268229, 0.742196] |
| damping097_fold2_seed29_all | hard | ranking_at_treatment_count | 0.495490 | [0.210603, 0.895476] |
| damping097_fold2_seed29_all | hard | coverage_with_treatment_ranking | 0.298475 | [0.162487, 0.487627] |
| damping097_fold2_seed29_all | hard | coverage_with_control_ranking | 0.282576 | [0.187495, 0.376066] |
| damping097_fold2_seed29_easy | all | full_total | -0.080810 | [-0.142121, -0.011235] |
| damping097_fold2_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | all | total | -0.080810 | [-0.142121, -0.011235] |
| damping097_fold2_seed29_easy | all | ranking_at_control_count | 0.019805 | [-0.030359, 0.077550] |
| damping097_fold2_seed29_easy | all | ranking_at_treatment_count | 0.011302 | [-0.018865, 0.044759] |
| damping097_fold2_seed29_easy | all | coverage_with_treatment_ranking | -0.100615 | [-0.136323, -0.057266] |
| damping097_fold2_seed29_easy | all | coverage_with_control_ranking | -0.092111 | [-0.133357, -0.047288] |
| damping097_fold2_seed29_easy | easy | full_total | -0.239857 | [-0.328903, -0.157574] |
| damping097_fold2_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | easy | total | -0.239857 | [-0.328903, -0.157574] |
| damping097_fold2_seed29_easy | easy | ranking_at_control_count | -0.105209 | [-0.143361, -0.065480] |
| damping097_fold2_seed29_easy | easy | ranking_at_treatment_count | -0.134794 | [-0.183238, -0.088943] |
| damping097_fold2_seed29_easy | easy | coverage_with_treatment_ranking | -0.134647 | [-0.196802, -0.075166] |
| damping097_fold2_seed29_easy | easy | coverage_with_control_ranking | -0.105063 | [-0.160435, -0.052224] |
| damping097_fold2_seed29_easy | hard | full_total | -0.007509 | [-0.069403, 0.066635] |
| damping097_fold2_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | hard | total | -0.007509 | [-0.069403, 0.066635] |
| damping097_fold2_seed29_easy | hard | ranking_at_control_count | 0.019696 | [-0.043712, 0.093489] |
| damping097_fold2_seed29_easy | hard | ranking_at_treatment_count | 0.019589 | [-0.026807, 0.080004] |
| damping097_fold2_seed29_easy | hard | coverage_with_treatment_ranking | -0.027204 | [-0.051749, -0.006515] |
| damping097_fold2_seed29_easy | hard | coverage_with_control_ranking | -0.027097 | [-0.051486, -0.005845] |
| neural_fold2_seed43_all | all | full_total | -0.165584 | [-0.307608, -0.027039] |
| neural_fold2_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | all | total | -0.165584 | [-0.307608, -0.027039] |
| neural_fold2_seed43_all | all | ranking_at_control_count | -0.209735 | [-0.340472, -0.082708] |
| neural_fold2_seed43_all | all | ranking_at_treatment_count | -0.213307 | [-0.367427, -0.049130] |
| neural_fold2_seed43_all | all | coverage_with_treatment_ranking | 0.044151 | [0.001679, 0.084010] |
| neural_fold2_seed43_all | all | coverage_with_control_ranking | 0.047723 | [-0.010261, 0.102863] |
| neural_fold2_seed43_all | easy | full_total | 0.558823 | [0.265207, 1.008989] |
| neural_fold2_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | easy | total | 0.558823 | [0.265207, 1.008989] |
| neural_fold2_seed43_all | easy | ranking_at_control_count | 0.274463 | [-0.002423, 0.567148] |
| neural_fold2_seed43_all | easy | ranking_at_treatment_count | 0.306861 | [0.026822, 0.607692] |
| neural_fold2_seed43_all | easy | coverage_with_treatment_ranking | 0.284360 | [0.024780, 0.574158] |
| neural_fold2_seed43_all | easy | coverage_with_control_ranking | 0.251962 | [-0.048835, 0.573503] |
| neural_fold2_seed43_all | hard | full_total | -0.246568 | [-0.448011, -0.055601] |
| neural_fold2_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | hard | total | -0.246568 | [-0.448011, -0.055601] |
| neural_fold2_seed43_all | hard | ranking_at_control_count | -0.291990 | [-0.487977, -0.112037] |
| neural_fold2_seed43_all | hard | ranking_at_treatment_count | -0.282172 | [-0.493090, -0.071086] |
| neural_fold2_seed43_all | hard | coverage_with_treatment_ranking | 0.045422 | [0.011228, 0.084258] |
| neural_fold2_seed43_all | hard | coverage_with_control_ranking | 0.035604 | [-0.024953, 0.098326] |
| neural_fold2_seed43_easy | all | full_total | 0.042066 | [-0.015992, 0.137507] |
| neural_fold2_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | all | total | 0.042066 | [-0.015992, 0.137507] |
| neural_fold2_seed43_easy | all | ranking_at_control_count | 0.043569 | [-0.014169, 0.137974] |
| neural_fold2_seed43_easy | all | ranking_at_treatment_count | 0.046679 | [-0.013684, 0.160305] |
| neural_fold2_seed43_easy | all | coverage_with_treatment_ranking | -0.001503 | [-0.004487, 0.000160] |
| neural_fold2_seed43_easy | all | coverage_with_control_ranking | -0.004613 | [-0.029644, 0.016185] |
| neural_fold2_seed43_easy | easy | full_total | 0.021912 | [-0.013529, 0.068597] |
| neural_fold2_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | easy | total | 0.021912 | [-0.013529, 0.068597] |
| neural_fold2_seed43_easy | easy | ranking_at_control_count | 0.027944 | [-0.002305, 0.071228] |
| neural_fold2_seed43_easy | easy | ranking_at_treatment_count | 0.029864 | [0.001036, 0.074243] |
| neural_fold2_seed43_easy | easy | coverage_with_treatment_ranking | -0.006032 | [-0.017591, 0.004048] |
| neural_fold2_seed43_easy | easy | coverage_with_control_ranking | -0.007952 | [-0.017958, 0.000408] |
| neural_fold2_seed43_easy | hard | full_total | 0.039350 | [-0.019140, 0.124960] |
| neural_fold2_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | hard | total | 0.039350 | [-0.019140, 0.124960] |
| neural_fold2_seed43_easy | hard | ranking_at_control_count | 0.041251 | [-0.017612, 0.126372] |
| neural_fold2_seed43_easy | hard | ranking_at_treatment_count | 0.030356 | [-0.015603, 0.115093] |
| neural_fold2_seed43_easy | hard | coverage_with_treatment_ranking | -0.001901 | [-0.005882, 0.000178] |
| neural_fold2_seed43_easy | hard | coverage_with_control_ranking | 0.008993 | [-0.007610, 0.034635] |
| damping097_fold2_seed43_all | all | full_total | 0.606598 | [0.428867, 0.792906] |
| damping097_fold2_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | all | total | 0.606598 | [0.428867, 0.792906] |
| damping097_fold2_seed43_all | all | ranking_at_control_count | 0.446048 | [0.297782, 0.623329] |
| damping097_fold2_seed43_all | all | ranking_at_treatment_count | 0.394533 | [0.231120, 0.590877] |
| damping097_fold2_seed43_all | all | coverage_with_treatment_ranking | 0.160550 | [0.111156, 0.207277] |
| damping097_fold2_seed43_all | all | coverage_with_control_ranking | 0.212065 | [0.116960, 0.320128] |
| damping097_fold2_seed43_all | easy | full_total | -0.041755 | [-0.095212, -0.000507] |
| damping097_fold2_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | easy | total | -0.041755 | [-0.095212, -0.000507] |
| damping097_fold2_seed43_all | easy | ranking_at_control_count | -0.001341 | [-0.032404, 0.025203] |
| damping097_fold2_seed43_all | easy | ranking_at_treatment_count | 0.029732 | [-0.001508, 0.069991] |
| damping097_fold2_seed43_all | easy | coverage_with_treatment_ranking | -0.040414 | [-0.106845, 0.002258] |
| damping097_fold2_seed43_all | easy | coverage_with_control_ranking | -0.071486 | [-0.158628, -0.003244] |
| damping097_fold2_seed43_all | hard | full_total | 0.780889 | [0.527675, 1.115648] |
| damping097_fold2_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | hard | total | 0.780889 | [0.527675, 1.115648] |
| damping097_fold2_seed43_all | hard | ranking_at_control_count | 0.592153 | [0.359945, 0.926235] |
| damping097_fold2_seed43_all | hard | ranking_at_treatment_count | 0.538432 | [0.288134, 0.873761] |
| damping097_fold2_seed43_all | hard | coverage_with_treatment_ranking | 0.188736 | [0.129406, 0.244134] |
| damping097_fold2_seed43_all | hard | coverage_with_control_ranking | 0.242457 | [0.139626, 0.358714] |
| damping097_fold2_seed43_easy | all | full_total | -0.052977 | [-0.090585, -0.015094] |
| damping097_fold2_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | all | total | -0.052977 | [-0.090585, -0.015094] |
| damping097_fold2_seed43_easy | all | ranking_at_control_count | 0.006983 | [-0.001863, 0.016117] |
| damping097_fold2_seed43_easy | all | ranking_at_treatment_count | 0.000952 | [-0.007012, 0.007736] |
| damping097_fold2_seed43_easy | all | coverage_with_treatment_ranking | -0.059960 | [-0.095775, -0.021093] |
| damping097_fold2_seed43_easy | all | coverage_with_control_ranking | -0.053929 | [-0.085192, -0.021189] |
| damping097_fold2_seed43_easy | easy | full_total | -0.264330 | [-0.377726, -0.163033] |
| damping097_fold2_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | easy | total | -0.264330 | [-0.377726, -0.163033] |
| damping097_fold2_seed43_easy | easy | ranking_at_control_count | -0.023681 | [-0.052761, 0.005518] |
| damping097_fold2_seed43_easy | easy | ranking_at_treatment_count | -0.042960 | [-0.074718, -0.014753] |
| damping097_fold2_seed43_easy | easy | coverage_with_treatment_ranking | -0.240650 | [-0.363479, -0.127109] |
| damping097_fold2_seed43_easy | easy | coverage_with_control_ranking | -0.221371 | [-0.333310, -0.122324] |
| damping097_fold2_seed43_easy | hard | full_total | -0.015129 | [-0.036349, 0.006362] |
| damping097_fold2_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | hard | total | -0.015129 | [-0.036349, 0.006362] |
| damping097_fold2_seed43_easy | hard | ranking_at_control_count | 0.000971 | [-0.013553, 0.018458] |
| damping097_fold2_seed43_easy | hard | ranking_at_treatment_count | -0.004980 | [-0.021294, 0.011242] |
| damping097_fold2_seed43_easy | hard | coverage_with_treatment_ranking | -0.016100 | [-0.034556, 0.005859] |
| damping097_fold2_seed43_easy | hard | coverage_with_control_ranking | -0.010149 | [-0.019476, 0.000076] |

Three seeds, 3,000 paired locality resamples, dependent unadjusted development views.
Detector-track image pixels, obs8/pred12 rawstride12. Not independent confirmation, seconds, metric,
human gold, physical safety, true 3D or foundation. No deployment, Stage5C or SMC.
