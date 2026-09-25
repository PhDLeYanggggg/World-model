# Frozen Risk Ranking at Matched Coverage

All 216 views after the pre-readout support amendment. No new training or deployable rule.
Four fitting/eight producer-excluded localities per group, shared twelve opened development localities.
Original decisions retain all support. Common-pool counterfactuals can violate the risk rule.

| View | ADE gain vs CV (%) | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Predicted risk violations |
|---|---:|---:|---:|---:|---|---:|---:|
| neural_fold0_seed17_all_product_original | 0.527849 | 0.788176 | 0.321778 | -1.686686 | 2/4 | 15.479901 | 0 |
| neural_fold0_seed17_all_hurdle_original | 1.513103 | 2.260063 | 1.095738 | -2.628291 | 2/4 | 24.389797 | 0 |
| neural_fold0_seed17_all_product_common | 0.527849 | 0.788176 | 0.321778 | -1.686686 | 2/4 | 15.479901 | 0 |
| neural_fold0_seed17_all_hurdle_common | 1.513103 | 2.260063 | 1.095738 | -2.628291 | 2/4 | 24.389797 | 0 |
| neural_fold0_seed17_all_hurdle_at_product | 0.570366 | 0.881145 | 0.339674 | -1.522682 | 2/4 | 15.479901 | 0 |
| neural_fold0_seed17_all_product_at_hurdle | 1.422783 | 2.153585 | 1.298125 | -1.789091 | 3/4 | 24.389797 | 18011 |
| neural_fold0_seed17_easy_product_original | 3.143858 | 4.985740 | 3.888187 | 2.316191 | 3/4 | 31.649402 | 0 |
| neural_fold0_seed17_easy_hurdle_original | 0.620696 | 0.911525 | 0.426415 | -1.518904 | 2/4 | 12.427651 | 0 |
| neural_fold0_seed17_easy_product_common | 3.143858 | 4.985740 | 3.888187 | 2.316191 | 3/4 | 31.649402 | 0 |
| neural_fold0_seed17_easy_hurdle_common | 0.620696 | 0.911525 | 0.426415 | -1.518904 | 2/4 | 12.427651 | 0 |
| neural_fold0_seed17_easy_hurdle_at_product | 2.240017 | 3.223571 | 2.236894 | 4.030252 | 3/4 | 31.649402 | 38856 |
| neural_fold0_seed17_easy_product_at_hurdle | 1.261951 | 2.035649 | 1.301712 | -0.993378 | 1/4 | 12.427651 | 0 |
| damping097_fold0_seed17_all_product_original | 2.275619 | 3.643927 | 2.234579 | -0.070270 | 0/4 | 47.957417 | 0 |
| damping097_fold0_seed17_all_hurdle_original | 3.247391 | 5.176999 | 3.119777 | -0.114654 | 0/4 | 59.761756 | 0 |
| damping097_fold0_seed17_all_product_common | 2.275619 | 3.643927 | 2.234579 | -0.070270 | 0/4 | 47.957417 | 0 |
| damping097_fold0_seed17_all_hurdle_common | 3.247391 | 5.176999 | 3.119777 | -0.114654 | 0/4 | 59.761756 | 0 |
| damping097_fold0_seed17_all_hurdle_at_product | 1.834693 | 2.988994 | 1.588797 | -0.151559 | 0/4 | 47.957417 | 0 |
| damping097_fold0_seed17_all_product_at_hurdle | 3.521332 | 5.686218 | 3.659737 | 0.083042 | 0/4 | 59.761756 | 23862 |
| damping097_fold0_seed17_easy_product_original | 1.920939 | 3.113760 | 1.675970 | 1.061809 | 0/4 | 52.171698 | 0 |
| damping097_fold0_seed17_easy_hurdle_original | 1.187116 | 1.917417 | 0.843077 | -0.791021 | 0/4 | 45.361768 | 0 |
| damping097_fold0_seed17_easy_product_common | 1.920939 | 3.113760 | 1.675970 | 1.061809 | 0/4 | 52.171698 | 0 |
| damping097_fold0_seed17_easy_hurdle_common | 1.187116 | 1.917417 | 0.843077 | -0.791021 | 0/4 | 45.361768 | 0 |
| damping097_fold0_seed17_easy_hurdle_at_product | 2.023928 | 3.141548 | 1.805592 | 2.109190 | 0/4 | 52.171698 | 13766 |
| damping097_fold0_seed17_easy_product_at_hurdle | 1.272604 | 2.179293 | 1.027824 | -1.068246 | 0/4 | 45.361768 | 0 |
| neural_fold0_seed29_all_product_original | 0.377272 | 0.601702 | 0.078860 | -1.702826 | 3/4 | 18.821545 | 0 |
| neural_fold0_seed29_all_hurdle_original | 1.760958 | 2.700001 | 1.221563 | -1.505954 | 3/4 | 32.673414 | 0 |
| neural_fold0_seed29_all_product_common | 0.377272 | 0.601702 | 0.078860 | -1.702826 | 3/4 | 18.821545 | 0 |
| neural_fold0_seed29_all_hurdle_common | 1.760958 | 2.700001 | 1.221563 | -1.505954 | 3/4 | 32.673414 | 0 |
| neural_fold0_seed29_all_hurdle_at_product | 0.599351 | 0.916518 | 0.346648 | -1.321145 | 3/4 | 18.821545 | 0 |
| neural_fold0_seed29_all_product_at_hurdle | 1.356291 | 2.108207 | 0.845638 | -1.781723 | 4/4 | 32.673414 | 28001 |
| neural_fold0_seed29_easy_product_original | 2.474484 | 3.753749 | 2.069729 | -3.312780 | 4/4 | 36.394487 | 0 |
| neural_fold0_seed29_easy_hurdle_original | 0.485973 | 0.747177 | 0.223089 | -2.546461 | 2/4 | 19.130233 | 0 |
| neural_fold0_seed29_easy_product_common | 2.474484 | 3.753749 | 2.069729 | -3.312780 | 4/4 | 36.394487 | 0 |
| neural_fold0_seed29_easy_hurdle_common | 0.485973 | 0.747177 | 0.223089 | -2.546461 | 2/4 | 19.130233 | 0 |
| neural_fold0_seed29_easy_hurdle_at_product | 1.833512 | 2.728531 | 1.051416 | -2.197293 | 3/4 | 36.394487 | 34899 |
| neural_fold0_seed29_easy_product_at_hurdle | 0.885407 | 1.382897 | 0.658608 | -2.292654 | 2/4 | 19.130233 | 0 |
| damping097_fold0_seed29_all_product_original | 1.623270 | 2.622914 | 1.435081 | -0.434938 | 0/4 | 43.679321 | 0 |
| damping097_fold0_seed29_all_hurdle_original | 3.091950 | 4.921277 | 2.948174 | 0.148147 | 0/4 | 56.259337 | 0 |
| damping097_fold0_seed29_all_product_common | 1.623270 | 2.622914 | 1.435081 | -0.434938 | 0/4 | 43.679321 | 0 |
| damping097_fold0_seed29_all_hurdle_common | 3.091950 | 4.921277 | 2.948174 | 0.148147 | 0/4 | 56.259337 | 0 |
| damping097_fold0_seed29_all_hurdle_at_product | 1.582580 | 2.548782 | 1.337284 | 0.059887 | 0/4 | 43.679321 | 0 |
| damping097_fold0_seed29_all_product_at_hurdle | 3.251391 | 5.230090 | 3.293871 | -0.001209 | 0/4 | 56.259337 | 25430 |
| damping097_fold0_seed29_easy_product_original | 1.546559 | 2.615305 | 1.318572 | -0.869818 | 0/4 | 50.952777 | 0 |
| damping097_fold0_seed29_easy_hurdle_original | 1.062633 | 1.707792 | 0.822121 | -0.855256 | 0/4 | 43.146538 | 0 |
| damping097_fold0_seed29_easy_product_common | 1.546559 | 2.615305 | 1.318572 | -0.869818 | 0/4 | 50.952777 | 0 |
| damping097_fold0_seed29_easy_hurdle_common | 1.062633 | 1.707792 | 0.822121 | -0.855256 | 0/4 | 43.146538 | 0 |
| damping097_fold0_seed29_easy_hurdle_at_product | 1.814200 | 2.810289 | 1.674519 | 1.030081 | 0/4 | 50.952777 | 15780 |
| damping097_fold0_seed29_easy_product_at_hurdle | 1.013498 | 1.737363 | 0.779256 | -1.305967 | 0/4 | 43.146538 | 0 |
| neural_fold0_seed43_all_product_original | 0.836001 | 1.288925 | 0.753556 | -1.069928 | 0/4 | 17.076272 | 0 |
| neural_fold0_seed43_all_hurdle_original | 2.422412 | 3.695595 | 2.154486 | -2.937191 | 3/4 | 35.118182 | 0 |
| neural_fold0_seed43_all_product_common | 0.836001 | 1.288925 | 0.753556 | -1.069928 | 0/4 | 17.076272 | 0 |
| neural_fold0_seed43_all_hurdle_common | 2.422412 | 3.695595 | 2.154486 | -2.937191 | 3/4 | 35.118182 | 0 |
| neural_fold0_seed43_all_hurdle_at_product | 0.587013 | 0.901268 | 0.410658 | -1.514217 | 2/4 | 17.076272 | 0 |
| neural_fold0_seed43_all_product_at_hurdle | 3.040145 | 4.508100 | 3.471908 | -1.749272 | 3/4 | 35.118182 | 36471 |
| neural_fold0_seed43_easy_product_original | 1.957573 | 3.083658 | 1.455764 | -2.031412 | 4/4 | 36.346007 | 0 |
| neural_fold0_seed43_easy_hurdle_original | 0.425819 | 0.669979 | 0.139920 | -2.850072 | 2/4 | 18.138870 | 0 |
| neural_fold0_seed43_easy_product_common | 1.957573 | 3.083658 | 1.455764 | -2.031412 | 4/4 | 36.346007 | 0 |
| neural_fold0_seed43_easy_hurdle_common | 0.425819 | 0.669979 | 0.139920 | -2.850072 | 2/4 | 18.138870 | 0 |
| neural_fold0_seed43_easy_hurdle_at_product | 1.631958 | 2.443076 | 0.859332 | -4.372501 | 3/4 | 36.346007 | 36805 |
| neural_fold0_seed43_easy_product_at_hurdle | 0.618236 | 1.006880 | 0.310339 | -1.906859 | 2/4 | 18.138870 | 0 |
| damping097_fold0_seed43_all_product_original | 2.412546 | 3.869791 | 2.486999 | -0.749784 | 0/4 | 48.152820 | 0 |
| damping097_fold0_seed43_all_hurdle_original | 3.442405 | 5.460787 | 3.344938 | 0.490194 | 0/4 | 61.733598 | 0 |
| damping097_fold0_seed43_all_product_common | 2.412546 | 3.869791 | 2.486999 | -0.749784 | 0/4 | 48.152820 | 0 |
| damping097_fold0_seed43_all_hurdle_common | 3.442405 | 5.460787 | 3.344938 | 0.490194 | 0/4 | 61.733598 | 0 |
| damping097_fold0_seed43_all_hurdle_at_product | 1.869737 | 3.046829 | 1.597112 | -0.605861 | 0/4 | 48.152820 | 0 |
| damping097_fold0_seed43_all_product_at_hurdle | 3.558213 | 5.711954 | 3.833338 | 1.695568 | 0/4 | 61.733598 | 27453 |
| damping097_fold0_seed43_easy_product_original | 1.008549 | 1.679505 | 0.611871 | -0.457379 | 0/4 | 48.542143 | 0 |
| damping097_fold0_seed43_easy_hurdle_original | 0.903708 | 1.437635 | 0.625197 | -0.738833 | 0/4 | 44.912093 | 0 |
| damping097_fold0_seed43_easy_product_common | 1.008549 | 1.679505 | 0.611871 | -0.457379 | 0/4 | 48.542143 | 0 |
| damping097_fold0_seed43_easy_hurdle_common | 0.903708 | 1.437635 | 0.625197 | -0.738833 | 0/4 | 44.912093 | 0 |
| damping097_fold0_seed43_easy_hurdle_at_product | 1.306823 | 2.054139 | 1.054554 | 0.117088 | 0/4 | 48.542143 | 7338 |
| damping097_fold0_seed43_easy_product_at_hurdle | 0.773797 | 1.331558 | 0.418667 | -0.814658 | 0/4 | 44.912093 | 0 |
| neural_fold1_seed17_all_product_original | 1.476063 | 2.104070 | 1.322426 | -2.691632 | 0/4 | 21.428015 | 0 |
| neural_fold1_seed17_all_hurdle_original | 1.940981 | 2.922913 | 1.251528 | -3.984851 | 2/4 | 35.059128 | 0 |
| neural_fold1_seed17_all_product_common | 1.476063 | 2.104070 | 1.322426 | -2.691632 | 0/4 | 21.428015 | 0 |
| neural_fold1_seed17_all_hurdle_common | 1.940981 | 2.922913 | 1.251528 | -3.984851 | 2/4 | 35.059128 | 0 |
| neural_fold1_seed17_all_hurdle_at_product | 1.120362 | 1.696921 | 0.761789 | -3.034713 | 0/4 | 21.428015 | 0 |
| neural_fold1_seed17_all_product_at_hurdle | 2.260837 | 3.210121 | 1.998136 | -3.344361 | 1/4 | 35.059128 | 40240 |
| neural_fold1_seed17_easy_product_original | 2.526328 | 4.384185 | 1.724122 | -3.702860 | 3/4 | 53.678605 | 0 |
| neural_fold1_seed17_easy_hurdle_original | 0.269287 | 0.459681 | 0.001728 | -0.927449 | 1/4 | 18.180124 | 0 |
| neural_fold1_seed17_easy_product_common | 2.526328 | 4.384185 | 1.724122 | -3.702860 | 3/4 | 53.678605 | 0 |
| neural_fold1_seed17_easy_hurdle_common | 0.269287 | 0.459681 | 0.001728 | -0.927449 | 1/4 | 18.180124 | 0 |
| neural_fold1_seed17_easy_hurdle_at_product | 1.202754 | 1.688280 | -0.622726 | 5.446634 | 3/4 | 53.678605 | 104794 |
| neural_fold1_seed17_easy_product_at_hurdle | 0.620689 | 0.948005 | 0.185382 | -1.624649 | 0/4 | 18.180124 | 0 |
| damping097_fold1_seed17_all_product_original | 2.652631 | 4.373271 | 2.632615 | 0.432737 | 0/4 | 43.741849 | 0 |
| damping097_fold1_seed17_all_hurdle_original | 3.056932 | 4.955346 | 2.787402 | -0.670979 | 0/4 | 55.578289 | 0 |
| damping097_fold1_seed17_all_product_common | 2.652631 | 4.373271 | 2.632615 | 0.432737 | 0/4 | 43.741849 | 0 |
| damping097_fold1_seed17_all_hurdle_common | 3.056932 | 4.955346 | 2.787402 | -0.670979 | 0/4 | 55.578289 | 0 |
| damping097_fold1_seed17_all_hurdle_at_product | 2.304211 | 3.776460 | 1.909752 | -0.670979 | 0/4 | 43.741849 | 6 |
| damping097_fold1_seed17_all_product_at_hurdle | 3.427955 | 5.668569 | 3.590769 | 0.426824 | 0/4 | 55.578289 | 34948 |
| damping097_fold1_seed17_easy_product_original | 1.097062 | 1.791481 | 0.356701 | -1.058738 | 0/4 | 45.640855 | 0 |
| damping097_fold1_seed17_easy_hurdle_original | 0.680209 | 1.146114 | 0.148872 | -1.671822 | 0/4 | 36.685106 | 0 |
| damping097_fold1_seed17_easy_product_common | 1.097062 | 1.791481 | 0.356701 | -1.058738 | 0/4 | 45.640855 | 0 |
| damping097_fold1_seed17_easy_hurdle_common | 0.675808 | 1.138172 | 0.148240 | -1.671822 | 0/4 | 36.675960 | 0 |
| damping097_fold1_seed17_easy_hurdle_at_product | 1.236350 | 2.053230 | 0.563107 | -1.493142 | 0/4 | 45.640855 | 26511 |
| damping097_fold1_seed17_easy_product_at_hurdle | 0.639222 | 1.138786 | 0.040366 | -0.564467 | 0/4 | 36.675960 | 46 |
| neural_fold1_seed29_all_product_original | 1.388330 | 2.072887 | 1.245516 | -3.559810 | 0/4 | 19.064927 | 0 |
| neural_fold1_seed29_all_hurdle_original | 1.690189 | 2.635867 | 0.998836 | -4.499661 | 2/4 | 29.314007 | 0 |
| neural_fold1_seed29_all_product_common | 1.388330 | 2.072887 | 1.245516 | -3.559810 | 0/4 | 19.064927 | 0 |
| neural_fold1_seed29_all_hurdle_common | 1.690189 | 2.635867 | 0.998836 | -4.499661 | 2/4 | 29.314007 | 0 |
| neural_fold1_seed29_all_hurdle_at_product | 1.044815 | 1.638408 | 0.644260 | -3.989888 | 0/4 | 19.064927 | 0 |
| neural_fold1_seed29_all_product_at_hurdle | 2.000530 | 3.095510 | 1.887715 | -3.694400 | 1/4 | 29.314007 | 30256 |
| neural_fold1_seed29_easy_product_original | 1.051769 | 1.719927 | 0.088573 | -2.989496 | 3/4 | 32.756676 | 0 |
| neural_fold1_seed29_easy_hurdle_original | -0.211893 | -0.516492 | -1.054115 | 0.674637 | 1/4 | 15.312984 | 0 |
| neural_fold1_seed29_easy_product_common | 1.051769 | 1.719927 | 0.088573 | -2.989496 | 3/4 | 32.756676 | 0 |
| neural_fold1_seed29_easy_hurdle_common | -0.211893 | -0.516492 | -1.054115 | 0.674637 | 1/4 | 15.312984 | 0 |
| neural_fold1_seed29_easy_hurdle_at_product | 0.091657 | 0.273814 | -1.179366 | 5.089066 | 2/4 | 32.756676 | 51495 |
| neural_fold1_seed29_easy_product_at_hurdle | 0.218253 | 0.368933 | 0.002992 | -1.242445 | 1/4 | 15.312984 | 0 |
| damping097_fold1_seed29_all_product_original | 2.402716 | 3.979663 | 2.593467 | -0.525681 | 0/4 | 39.683680 | 0 |
| damping097_fold1_seed29_all_hurdle_original | 3.247313 | 5.401469 | 3.334235 | 1.173901 | 0/4 | 55.078978 | 0 |
| damping097_fold1_seed29_all_product_common | 2.402716 | 3.979663 | 2.593467 | -0.525681 | 0/4 | 39.683680 | 0 |
| damping097_fold1_seed29_all_hurdle_common | 3.247313 | 5.401469 | 3.334235 | 1.173901 | 0/4 | 55.078978 | 0 |
| damping097_fold1_seed29_all_hurdle_at_product | 2.168333 | 3.558242 | 1.758861 | -0.983485 | 0/4 | 39.683680 | 0 |
| damping097_fold1_seed29_all_product_at_hurdle | 3.494619 | 5.728755 | 4.080318 | 0.502813 | 0/4 | 55.078978 | 45448 |
| damping097_fold1_seed29_easy_product_original | 1.050601 | 1.744625 | 0.315404 | -1.224087 | 0/4 | 43.255749 | 0 |
| damping097_fold1_seed29_easy_hurdle_original | 0.751618 | 1.248724 | 0.224115 | -1.283604 | 0/4 | 37.050951 | 0 |
| damping097_fold1_seed29_easy_product_common | 1.050601 | 1.744625 | 0.315404 | -1.224087 | 0/4 | 43.255749 | 0 |
| damping097_fold1_seed29_easy_hurdle_common | 0.750822 | 1.248725 | 0.224124 | -1.283604 | 0/4 | 37.049934 | 0 |
| damping097_fold1_seed29_easy_hurdle_at_product | 1.367037 | 2.246959 | 0.711962 | -1.025310 | 0/4 | 43.255749 | 18320 |
| damping097_fold1_seed29_easy_product_at_hurdle | 0.622284 | 1.047356 | 0.067273 | -1.072474 | 0/4 | 37.049934 | 0 |
| neural_fold1_seed43_all_product_original | 0.904338 | 1.590134 | 1.240587 | 3.422871 | 1/4 | 20.747476 | 0 |
| neural_fold1_seed43_all_hurdle_original | 2.010392 | 3.141469 | 1.322067 | -2.243388 | 3/4 | 36.633955 | 0 |
| neural_fold1_seed43_all_product_common | 0.904338 | 1.590134 | 1.240587 | 3.422871 | 1/4 | 20.747476 | 0 |
| neural_fold1_seed43_all_hurdle_common | 2.010392 | 3.141469 | 1.322067 | -2.243388 | 3/4 | 36.633955 | 0 |
| neural_fold1_seed43_all_hurdle_at_product | 1.075558 | 1.788501 | 0.676857 | -2.186799 | 1/4 | 20.747476 | 34 |
| neural_fold1_seed43_all_product_at_hurdle | 1.985370 | 3.101012 | 2.229174 | -2.325950 | 2/4 | 36.633955 | 46932 |
| neural_fold1_seed43_easy_product_original | 1.285332 | 2.083556 | 0.131553 | -2.396419 | 3/4 | 37.088890 | 0 |
| neural_fold1_seed43_easy_hurdle_original | 0.235886 | 0.419044 | -0.000060 | -0.735934 | 1/4 | 16.353948 | 0 |
| neural_fold1_seed43_easy_product_common | 1.285332 | 2.083556 | 0.131553 | -2.396419 | 3/4 | 37.088890 | 0 |
| neural_fold1_seed43_easy_hurdle_common | 0.235886 | 0.419044 | -0.000060 | -0.735934 | 1/4 | 16.353948 | 0 |
| neural_fold1_seed43_easy_hurdle_at_product | 0.810687 | 1.445624 | 0.082462 | -0.719379 | 2/4 | 37.088890 | 61211 |
| neural_fold1_seed43_easy_product_at_hurdle | 0.252437 | 0.403700 | 0.013146 | -0.602124 | 1/4 | 16.353948 | 0 |
| damping097_fold1_seed43_all_product_original | 2.690452 | 4.441297 | 2.860899 | 0.242864 | 0/4 | 41.980034 | 0 |
| damping097_fold1_seed43_all_hurdle_original | 3.123022 | 5.164768 | 2.988168 | 1.037620 | 0/4 | 55.111159 | 0 |
| damping097_fold1_seed43_all_product_common | 2.690452 | 4.441297 | 2.860899 | 0.242864 | 0/4 | 41.980034 | 0 |
| damping097_fold1_seed43_all_hurdle_common | 3.123022 | 5.164768 | 2.988168 | 1.037620 | 0/4 | 55.111159 | 0 |
| damping097_fold1_seed43_all_hurdle_at_product | 2.336985 | 3.813572 | 2.014018 | -0.893133 | 0/4 | 41.980034 | 0 |
| damping097_fold1_seed43_all_product_at_hurdle | 3.568836 | 5.993709 | 4.170073 | 0.575327 | 0/4 | 55.111159 | 38764 |
| damping097_fold1_seed43_easy_product_original | 1.473118 | 2.250746 | 0.755746 | 0.333971 | 0/4 | 49.030680 | 0 |
| damping097_fold1_seed43_easy_hurdle_original | 0.592051 | 0.997293 | 0.132836 | -0.733756 | 0/4 | 35.305735 | 0 |
| damping097_fold1_seed43_easy_product_common | 1.473118 | 2.250746 | 0.755746 | 0.333971 | 0/4 | 49.030680 | 0 |
| damping097_fold1_seed43_easy_hurdle_common | 0.592051 | 0.997293 | 0.132836 | -0.733756 | 0/4 | 35.305735 | 0 |
| damping097_fold1_seed43_easy_hurdle_at_product | 1.534169 | 2.464476 | 1.010832 | 1.045344 | 0/4 | 49.030680 | 40517 |
| damping097_fold1_seed43_easy_product_at_hurdle | 0.599296 | 0.993396 | 0.102619 | -0.936447 | 0/4 | 35.305735 | 0 |
| neural_fold2_seed17_all_product_original | 0.358013 | 0.475953 | 0.317783 | 0.074762 | 0/0 | 7.168617 | 0 |
| neural_fold2_seed17_all_hurdle_original | 0.662188 | 0.947734 | 0.585514 | 0.348621 | 0/0 | 11.425828 | 0 |
| neural_fold2_seed17_all_product_common | 0.358013 | 0.475953 | 0.317783 | 0.074762 | 0/0 | 7.168617 | 0 |
| neural_fold2_seed17_all_hurdle_common | 0.662188 | 0.947734 | 0.585514 | 0.348621 | 0/0 | 11.425828 | 0 |
| neural_fold2_seed17_all_hurdle_at_product | 0.484127 | 0.683496 | 0.460730 | 0.216256 | 0/0 | 7.168617 | 0 |
| neural_fold2_seed17_all_product_at_hurdle | 0.655354 | 0.898827 | 0.639964 | 0.630176 | 0/0 | 11.425828 | 5985 |
| neural_fold2_seed17_easy_product_original | 1.968109 | 2.594642 | 1.702011 | 17.254557 | 0/0 | 29.253477 | 0 |
| neural_fold2_seed17_easy_hurdle_original | 0.591345 | 0.778075 | 0.649846 | 0.047737 | 0/0 | 1.617527 | 0 |
| neural_fold2_seed17_easy_product_common | 1.968109 | 2.594642 | 1.702011 | 17.254557 | 0/0 | 29.253477 | 0 |
| neural_fold2_seed17_easy_hurdle_common | 0.591345 | 0.778075 | 0.649846 | 0.047737 | 0/0 | 1.617527 | 0 |
| neural_fold2_seed17_easy_hurdle_at_product | 2.589743 | 3.619297 | 2.678066 | 13.015050 | 0/0 | 29.253477 | 38852 |
| neural_fold2_seed17_easy_product_at_hurdle | 0.149288 | 0.186156 | 0.114912 | 0.595538 | 0/0 | 1.617527 | 0 |
| damping097_fold2_seed17_all_product_original | 3.140129 | 4.949794 | 3.108058 | -2.678459 | 0/0 | 46.588185 | 0 |
| damping097_fold2_seed17_all_hurdle_original | 2.928583 | 4.698599 | 2.446213 | -2.899898 | 0/0 | 61.506562 | 0 |
| damping097_fold2_seed17_all_product_common | 3.140129 | 4.949794 | 3.108058 | -2.678459 | 0/0 | 46.588185 | 0 |
| damping097_fold2_seed17_all_hurdle_common | 2.928583 | 4.698599 | 2.446213 | -2.899898 | 0/0 | 61.506562 | 0 |
| damping097_fold2_seed17_all_hurdle_at_product | 1.737358 | 2.789872 | 1.282563 | -3.059848 | 0/0 | 46.588185 | 0 |
| damping097_fold2_seed17_all_product_at_hurdle | 4.188553 | 6.684907 | 4.110333 | -3.066241 | 0/0 | 61.506562 | 20973 |
| damping097_fold2_seed17_easy_product_original | 4.419057 | 7.091983 | 4.349350 | -1.773201 | 0/0 | 72.893267 | 0 |
| damping097_fold2_seed17_easy_hurdle_original | 0.869042 | 1.362455 | 0.474012 | -3.492180 | 0/0 | 38.063094 | 0 |
| damping097_fold2_seed17_easy_product_common | 4.419057 | 7.091983 | 4.349350 | -1.773201 | 0/0 | 72.893267 | 0 |
| damping097_fold2_seed17_easy_hurdle_common | 0.869042 | 1.362455 | 0.474012 | -3.492180 | 0/0 | 38.063094 | 0 |
| damping097_fold2_seed17_easy_hurdle_at_product | 4.116773 | 6.540786 | 3.850663 | -1.632874 | 0/0 | 72.893267 | 48966 |
| damping097_fold2_seed17_easy_product_at_hurdle | 1.574150 | 2.613606 | 1.354676 | -2.469402 | 0/0 | 38.063094 | 0 |
| neural_fold2_seed29_all_product_original | 1.284110 | 1.753228 | 1.317918 | 1.195401 | 0/0 | 7.752605 | 0 |
| neural_fold2_seed29_all_hurdle_original | 1.179131 | 1.521074 | 1.145000 | 0.195236 | 0/0 | 9.912153 | 0 |
| neural_fold2_seed29_all_product_common | 1.284110 | 1.753228 | 1.317918 | 1.195401 | 0/0 | 7.752605 | 0 |
| neural_fold2_seed29_all_hurdle_common | 1.179131 | 1.521074 | 1.145000 | 0.195236 | 0/0 | 9.912153 | 0 |
| neural_fold2_seed29_all_hurdle_at_product | 1.077465 | 1.343626 | 1.053254 | 0.103960 | 0/0 | 7.752605 | 0 |
| neural_fold2_seed29_all_product_at_hurdle | 1.435854 | 2.045756 | 1.462888 | 1.228313 | 0/0 | 9.912153 | 3036 |
| neural_fold2_seed29_easy_product_original | 2.423747 | 3.254269 | 2.245071 | 16.526276 | 0/0 | 29.986841 | 0 |
| neural_fold2_seed29_easy_hurdle_original | 0.904723 | 1.165288 | 0.918200 | 0.085972 | 0/0 | 2.695878 | 0 |
| neural_fold2_seed29_easy_product_common | 2.423747 | 3.254269 | 2.245071 | 16.526276 | 0/0 | 29.986841 | 0 |
| neural_fold2_seed29_easy_hurdle_common | 0.904723 | 1.165288 | 0.918200 | 0.085972 | 0/0 | 2.695878 | 0 |
| neural_fold2_seed29_easy_hurdle_at_product | 3.113450 | 4.360460 | 3.112274 | 13.708315 | 0/0 | 29.986841 | 38367 |
| neural_fold2_seed29_easy_product_at_hurdle | 0.168317 | 0.208090 | 0.082908 | 1.207756 | 0/0 | 2.695878 | 0 |
| damping097_fold2_seed29_all_product_original | 3.013484 | 4.808199 | 2.982758 | -2.382294 | 0/0 | 42.521606 | 0 |
| damping097_fold2_seed29_all_hurdle_original | 2.759739 | 4.405037 | 2.278549 | -2.202410 | 0/0 | 54.805278 | 0 |
| damping097_fold2_seed29_all_product_common | 3.013484 | 4.808199 | 2.982758 | -2.382294 | 0/0 | 42.521606 | 0 |
| damping097_fold2_seed29_all_hurdle_common | 2.759739 | 4.405037 | 2.278549 | -2.202410 | 0/0 | 54.805278 | 0 |
| damping097_fold2_seed29_all_hurdle_at_product | 1.799637 | 2.888604 | 1.385871 | -2.372156 | 0/0 | 42.521606 | 0 |
| damping097_fold2_seed29_all_product_at_hurdle | 3.829686 | 6.118442 | 3.737309 | -2.472881 | 0/0 | 54.805278 | 17269 |
| damping097_fold2_seed29_easy_product_original | 4.520098 | 7.302772 | 4.553596 | -1.359587 | 0/0 | 66.182736 | 0 |
| damping097_fold2_seed29_easy_hurdle_original | 0.661714 | 1.041732 | 0.333431 | -2.802258 | 0/0 | 30.659743 | 0 |
| damping097_fold2_seed29_easy_product_common | 4.520098 | 7.302772 | 4.553596 | -1.359587 | 0/0 | 66.182736 | 0 |
| damping097_fold2_seed29_easy_hurdle_common | 0.661714 | 1.041732 | 0.333431 | -2.802258 | 0/0 | 30.659743 | 0 |
| damping097_fold2_seed29_easy_hurdle_at_product | 4.020546 | 6.420823 | 3.792270 | -1.078439 | 0/0 | 66.182736 | 49940 |
| damping097_fold2_seed29_easy_product_at_hurdle | 1.428360 | 2.349266 | 1.275569 | -2.112507 | 0/0 | 30.659743 | 0 |
| neural_fold2_seed43_all_product_original | 0.293096 | 0.399954 | 0.254047 | -0.018139 | 0/0 | 7.618167 | 0 |
| neural_fold2_seed43_all_hurdle_original | 0.616125 | 0.899569 | 0.617665 | 0.354457 | 0/0 | 10.298396 | 0 |
| neural_fold2_seed43_all_product_common | 0.293096 | 0.399954 | 0.254047 | -0.018139 | 0/0 | 7.618167 | 0 |
| neural_fold2_seed43_all_hurdle_common | 0.616125 | 0.899569 | 0.617665 | 0.354457 | 0/0 | 10.298396 | 0 |
| neural_fold2_seed43_all_hurdle_at_product | 0.469524 | 0.692062 | 0.432896 | -0.223859 | 0/0 | 7.618167 | 0 |
| neural_fold2_seed43_all_product_at_hurdle | 0.432398 | 0.598061 | 0.391833 | 0.482482 | 0/0 | 10.298396 | 3768 |
| neural_fold2_seed43_easy_product_original | 1.832685 | 2.502230 | 1.597398 | 15.053146 | 0/0 | 27.401216 | 0 |
| neural_fold2_seed43_easy_hurdle_original | 0.587082 | 0.847449 | 0.654504 | 0.076986 | 0/0 | 1.783263 | 0 |
| neural_fold2_seed43_easy_product_common | 1.832685 | 2.502230 | 1.597398 | 15.053146 | 0/0 | 27.401216 | 0 |
| neural_fold2_seed43_easy_hurdle_common | 0.587082 | 0.847449 | 0.654504 | 0.076986 | 0/0 | 1.783263 | 0 |
| neural_fold2_seed43_easy_hurdle_at_product | 2.102285 | 3.150935 | 2.121881 | 8.342517 | 0/0 | 27.401216 | 36015 |
| neural_fold2_seed43_easy_product_at_hurdle | 0.124394 | 0.177412 | 0.064094 | 0.796091 | 0/0 | 1.783263 | 0 |
| damping097_fold2_seed43_all_product_original | 2.756897 | 4.366384 | 2.689216 | -2.166192 | 0/0 | 43.339617 | 0 |
| damping097_fold2_seed43_all_hurdle_original | 2.722615 | 4.345347 | 2.221158 | -2.219850 | 0/0 | 57.659779 | 0 |
| damping097_fold2_seed43_all_product_common | 2.756897 | 4.366384 | 2.689216 | -2.166192 | 0/0 | 43.339617 | 0 |
| damping097_fold2_seed43_all_hurdle_common | 2.722615 | 4.345347 | 2.221158 | -2.219850 | 0/0 | 57.659779 | 0 |
| damping097_fold2_seed43_all_hurdle_at_product | 1.553007 | 2.500698 | 1.124334 | -2.341427 | 0/0 | 43.339617 | 0 |
| damping097_fold2_seed43_all_product_at_hurdle | 3.851332 | 6.109753 | 3.763394 | -2.075955 | 0/0 | 57.659779 | 20132 |
| damping097_fold2_seed43_easy_product_original | 3.909375 | 6.344618 | 3.891994 | -2.039558 | 0/0 | 65.024007 | 0 |
| damping097_fold2_seed43_easy_hurdle_original | 0.695502 | 1.105726 | 0.324467 | -2.853000 | 0/0 | 32.538322 | 0 |
| damping097_fold2_seed43_easy_product_common | 3.909375 | 6.344618 | 3.891994 | -2.039558 | 0/0 | 65.024007 | 0 |
| damping097_fold2_seed43_easy_hurdle_common | 0.695502 | 1.105726 | 0.324467 | -2.853000 | 0/0 | 32.538322 | 0 |
| damping097_fold2_seed43_easy_hurdle_at_product | 3.434469 | 5.411815 | 3.043608 | -1.975074 | 0/0 | 65.024007 | 45670 |
| damping097_fold2_seed43_easy_product_at_hurdle | 1.555325 | 2.568980 | 1.366441 | -1.877241 | 0/0 | 32.538322 | 0 |

## Additive Components

Percentage points of CV-normalized improvement, not percentages against changing policy denominators.
Positive ranking components favor hurdle ordering. Both paths retained; no unique causal attribution.

| Group | Subset | Component | Mean (pp) | Conditional 95% CI (pp) |
|---|---|---|---:|---|
| neural_fold0_seed17_all | all | full_total | 0.985254 | [0.738472, 1.230709] |
| neural_fold0_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | all | total | 0.985254 | [0.738472, 1.230709] |
| neural_fold0_seed17_all | all | ranking_at_product_count | 0.042518 | [-0.071711, 0.166289] |
| neural_fold0_seed17_all | all | coverage_with_hurdle_ranking | 0.942736 | [0.733957, 1.133437] |
| neural_fold0_seed17_all | all | coverage_with_product_ranking | 0.894934 | [0.519248, 1.237582] |
| neural_fold0_seed17_all | all | ranking_at_hurdle_count | 0.090319 | [-0.212506, 0.413645] |
| neural_fold0_seed17_all | easy | full_total | 2.216044 | [1.622579, 2.809510] |
| neural_fold0_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | easy | total | 2.216044 | [1.622579, 2.809510] |
| neural_fold0_seed17_all | easy | ranking_at_product_count | 0.112854 | [-0.207798, 0.400608] |
| neural_fold0_seed17_all | easy | coverage_with_hurdle_ranking | 2.103191 | [1.509112, 2.612381] |
| neural_fold0_seed17_all | easy | coverage_with_product_ranking | 1.480191 | [0.882410, 2.046817] |
| neural_fold0_seed17_all | easy | ranking_at_hurdle_count | 0.735853 | [0.328293, 1.089125] |
| neural_fold0_seed17_all | hard | full_total | 0.773960 | [0.408550, 1.133178] |
| neural_fold0_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_all | hard | total | 0.773960 | [0.408550, 1.133178] |
| neural_fold0_seed17_all | hard | ranking_at_product_count | 0.017896 | [-0.111489, 0.169404] |
| neural_fold0_seed17_all | hard | coverage_with_hurdle_ranking | 0.756064 | [0.417835, 1.105747] |
| neural_fold0_seed17_all | hard | coverage_with_product_ranking | 0.976346 | [0.483384, 1.514007] |
| neural_fold0_seed17_all | hard | ranking_at_hurdle_count | -0.202387 | [-0.508220, 0.103441] |
| neural_fold0_seed17_easy | all | full_total | -2.523162 | [-3.909573, -0.631396] |
| neural_fold0_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | all | total | -2.523162 | [-3.909573, -0.631396] |
| neural_fold0_seed17_easy | all | ranking_at_product_count | -0.903840 | [-1.803914, -0.082876] |
| neural_fold0_seed17_easy | all | coverage_with_hurdle_ranking | -1.619321 | [-2.459273, -0.368987] |
| neural_fold0_seed17_easy | all | coverage_with_product_ranking | -1.881907 | [-3.063473, -0.259482] |
| neural_fold0_seed17_easy | all | ranking_at_hurdle_count | -0.641255 | [-0.893839, -0.356546] |
| neural_fold0_seed17_easy | easy | full_total | -1.360708 | [-3.292423, 1.092693] |
| neural_fold0_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | easy | total | -1.360708 | [-3.292423, 1.092693] |
| neural_fold0_seed17_easy | easy | ranking_at_product_count | 0.836312 | [-0.033281, 1.554044] |
| neural_fold0_seed17_easy | easy | coverage_with_hurdle_ranking | -2.197020 | [-4.676470, 1.059786] |
| neural_fold0_seed17_easy | easy | coverage_with_product_ranking | -2.381656 | [-4.656163, 0.366465] |
| neural_fold0_seed17_easy | easy | ranking_at_hurdle_count | 1.020948 | [0.561181, 1.479642] |
| neural_fold0_seed17_easy | hard | full_total | -3.461772 | [-4.453944, -2.230949] |
| neural_fold0_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed17_easy | hard | total | -3.461772 | [-4.453944, -2.230949] |
| neural_fold0_seed17_easy | hard | ranking_at_product_count | -1.651292 | [-2.549494, -0.855179] |
| neural_fold0_seed17_easy | hard | coverage_with_hurdle_ranking | -1.810479 | [-2.540474, -1.131157] |
| neural_fold0_seed17_easy | hard | coverage_with_product_ranking | -2.586475 | [-3.529393, -1.446483] |
| neural_fold0_seed17_easy | hard | ranking_at_hurdle_count | -0.875297 | [-1.021056, -0.748265] |
| damping097_fold0_seed17_all | all | full_total | 0.971772 | [0.584976, 1.344234] |
| damping097_fold0_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | all | total | 0.971772 | [0.584976, 1.344234] |
| damping097_fold0_seed17_all | all | ranking_at_product_count | -0.440926 | [-0.775060, -0.095616] |
| damping097_fold0_seed17_all | all | coverage_with_hurdle_ranking | 1.412698 | [0.975655, 1.724121] |
| damping097_fold0_seed17_all | all | coverage_with_product_ranking | 1.245713 | [0.830925, 1.567345] |
| damping097_fold0_seed17_all | all | ranking_at_hurdle_count | -0.273941 | [-0.744292, 0.200845] |
| damping097_fold0_seed17_all | easy | full_total | 0.285452 | [0.036023, 0.513187] |
| damping097_fold0_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | easy | total | 0.285452 | [0.036023, 0.513187] |
| damping097_fold0_seed17_all | easy | ranking_at_product_count | 0.377305 | [0.210841, 0.587133] |
| damping097_fold0_seed17_all | easy | coverage_with_hurdle_ranking | -0.091853 | [-0.245065, 0.040041] |
| damping097_fold0_seed17_all | easy | coverage_with_product_ranking | 0.022221 | [-0.194789, 0.214313] |
| damping097_fold0_seed17_all | easy | ranking_at_hurdle_count | 0.263231 | [0.085448, 0.517067] |
| damping097_fold0_seed17_all | hard | full_total | 0.885199 | [0.465228, 1.308047] |
| damping097_fold0_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_all | hard | total | 0.885199 | [0.465228, 1.308047] |
| damping097_fold0_seed17_all | hard | ranking_at_product_count | -0.645782 | [-1.072316, -0.169825] |
| damping097_fold0_seed17_all | hard | coverage_with_hurdle_ranking | 1.530980 | [1.017152, 1.894089] |
| damping097_fold0_seed17_all | hard | coverage_with_product_ranking | 1.425159 | [0.968999, 1.734517] |
| damping097_fold0_seed17_all | hard | ranking_at_hurdle_count | -0.539960 | [-1.058241, 0.004911] |
| damping097_fold0_seed17_easy | all | full_total | -0.733824 | [-1.094571, -0.326794] |
| damping097_fold0_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | all | total | -0.733824 | [-1.094571, -0.326794] |
| damping097_fold0_seed17_easy | all | ranking_at_product_count | 0.102989 | [-0.222658, 0.454376] |
| damping097_fold0_seed17_easy | all | coverage_with_hurdle_ranking | -0.836813 | [-1.062342, -0.595463] |
| damping097_fold0_seed17_easy | all | coverage_with_product_ranking | -0.648336 | [-0.838735, -0.398748] |
| damping097_fold0_seed17_easy | all | ranking_at_hurdle_count | -0.085488 | [-0.300672, 0.142837] |
| damping097_fold0_seed17_easy | easy | full_total | 0.363031 | [0.040176, 0.839130] |
| damping097_fold0_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | easy | total | 0.363031 | [0.040176, 0.839130] |
| damping097_fold0_seed17_easy | easy | ranking_at_product_count | -0.019717 | [-0.355948, 0.228972] |
| damping097_fold0_seed17_easy | easy | coverage_with_hurdle_ranking | 0.382749 | [-0.018152, 1.123436] |
| damping097_fold0_seed17_easy | easy | coverage_with_product_ranking | 0.218541 | [-0.144484, 0.807971] |
| damping097_fold0_seed17_easy | easy | ranking_at_hurdle_count | 0.144490 | [0.005978, 0.258888] |
| damping097_fold0_seed17_easy | hard | full_total | -0.832893 | [-1.269044, -0.327650] |
| damping097_fold0_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed17_easy | hard | total | -0.832893 | [-1.269044, -0.327650] |
| damping097_fold0_seed17_easy | hard | ranking_at_product_count | 0.129622 | [-0.189920, 0.494454] |
| damping097_fold0_seed17_easy | hard | coverage_with_hurdle_ranking | -0.962516 | [-1.184020, -0.737678] |
| damping097_fold0_seed17_easy | hard | coverage_with_product_ranking | -0.648145 | [-0.872312, -0.366474] |
| damping097_fold0_seed17_easy | hard | ranking_at_hurdle_count | -0.184748 | [-0.466658, 0.117690] |
| neural_fold0_seed29_all | all | full_total | 1.383687 | [1.004367, 1.723473] |
| neural_fold0_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | all | total | 1.383687 | [1.004367, 1.723473] |
| neural_fold0_seed29_all | all | ranking_at_product_count | 0.222080 | [0.115914, 0.341141] |
| neural_fold0_seed29_all | all | coverage_with_hurdle_ranking | 1.161607 | [0.833491, 1.443753] |
| neural_fold0_seed29_all | all | coverage_with_product_ranking | 0.979020 | [0.509934, 1.483212] |
| neural_fold0_seed29_all | all | ranking_at_hurdle_count | 0.404667 | [-0.013525, 0.827678] |
| neural_fold0_seed29_all | easy | full_total | 2.862814 | [1.787385, 3.814846] |
| neural_fold0_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | easy | total | 2.862814 | [1.787385, 3.814846] |
| neural_fold0_seed29_all | easy | ranking_at_product_count | -0.470616 | [-0.643240, -0.304902] |
| neural_fold0_seed29_all | easy | coverage_with_hurdle_ranking | 3.333429 | [2.188637, 4.377577] |
| neural_fold0_seed29_all | easy | coverage_with_product_ranking | 2.739812 | [1.814394, 3.531348] |
| neural_fold0_seed29_all | easy | ranking_at_hurdle_count | 0.123001 | [-0.184101, 0.390675] |
| neural_fold0_seed29_all | hard | full_total | 1.142703 | [0.665448, 1.641759] |
| neural_fold0_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_all | hard | total | 1.142703 | [0.665448, 1.641759] |
| neural_fold0_seed29_all | hard | ranking_at_product_count | 0.267789 | [0.131103, 0.414600] |
| neural_fold0_seed29_all | hard | coverage_with_hurdle_ranking | 0.874914 | [0.477394, 1.280380] |
| neural_fold0_seed29_all | hard | coverage_with_product_ranking | 0.766779 | [0.171635, 1.523741] |
| neural_fold0_seed29_all | hard | ranking_at_hurdle_count | 0.375924 | [-0.213273, 0.965860] |
| neural_fold0_seed29_easy | all | full_total | -1.988512 | [-2.699978, -1.219527] |
| neural_fold0_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | all | total | -1.988512 | [-2.699978, -1.219527] |
| neural_fold0_seed29_easy | all | ranking_at_product_count | -0.640973 | [-1.032330, -0.345051] |
| neural_fold0_seed29_easy | all | coverage_with_hurdle_ranking | -1.347539 | [-2.005063, -0.697546] |
| neural_fold0_seed29_easy | all | coverage_with_product_ranking | -1.589077 | [-2.217887, -0.941670] |
| neural_fold0_seed29_easy | all | ranking_at_hurdle_count | -0.399434 | [-0.555327, -0.228688] |
| neural_fold0_seed29_easy | easy | full_total | -3.950426 | [-5.470267, -2.464954] |
| neural_fold0_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | easy | total | -3.950426 | [-5.470267, -2.464954] |
| neural_fold0_seed29_easy | easy | ranking_at_product_count | 0.710837 | [0.115930, 1.174659] |
| neural_fold0_seed29_easy | easy | coverage_with_hurdle_ranking | -4.661264 | [-6.491088, -2.738793] |
| neural_fold0_seed29_easy | easy | coverage_with_product_ranking | -5.017737 | [-6.892318, -3.119949] |
| neural_fold0_seed29_easy | easy | ranking_at_hurdle_count | 1.067311 | [0.465264, 1.785693] |
| neural_fold0_seed29_easy | hard | full_total | -1.846640 | [-2.678094, -1.004737] |
| neural_fold0_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed29_easy | hard | total | -1.846640 | [-2.678094, -1.004737] |
| neural_fold0_seed29_easy | hard | ranking_at_product_count | -1.018313 | [-1.452662, -0.635256] |
| neural_fold0_seed29_easy | hard | coverage_with_hurdle_ranking | -0.828327 | [-1.597239, -0.248765] |
| neural_fold0_seed29_easy | hard | coverage_with_product_ranking | -1.411121 | [-2.206457, -0.563487] |
| neural_fold0_seed29_easy | hard | ranking_at_hurdle_count | -0.435519 | [-0.584379, -0.270359] |
| damping097_fold0_seed29_all | all | full_total | 1.468679 | [0.961152, 1.943176] |
| damping097_fold0_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | all | total | 1.468679 | [0.961152, 1.943176] |
| damping097_fold0_seed29_all | all | ranking_at_product_count | -0.040690 | [-0.254207, 0.174595] |
| damping097_fold0_seed29_all | all | coverage_with_hurdle_ranking | 1.509369 | [0.978220, 1.953072] |
| damping097_fold0_seed29_all | all | coverage_with_product_ranking | 1.628121 | [1.029401, 2.131825] |
| damping097_fold0_seed29_all | all | ranking_at_hurdle_count | -0.159441 | [-0.474131, 0.132459] |
| damping097_fold0_seed29_all | easy | full_total | 0.017505 | [-0.218272, 0.212106] |
| damping097_fold0_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | easy | total | 0.017505 | [-0.218272, 0.212106] |
| damping097_fold0_seed29_all | easy | ranking_at_product_count | 0.122100 | [-0.088096, 0.274451] |
| damping097_fold0_seed29_all | easy | coverage_with_hurdle_ranking | -0.104595 | [-0.246164, 0.018199] |
| damping097_fold0_seed29_all | easy | coverage_with_product_ranking | -0.150342 | [-0.408767, 0.067153] |
| damping097_fold0_seed29_all | easy | ranking_at_hurdle_count | 0.167847 | [-0.000982, 0.346635] |
| damping097_fold0_seed29_all | hard | full_total | 1.513093 | [0.988913, 2.022595] |
| damping097_fold0_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_all | hard | total | 1.513093 | [0.988913, 2.022595] |
| damping097_fold0_seed29_all | hard | ranking_at_product_count | -0.097797 | [-0.419097, 0.263858] |
| damping097_fold0_seed29_all | hard | coverage_with_hurdle_ranking | 1.610890 | [0.946754, 2.154581] |
| damping097_fold0_seed29_all | hard | coverage_with_product_ranking | 1.858789 | [1.330532, 2.301363] |
| damping097_fold0_seed29_all | hard | ranking_at_hurdle_count | -0.345697 | [-0.642207, -0.051665] |
| damping097_fold0_seed29_easy | all | full_total | -0.483925 | [-0.838583, -0.135385] |
| damping097_fold0_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | all | total | -0.483925 | [-0.838583, -0.135385] |
| damping097_fold0_seed29_easy | all | ranking_at_product_count | 0.267641 | [0.009657, 0.537723] |
| damping097_fold0_seed29_easy | all | coverage_with_hurdle_ranking | -0.751566 | [-0.944747, -0.506701] |
| damping097_fold0_seed29_easy | all | coverage_with_product_ranking | -0.533061 | [-0.728696, -0.331982] |
| damping097_fold0_seed29_easy | all | ranking_at_hurdle_count | 0.049135 | [-0.131659, 0.225980] |
| damping097_fold0_seed29_easy | easy | full_total | 0.220805 | [0.036018, 0.434773] |
| damping097_fold0_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | easy | total | 0.220805 | [0.036018, 0.434773] |
| damping097_fold0_seed29_easy | easy | ranking_at_product_count | -0.044281 | [-0.640638, 0.376202] |
| damping097_fold0_seed29_easy | easy | coverage_with_hurdle_ranking | 0.265086 | [-0.007317, 0.746864] |
| damping097_fold0_seed29_easy | easy | coverage_with_product_ranking | 0.112306 | [-0.088153, 0.332027] |
| damping097_fold0_seed29_easy | easy | ranking_at_hurdle_count | 0.108499 | [-0.060693, 0.225644] |
| damping097_fold0_seed29_easy | hard | full_total | -0.496452 | [-0.988088, -0.019342] |
| damping097_fold0_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed29_easy | hard | total | -0.496452 | [-0.988088, -0.019342] |
| damping097_fold0_seed29_easy | hard | ranking_at_product_count | 0.355947 | [-0.017772, 0.698757] |
| damping097_fold0_seed29_easy | hard | coverage_with_hurdle_ranking | -0.852398 | [-1.004808, -0.701495] |
| damping097_fold0_seed29_easy | hard | coverage_with_product_ranking | -0.539317 | [-0.775095, -0.316422] |
| damping097_fold0_seed29_easy | hard | ranking_at_hurdle_count | 0.042865 | [-0.241884, 0.310883] |
| neural_fold0_seed43_all | all | full_total | 1.586411 | [1.168047, 2.049373] |
| neural_fold0_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | all | total | 1.586411 | [1.168047, 2.049373] |
| neural_fold0_seed43_all | all | ranking_at_product_count | -0.248988 | [-0.420484, -0.079158] |
| neural_fold0_seed43_all | all | coverage_with_hurdle_ranking | 1.835399 | [1.345239, 2.299578] |
| neural_fold0_seed43_all | all | coverage_with_product_ranking | 2.204144 | [1.245252, 3.046592] |
| neural_fold0_seed43_all | all | ranking_at_hurdle_count | -0.617732 | [-1.214168, 0.054282] |
| neural_fold0_seed43_all | easy | full_total | 4.454148 | [3.276647, 5.727468] |
| neural_fold0_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | easy | total | 4.454148 | [3.276647, 5.727468] |
| neural_fold0_seed43_all | easy | ranking_at_product_count | 0.285023 | [0.000391, 0.587427] |
| neural_fold0_seed43_all | easy | coverage_with_hurdle_ranking | 4.169125 | [2.858748, 5.444555] |
| neural_fold0_seed43_all | easy | coverage_with_product_ranking | 2.864698 | [1.789515, 4.052814] |
| neural_fold0_seed43_all | easy | ranking_at_hurdle_count | 1.589450 | [1.192773, 2.001713] |
| neural_fold0_seed43_all | hard | full_total | 1.400930 | [0.825880, 2.059835] |
| neural_fold0_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_all | hard | total | 1.400930 | [0.825880, 2.059835] |
| neural_fold0_seed43_all | hard | ranking_at_product_count | -0.342897 | [-0.607145, -0.119844] |
| neural_fold0_seed43_all | hard | coverage_with_hurdle_ranking | 1.743827 | [1.107475, 2.471854] |
| neural_fold0_seed43_all | hard | coverage_with_product_ranking | 2.718352 | [1.730226, 3.647467] |
| neural_fold0_seed43_all | hard | ranking_at_hurdle_count | -1.317422 | [-1.938978, -0.664241] |
| neural_fold0_seed43_easy | all | full_total | -1.531754 | [-2.273082, -0.823338] |
| neural_fold0_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | all | total | -1.531754 | [-2.273082, -0.823338] |
| neural_fold0_seed43_easy | all | ranking_at_product_count | -0.325614 | [-0.817651, 0.135261] |
| neural_fold0_seed43_easy | all | coverage_with_hurdle_ranking | -1.206139 | [-1.671835, -0.773932] |
| neural_fold0_seed43_easy | all | coverage_with_product_ranking | -1.339337 | [-1.930840, -0.817168] |
| neural_fold0_seed43_easy | all | ranking_at_hurdle_count | -0.192417 | [-0.391155, 0.019528] |
| neural_fold0_seed43_easy | easy | full_total | -4.002571 | [-5.885786, -1.890539] |
| neural_fold0_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | easy | total | -4.002571 | [-5.885786, -1.890539] |
| neural_fold0_seed43_easy | easy | ranking_at_product_count | 1.385762 | [1.139664, 1.696091] |
| neural_fold0_seed43_easy | easy | coverage_with_hurdle_ranking | -5.388332 | [-7.096543, -3.560451] |
| neural_fold0_seed43_easy | easy | coverage_with_product_ranking | -4.765133 | [-6.861687, -2.559433] |
| neural_fold0_seed43_easy | easy | ranking_at_hurdle_count | 0.762562 | [0.295924, 1.275628] |
| neural_fold0_seed43_easy | hard | full_total | -1.315845 | [-1.918131, -0.736883] |
| neural_fold0_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold0_seed43_easy | hard | total | -1.315845 | [-1.918131, -0.736883] |
| neural_fold0_seed43_easy | hard | ranking_at_product_count | -0.596432 | [-1.141642, -0.088142] |
| neural_fold0_seed43_easy | hard | coverage_with_hurdle_ranking | -0.719413 | [-1.263277, -0.315230] |
| neural_fold0_seed43_easy | hard | coverage_with_product_ranking | -1.145425 | [-1.666602, -0.646957] |
| neural_fold0_seed43_easy | hard | ranking_at_hurdle_count | -0.170419 | [-0.308790, -0.032307] |
| damping097_fold0_seed43_all | all | full_total | 1.029859 | [0.572893, 1.460617] |
| damping097_fold0_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | all | total | 1.029859 | [0.572893, 1.460617] |
| damping097_fold0_seed43_all | all | ranking_at_product_count | -0.542809 | [-0.905706, -0.188173] |
| damping097_fold0_seed43_all | all | coverage_with_hurdle_ranking | 1.572668 | [1.042161, 1.943827] |
| damping097_fold0_seed43_all | all | coverage_with_product_ranking | 1.145668 | [0.700122, 1.494839] |
| damping097_fold0_seed43_all | all | ranking_at_hurdle_count | -0.115808 | [-0.546021, 0.268744] |
| damping097_fold0_seed43_all | easy | full_total | -0.059783 | [-0.474676, 0.276014] |
| damping097_fold0_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | easy | total | -0.059783 | [-0.474676, 0.276014] |
| damping097_fold0_seed43_all | easy | ranking_at_product_count | 0.209461 | [0.032180, 0.409194] |
| damping097_fold0_seed43_all | easy | coverage_with_hurdle_ranking | -0.269244 | [-0.554452, -0.039315] |
| damping097_fold0_seed43_all | easy | coverage_with_product_ranking | -0.358045 | [-1.024613, 0.080616] |
| damping097_fold0_seed43_all | easy | ranking_at_hurdle_count | 0.298262 | [0.012121, 0.644387] |
| damping097_fold0_seed43_all | hard | full_total | 0.857938 | [0.393355, 1.310479] |
| damping097_fold0_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_all | hard | total | 0.857938 | [0.393355, 1.310479] |
| damping097_fold0_seed43_all | hard | ranking_at_product_count | -0.889887 | [-1.248326, -0.502743] |
| damping097_fold0_seed43_all | hard | coverage_with_hurdle_ranking | 1.747826 | [1.147748, 2.180935] |
| damping097_fold0_seed43_all | hard | coverage_with_product_ranking | 1.346338 | [0.956808, 1.661321] |
| damping097_fold0_seed43_all | hard | ranking_at_hurdle_count | -0.488400 | [-0.839177, -0.149477] |
| damping097_fold0_seed43_easy | all | full_total | -0.104841 | [-0.240334, 0.039298] |
| damping097_fold0_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | all | total | -0.104841 | [-0.240334, 0.039298] |
| damping097_fold0_seed43_easy | all | ranking_at_product_count | 0.298274 | [0.164874, 0.418796] |
| damping097_fold0_seed43_easy | all | coverage_with_hurdle_ranking | -0.403115 | [-0.494374, -0.316412] |
| damping097_fold0_seed43_easy | all | coverage_with_product_ranking | -0.234752 | [-0.357039, -0.085911] |
| damping097_fold0_seed43_easy | all | ranking_at_hurdle_count | 0.129911 | [0.015070, 0.251376] |
| damping097_fold0_seed43_easy | easy | full_total | 0.228939 | [0.034739, 0.438612] |
| damping097_fold0_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | easy | total | 0.228939 | [0.034739, 0.438612] |
| damping097_fold0_seed43_easy | easy | ranking_at_product_count | 0.093428 | [-0.166516, 0.347030] |
| damping097_fold0_seed43_easy | easy | coverage_with_hurdle_ranking | 0.135512 | [0.009048, 0.350459] |
| damping097_fold0_seed43_easy | easy | coverage_with_product_ranking | 0.178128 | [-0.006393, 0.369386] |
| damping097_fold0_seed43_easy | easy | ranking_at_hurdle_count | 0.050811 | [0.004512, 0.092127] |
| damping097_fold0_seed43_easy | hard | full_total | 0.013325 | [-0.229967, 0.300743] |
| damping097_fold0_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold0_seed43_easy | hard | total | 0.013325 | [-0.229967, 0.300743] |
| damping097_fold0_seed43_easy | hard | ranking_at_product_count | 0.442683 | [0.156980, 0.770131] |
| damping097_fold0_seed43_easy | hard | coverage_with_hurdle_ranking | -0.429357 | [-0.537823, -0.340091] |
| damping097_fold0_seed43_easy | hard | coverage_with_product_ranking | -0.193204 | [-0.296541, -0.072396] |
| damping097_fold0_seed43_easy | hard | ranking_at_hurdle_count | 0.206530 | [-0.001932, 0.420679] |
| neural_fold1_seed17_all | all | full_total | 0.464918 | [0.091506, 0.815553] |
| neural_fold1_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | all | total | 0.464918 | [0.091506, 0.815553] |
| neural_fold1_seed17_all | all | ranking_at_product_count | -0.355701 | [-0.628106, -0.111528] |
| neural_fold1_seed17_all | all | coverage_with_hurdle_ranking | 0.820619 | [0.557034, 1.094804] |
| neural_fold1_seed17_all | all | coverage_with_product_ranking | 0.784775 | [0.326135, 1.136539] |
| neural_fold1_seed17_all | all | ranking_at_hurdle_count | -0.319857 | [-0.717107, 0.131591] |
| neural_fold1_seed17_all | easy | full_total | 2.361897 | [1.708894, 3.109649] |
| neural_fold1_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | easy | total | 2.361897 | [1.708894, 3.109649] |
| neural_fold1_seed17_all | easy | ranking_at_product_count | 0.779328 | [0.593036, 0.954776] |
| neural_fold1_seed17_all | easy | coverage_with_hurdle_ranking | 1.582569 | [1.031138, 2.252461] |
| neural_fold1_seed17_all | easy | coverage_with_product_ranking | 1.511850 | [1.005498, 2.110316] |
| neural_fold1_seed17_all | easy | ranking_at_hurdle_count | 0.850047 | [0.488697, 1.359326] |
| neural_fold1_seed17_all | hard | full_total | -0.070898 | [-0.369785, 0.160137] |
| neural_fold1_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_all | hard | total | -0.070898 | [-0.369785, 0.160137] |
| neural_fold1_seed17_all | hard | ranking_at_product_count | -0.560638 | [-0.937726, -0.212087] |
| neural_fold1_seed17_all | hard | coverage_with_hurdle_ranking | 0.489739 | [0.261182, 0.746283] |
| neural_fold1_seed17_all | hard | coverage_with_product_ranking | 0.675710 | [0.208550, 1.087668] |
| neural_fold1_seed17_all | hard | ranking_at_hurdle_count | -0.746608 | [-1.216035, -0.202238] |
| neural_fold1_seed17_easy | all | full_total | -2.257041 | [-3.238540, -1.227084] |
| neural_fold1_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | all | total | -2.257041 | [-3.238540, -1.227084] |
| neural_fold1_seed17_easy | all | ranking_at_product_count | -1.323573 | [-2.903381, -0.365473] |
| neural_fold1_seed17_easy | all | coverage_with_hurdle_ranking | -0.933468 | [-2.545168, 1.539044] |
| neural_fold1_seed17_easy | all | coverage_with_product_ranking | -1.905639 | [-2.828031, -0.990879] |
| neural_fold1_seed17_easy | all | ranking_at_hurdle_count | -0.351402 | [-0.520308, -0.209774] |
| neural_fold1_seed17_easy | easy | full_total | -3.731216 | [-5.034033, -2.247014] |
| neural_fold1_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | easy | total | -3.731216 | [-5.034033, -2.247014] |
| neural_fold1_seed17_easy | easy | ranking_at_product_count | -0.739145 | [-3.364434, 0.960100] |
| neural_fold1_seed17_easy | easy | coverage_with_hurdle_ranking | -2.992071 | [-5.486847, 0.147659] |
| neural_fold1_seed17_easy | easy | coverage_with_product_ranking | -4.822709 | [-6.199708, -3.391443] |
| neural_fold1_seed17_easy | easy | ranking_at_hurdle_count | 1.091492 | [0.386567, 1.735296] |
| neural_fold1_seed17_easy | hard | full_total | -1.722394 | [-2.540852, -0.938035] |
| neural_fold1_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed17_easy | hard | total | -1.722394 | [-2.540852, -0.938035] |
| neural_fold1_seed17_easy | hard | ranking_at_product_count | -2.346849 | [-4.326970, -1.067901] |
| neural_fold1_seed17_easy | hard | coverage_with_hurdle_ranking | 0.624454 | [-0.924644, 3.154937] |
| neural_fold1_seed17_easy | hard | coverage_with_product_ranking | -1.538740 | [-2.332215, -0.797647] |
| neural_fold1_seed17_easy | hard | ranking_at_hurdle_count | -0.183654 | [-0.270969, -0.093671] |
| damping097_fold1_seed17_all | all | full_total | 0.404301 | [0.136774, 0.625836] |
| damping097_fold1_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | all | total | 0.404301 | [0.136774, 0.625836] |
| damping097_fold1_seed17_all | all | ranking_at_product_count | -0.348419 | [-0.487673, -0.161618] |
| damping097_fold1_seed17_all | all | coverage_with_hurdle_ranking | 0.752720 | [0.418510, 1.050592] |
| damping097_fold1_seed17_all | all | coverage_with_product_ranking | 0.775325 | [0.401794, 1.116713] |
| damping097_fold1_seed17_all | all | ranking_at_hurdle_count | -0.371024 | [-0.593402, -0.118543] |
| damping097_fold1_seed17_all | easy | full_total | 0.108941 | [-0.189047, 0.456014] |
| damping097_fold1_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | easy | total | 0.108941 | [-0.189047, 0.456014] |
| damping097_fold1_seed17_all | easy | ranking_at_product_count | 0.318669 | [0.059454, 0.610498] |
| damping097_fold1_seed17_all | easy | coverage_with_hurdle_ranking | -0.209728 | [-0.450354, -0.008673] |
| damping097_fold1_seed17_all | easy | coverage_with_product_ranking | -0.058941 | [-0.220400, 0.046000] |
| damping097_fold1_seed17_all | easy | ranking_at_hurdle_count | 0.167882 | [-0.171972, 0.515088] |
| damping097_fold1_seed17_all | hard | full_total | 0.154787 | [-0.145066, 0.389923] |
| damping097_fold1_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_all | hard | total | 0.154787 | [-0.145066, 0.389923] |
| damping097_fold1_seed17_all | hard | ranking_at_product_count | -0.722863 | [-0.859057, -0.548808] |
| damping097_fold1_seed17_all | hard | coverage_with_hurdle_ranking | 0.877650 | [0.517657, 1.186437] |
| damping097_fold1_seed17_all | hard | coverage_with_product_ranking | 0.958154 | [0.506440, 1.345943] |
| damping097_fold1_seed17_all | hard | ranking_at_hurdle_count | -0.803367 | [-1.060564, -0.517761] |
| damping097_fold1_seed17_easy | all | full_total | -0.416853 | [-0.733641, -0.128772] |
| damping097_fold1_seed17_easy | all | support_difference | 0.004401 | [0.000589, 0.010823] |
| damping097_fold1_seed17_easy | all | total | -0.421254 | [-0.735854, -0.138439] |
| damping097_fold1_seed17_easy | all | ranking_at_product_count | 0.139287 | [-0.034027, 0.354200] |
| damping097_fold1_seed17_easy | all | coverage_with_hurdle_ranking | -0.560541 | [-0.822634, -0.294740] |
| damping097_fold1_seed17_easy | all | coverage_with_product_ranking | -0.457840 | [-0.735707, -0.207973] |
| damping097_fold1_seed17_easy | all | ranking_at_hurdle_count | 0.036586 | [-0.027112, 0.115116] |
| damping097_fold1_seed17_easy | easy | full_total | 0.241292 | [0.032215, 0.495922] |
| damping097_fold1_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed17_easy | easy | total | 0.241292 | [0.032215, 0.495922] |
| damping097_fold1_seed17_easy | easy | ranking_at_product_count | 0.071451 | [-0.096877, 0.243044] |
| damping097_fold1_seed17_easy | easy | coverage_with_hurdle_ranking | 0.169841 | [0.065767, 0.306433] |
| damping097_fold1_seed17_easy | easy | coverage_with_product_ranking | 0.111513 | [-0.148906, 0.431039] |
| damping097_fold1_seed17_easy | easy | ranking_at_hurdle_count | 0.129780 | [-0.062336, 0.428567] |
| damping097_fold1_seed17_easy | hard | full_total | -0.207829 | [-0.411060, -0.012574] |
| damping097_fold1_seed17_easy | hard | support_difference | 0.000632 | [0.000033, 0.001560] |
| damping097_fold1_seed17_easy | hard | total | -0.208461 | [-0.411150, -0.012845] |
| damping097_fold1_seed17_easy | hard | ranking_at_product_count | 0.206406 | [0.051222, 0.394026] |
| damping097_fold1_seed17_easy | hard | coverage_with_hurdle_ranking | -0.414867 | [-0.561115, -0.255612] |
| damping097_fold1_seed17_easy | hard | coverage_with_product_ranking | -0.316335 | [-0.490265, -0.161597] |
| damping097_fold1_seed17_easy | hard | ranking_at_hurdle_count | 0.107874 | [0.037063, 0.197186] |
| neural_fold1_seed29_all | all | full_total | 0.301858 | [-0.006607, 0.652456] |
| neural_fold1_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | all | total | 0.301858 | [-0.006607, 0.652456] |
| neural_fold1_seed29_all | all | ranking_at_product_count | -0.343516 | [-0.568183, -0.099074] |
| neural_fold1_seed29_all | all | coverage_with_hurdle_ranking | 0.645374 | [0.476247, 0.813774] |
| neural_fold1_seed29_all | all | coverage_with_product_ranking | 0.612200 | [0.225698, 1.031810] |
| neural_fold1_seed29_all | all | ranking_at_hurdle_count | -0.310341 | [-0.822404, 0.271965] |
| neural_fold1_seed29_all | easy | full_total | 1.956755 | [1.339270, 2.603367] |
| neural_fold1_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | easy | total | 1.956755 | [1.339270, 2.603367] |
| neural_fold1_seed29_all | easy | ranking_at_product_count | 0.905247 | [0.600823, 1.169471] |
| neural_fold1_seed29_all | easy | coverage_with_hurdle_ranking | 1.051508 | [0.610243, 1.546068] |
| neural_fold1_seed29_all | easy | coverage_with_product_ranking | 1.025848 | [0.565821, 1.535808] |
| neural_fold1_seed29_all | easy | ranking_at_hurdle_count | 0.930907 | [0.395416, 1.571685] |
| neural_fold1_seed29_all | hard | full_total | -0.246680 | [-0.495934, -0.035406] |
| neural_fold1_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_all | hard | total | -0.246680 | [-0.495934, -0.035406] |
| neural_fold1_seed29_all | hard | ranking_at_product_count | -0.601256 | [-0.879785, -0.320848] |
| neural_fold1_seed29_all | hard | coverage_with_hurdle_ranking | 0.354576 | [0.155961, 0.569041] |
| neural_fold1_seed29_all | hard | coverage_with_product_ranking | 0.642199 | [0.314123, 1.001766] |
| neural_fold1_seed29_all | hard | ranking_at_hurdle_count | -0.888879 | [-1.391748, -0.442807] |
| neural_fold1_seed29_easy | all | full_total | -1.263662 | [-1.906940, -0.723184] |
| neural_fold1_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | all | total | -1.263662 | [-1.906940, -0.723184] |
| neural_fold1_seed29_easy | all | ranking_at_product_count | -0.960112 | [-2.182699, -0.047542] |
| neural_fold1_seed29_easy | all | coverage_with_hurdle_ranking | -0.303550 | [-0.945010, 0.358767] |
| neural_fold1_seed29_easy | all | coverage_with_product_ranking | -0.833516 | [-1.120093, -0.552668] |
| neural_fold1_seed29_easy | all | ranking_at_hurdle_count | -0.430146 | [-1.113214, -0.022379] |
| neural_fold1_seed29_easy | easy | full_total | -4.595305 | [-5.318468, -3.821971] |
| neural_fold1_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | easy | total | -4.595305 | [-5.318468, -3.821971] |
| neural_fold1_seed29_easy | easy | ranking_at_product_count | -1.196633 | [-3.215357, -0.032170] |
| neural_fold1_seed29_easy | easy | coverage_with_hurdle_ranking | -3.398671 | [-5.147214, -0.925885] |
| neural_fold1_seed29_easy | easy | coverage_with_product_ranking | -4.764096 | [-6.015514, -3.474367] |
| neural_fold1_seed29_easy | easy | ranking_at_hurdle_count | 0.168791 | [-0.610247, 0.831156] |
| neural_fold1_seed29_easy | hard | full_total | -1.142689 | [-3.129051, -0.060485] |
| neural_fold1_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed29_easy | hard | total | -1.142689 | [-3.129051, -0.060485] |
| neural_fold1_seed29_easy | hard | ranking_at_product_count | -1.267940 | [-3.407662, 0.186917] |
| neural_fold1_seed29_easy | hard | coverage_with_hurdle_ranking | 0.125251 | [-0.369103, 0.606547] |
| neural_fold1_seed29_easy | hard | coverage_with_product_ranking | -0.085581 | [-0.118824, -0.049214] |
| neural_fold1_seed29_easy | hard | ranking_at_hurdle_count | -1.057108 | [-3.071498, 0.045046] |
| damping097_fold1_seed29_all | all | full_total | 0.844597 | [0.427650, 1.192813] |
| damping097_fold1_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | all | total | 0.844597 | [0.427650, 1.192813] |
| damping097_fold1_seed29_all | all | ranking_at_product_count | -0.234384 | [-0.410113, -0.068946] |
| damping097_fold1_seed29_all | all | coverage_with_hurdle_ranking | 1.078980 | [0.533036, 1.528706] |
| damping097_fold1_seed29_all | all | coverage_with_product_ranking | 1.091903 | [0.798248, 1.391339] |
| damping097_fold1_seed29_all | all | ranking_at_hurdle_count | -0.247307 | [-0.421785, -0.084849] |
| damping097_fold1_seed29_all | easy | full_total | -0.224760 | [-0.714334, 0.180454] |
| damping097_fold1_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | easy | total | -0.224760 | [-0.714334, 0.180454] |
| damping097_fold1_seed29_all | easy | ranking_at_product_count | 0.247934 | [0.119603, 0.377426] |
| damping097_fold1_seed29_all | easy | coverage_with_hurdle_ranking | -0.472695 | [-1.040562, -0.018142] |
| damping097_fold1_seed29_all | easy | coverage_with_product_ranking | -0.220116 | [-0.525727, 0.073270] |
| damping097_fold1_seed29_all | easy | ranking_at_hurdle_count | -0.004644 | [-0.309838, 0.280641] |
| damping097_fold1_seed29_all | hard | full_total | 0.740768 | [0.397398, 1.040456] |
| damping097_fold1_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_all | hard | total | 0.740768 | [0.397398, 1.040456] |
| damping097_fold1_seed29_all | hard | ranking_at_product_count | -0.834606 | [-1.415905, -0.409875] |
| damping097_fold1_seed29_all | hard | coverage_with_hurdle_ranking | 1.575374 | [1.109755, 1.990536] |
| damping097_fold1_seed29_all | hard | coverage_with_product_ranking | 1.486851 | [1.180542, 1.781802] |
| damping097_fold1_seed29_all | hard | ranking_at_hurdle_count | -0.746083 | [-1.215324, -0.378616] |
| damping097_fold1_seed29_easy | all | full_total | -0.298983 | [-0.582336, -0.050161] |
| damping097_fold1_seed29_easy | all | support_difference | 0.000796 | [-0.000011, 0.002395] |
| damping097_fold1_seed29_easy | all | total | -0.299779 | [-0.582331, -0.051742] |
| damping097_fold1_seed29_easy | all | ranking_at_product_count | 0.316436 | [0.143494, 0.506312] |
| damping097_fold1_seed29_easy | all | coverage_with_hurdle_ranking | -0.616215 | [-0.837423, -0.400163] |
| damping097_fold1_seed29_easy | all | coverage_with_product_ranking | -0.428317 | [-0.660534, -0.244861] |
| damping097_fold1_seed29_easy | all | ranking_at_hurdle_count | 0.128539 | [0.051374, 0.224284] |
| damping097_fold1_seed29_easy | easy | full_total | 0.119071 | [-0.005879, 0.293991] |
| damping097_fold1_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed29_easy | easy | total | 0.119071 | [-0.005879, 0.293991] |
| damping097_fold1_seed29_easy | easy | ranking_at_product_count | 0.015265 | [-0.117595, 0.151096] |
| damping097_fold1_seed29_easy | easy | coverage_with_hurdle_ranking | 0.103806 | [-0.014889, 0.222821] |
| damping097_fold1_seed29_easy | easy | coverage_with_product_ranking | 0.141701 | [-0.002077, 0.318359] |
| damping097_fold1_seed29_easy | easy | ranking_at_hurdle_count | -0.022630 | [-0.081185, 0.053032] |
| damping097_fold1_seed29_easy | hard | full_total | -0.091289 | [-0.252548, 0.066719] |
| damping097_fold1_seed29_easy | hard | support_difference | -0.000009 | [-0.000026, 0.000000] |
| damping097_fold1_seed29_easy | hard | total | -0.091281 | [-0.252548, 0.066744] |
| damping097_fold1_seed29_easy | hard | ranking_at_product_count | 0.396558 | [0.236303, 0.557639] |
| damping097_fold1_seed29_easy | hard | coverage_with_hurdle_ranking | -0.487839 | [-0.648308, -0.313016] |
| damping097_fold1_seed29_easy | hard | coverage_with_product_ranking | -0.248131 | [-0.389308, -0.115995] |
| damping097_fold1_seed29_easy | hard | ranking_at_hurdle_count | 0.156850 | [0.080990, 0.249995] |
| neural_fold1_seed43_all | all | full_total | 1.106054 | [0.418290, 1.934565] |
| neural_fold1_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | all | total | 1.106054 | [0.418290, 1.934565] |
| neural_fold1_seed43_all | all | ranking_at_product_count | 0.171220 | [-0.643434, 1.214344] |
| neural_fold1_seed43_all | all | coverage_with_hurdle_ranking | 0.934834 | [0.458916, 1.377158] |
| neural_fold1_seed43_all | all | coverage_with_product_ranking | 1.081032 | [0.523446, 1.656774] |
| neural_fold1_seed43_all | all | ranking_at_hurdle_count | 0.025022 | [-0.785598, 0.993235] |
| neural_fold1_seed43_all | easy | full_total | 2.574844 | [1.376745, 3.951661] |
| neural_fold1_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | easy | total | 2.574844 | [1.376745, 3.951661] |
| neural_fold1_seed43_all | easy | ranking_at_product_count | 1.353852 | [0.431816, 2.669085] |
| neural_fold1_seed43_all | easy | coverage_with_hurdle_ranking | 1.220992 | [0.515826, 2.045146] |
| neural_fold1_seed43_all | easy | coverage_with_product_ranking | 1.572958 | [0.489731, 2.986091] |
| neural_fold1_seed43_all | easy | ranking_at_hurdle_count | 1.001885 | [0.324662, 1.973218] |
| neural_fold1_seed43_all | hard | full_total | 0.081480 | [-0.387238, 0.642127] |
| neural_fold1_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_all | hard | total | 0.081480 | [-0.387238, 0.642127] |
| neural_fold1_seed43_all | hard | ranking_at_product_count | -0.563730 | [-1.205382, 0.187139] |
| neural_fold1_seed43_all | hard | coverage_with_hurdle_ranking | 0.645210 | [0.307765, 0.990077] |
| neural_fold1_seed43_all | hard | coverage_with_product_ranking | 0.988587 | [0.623596, 1.458745] |
| neural_fold1_seed43_all | hard | ranking_at_hurdle_count | -0.907107 | [-1.697924, -0.046340] |
| neural_fold1_seed43_easy | all | full_total | -1.049446 | [-1.442124, -0.667767] |
| neural_fold1_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | all | total | -1.049446 | [-1.442124, -0.667767] |
| neural_fold1_seed43_easy | all | ranking_at_product_count | -0.474645 | [-0.921585, -0.125414] |
| neural_fold1_seed43_easy | all | coverage_with_hurdle_ranking | -0.574801 | [-1.075931, 0.100315] |
| neural_fold1_seed43_easy | all | coverage_with_product_ranking | -1.032895 | [-1.371286, -0.699457] |
| neural_fold1_seed43_easy | all | ranking_at_hurdle_count | -0.016551 | [-0.085022, 0.073898] |
| neural_fold1_seed43_easy | easy | full_total | -4.596557 | [-5.944261, -3.196239] |
| neural_fold1_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | easy | total | -4.596557 | [-5.944261, -3.196239] |
| neural_fold1_seed43_easy | easy | ranking_at_product_count | -0.117427 | [-0.649228, 0.312433] |
| neural_fold1_seed43_easy | easy | coverage_with_hurdle_ranking | -4.479130 | [-6.103875, -2.674341] |
| neural_fold1_seed43_easy | easy | coverage_with_product_ranking | -5.134972 | [-6.943947, -3.321072] |
| neural_fold1_seed43_easy | easy | ranking_at_hurdle_count | 0.538415 | [-0.013415, 1.100076] |
| neural_fold1_seed43_easy | hard | full_total | -0.131613 | [-0.186355, -0.073965] |
| neural_fold1_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold1_seed43_easy | hard | total | -0.131613 | [-0.186355, -0.073965] |
| neural_fold1_seed43_easy | hard | ranking_at_product_count | -0.049091 | [-0.417587, 0.333627] |
| neural_fold1_seed43_easy | hard | coverage_with_hurdle_ranking | -0.082521 | [-0.424540, 0.244430] |
| neural_fold1_seed43_easy | hard | coverage_with_product_ranking | -0.118407 | [-0.176857, -0.062007] |
| neural_fold1_seed43_easy | hard | ranking_at_hurdle_count | -0.013205 | [-0.058220, 0.028619] |
| damping097_fold1_seed43_all | all | full_total | 0.432570 | [0.040774, 0.709895] |
| damping097_fold1_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | all | total | 0.432570 | [0.040774, 0.709895] |
| damping097_fold1_seed43_all | all | ranking_at_product_count | -0.353467 | [-0.566485, -0.105812] |
| damping097_fold1_seed43_all | all | coverage_with_hurdle_ranking | 0.786036 | [0.327141, 1.161829] |
| damping097_fold1_seed43_all | all | coverage_with_product_ranking | 0.878384 | [0.655719, 1.110495] |
| damping097_fold1_seed43_all | all | ranking_at_hurdle_count | -0.445814 | [-0.754349, -0.150408] |
| damping097_fold1_seed43_all | easy | full_total | 0.026210 | [-0.299590, 0.306701] |
| damping097_fold1_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | easy | total | 0.026210 | [-0.299590, 0.306701] |
| damping097_fold1_seed43_all | easy | ranking_at_product_count | 0.481966 | [0.255190, 0.732070] |
| damping097_fold1_seed43_all | easy | coverage_with_hurdle_ranking | -0.455756 | [-0.944283, -0.081747] |
| damping097_fold1_seed43_all | easy | coverage_with_product_ranking | -0.144681 | [-0.379145, 0.051541] |
| damping097_fold1_seed43_all | easy | ranking_at_hurdle_count | 0.170891 | [-0.177586, 0.486897] |
| damping097_fold1_seed43_all | hard | full_total | 0.127268 | [-0.465942, 0.496287] |
| damping097_fold1_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_all | hard | total | 0.127268 | [-0.465942, 0.496287] |
| damping097_fold1_seed43_all | hard | ranking_at_product_count | -0.846881 | [-1.239600, -0.444355] |
| damping097_fold1_seed43_all | hard | coverage_with_hurdle_ranking | 0.974149 | [0.502208, 1.350738] |
| damping097_fold1_seed43_all | hard | coverage_with_product_ranking | 1.309174 | [0.942295, 1.687625] |
| damping097_fold1_seed43_all | hard | ranking_at_hurdle_count | -1.181905 | [-2.128773, -0.533910] |
| damping097_fold1_seed43_easy | all | full_total | -0.881067 | [-1.122668, -0.692194] |
| damping097_fold1_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | all | total | -0.881067 | [-1.122668, -0.692194] |
| damping097_fold1_seed43_easy | all | ranking_at_product_count | 0.061052 | [-0.282320, 0.383030] |
| damping097_fold1_seed43_easy | all | coverage_with_hurdle_ranking | -0.942118 | [-1.315981, -0.530580] |
| damping097_fold1_seed43_easy | all | coverage_with_product_ranking | -0.873822 | [-1.110672, -0.677731] |
| damping097_fold1_seed43_easy | all | ranking_at_hurdle_count | -0.007244 | [-0.086212, 0.086709] |
| damping097_fold1_seed43_easy | easy | full_total | 0.437336 | [0.072188, 0.862635] |
| damping097_fold1_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | easy | total | 0.437336 | [0.072188, 0.862635] |
| damping097_fold1_seed43_easy | easy | ranking_at_product_count | 0.013810 | [-0.246598, 0.248722] |
| damping097_fold1_seed43_easy | easy | coverage_with_hurdle_ranking | 0.423526 | [0.034738, 0.891291] |
| damping097_fold1_seed43_easy | easy | coverage_with_product_ranking | 0.476570 | [0.092364, 0.919015] |
| damping097_fold1_seed43_easy | easy | ranking_at_hurdle_count | -0.039233 | [-0.094914, 0.002843] |
| damping097_fold1_seed43_easy | hard | full_total | -0.622910 | [-0.827726, -0.431866] |
| damping097_fold1_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold1_seed43_easy | hard | total | -0.622910 | [-0.827726, -0.431866] |
| damping097_fold1_seed43_easy | hard | ranking_at_product_count | 0.255087 | [-0.047884, 0.546254] |
| damping097_fold1_seed43_easy | hard | coverage_with_hurdle_ranking | -0.877996 | [-1.107068, -0.633826] |
| damping097_fold1_seed43_easy | hard | coverage_with_product_ranking | -0.653127 | [-0.871500, -0.445856] |
| damping097_fold1_seed43_easy | hard | ranking_at_hurdle_count | 0.030217 | [-0.035859, 0.114705] |
| neural_fold2_seed17_all | all | full_total | 0.304174 | [0.189278, 0.429064] |
| neural_fold2_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | all | total | 0.304174 | [0.189278, 0.429064] |
| neural_fold2_seed17_all | all | ranking_at_product_count | 0.126114 | [0.054146, 0.200517] |
| neural_fold2_seed17_all | all | coverage_with_hurdle_ranking | 0.178061 | [0.101328, 0.251261] |
| neural_fold2_seed17_all | all | coverage_with_product_ranking | 0.297341 | [0.149610, 0.502547] |
| neural_fold2_seed17_all | all | ranking_at_hurdle_count | 0.006834 | [-0.106702, 0.093638] |
| neural_fold2_seed17_all | easy | full_total | 0.813367 | [0.333665, 1.320687] |
| neural_fold2_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | easy | total | 0.813367 | [0.333665, 1.320687] |
| neural_fold2_seed17_all | easy | ranking_at_product_count | -0.044010 | [-0.215827, 0.130846] |
| neural_fold2_seed17_all | easy | coverage_with_hurdle_ranking | 0.857377 | [0.428561, 1.307108] |
| neural_fold2_seed17_all | easy | coverage_with_product_ranking | 0.546234 | [-0.002727, 1.108747] |
| neural_fold2_seed17_all | easy | ranking_at_hurdle_count | 0.267133 | [-0.023668, 0.552824] |
| neural_fold2_seed17_all | hard | full_total | 0.267731 | [0.141676, 0.423032] |
| neural_fold2_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_all | hard | total | 0.267731 | [0.141676, 0.423032] |
| neural_fold2_seed17_all | hard | ranking_at_product_count | 0.142947 | [0.058999, 0.228340] |
| neural_fold2_seed17_all | hard | coverage_with_hurdle_ranking | 0.124784 | [0.028457, 0.224077] |
| neural_fold2_seed17_all | hard | coverage_with_product_ranking | 0.322181 | [0.130847, 0.569109] |
| neural_fold2_seed17_all | hard | ranking_at_hurdle_count | -0.054450 | [-0.292855, 0.097420] |
| neural_fold2_seed17_easy | all | full_total | -1.376763 | [-2.046740, -0.580989] |
| neural_fold2_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | all | total | -1.376763 | [-2.046740, -0.580989] |
| neural_fold2_seed17_easy | all | ranking_at_product_count | 0.621635 | [0.003517, 1.157812] |
| neural_fold2_seed17_easy | all | coverage_with_hurdle_ranking | -1.998398 | [-2.711638, -1.305999] |
| neural_fold2_seed17_easy | all | coverage_with_product_ranking | -1.818821 | [-2.486817, -1.078783] |
| neural_fold2_seed17_easy | all | ranking_at_hurdle_count | 0.442058 | [0.210421, 0.680476] |
| neural_fold2_seed17_easy | easy | full_total | 4.476349 | [0.466191, 8.918091] |
| neural_fold2_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | easy | total | 4.476349 | [0.466191, 8.918091] |
| neural_fold2_seed17_easy | easy | ranking_at_product_count | 4.553325 | [2.460761, 6.435583] |
| neural_fold2_seed17_easy | easy | coverage_with_hurdle_ranking | -0.076976 | [-4.772528, 4.638976] |
| neural_fold2_seed17_easy | easy | coverage_with_product_ranking | 4.237473 | [0.189197, 8.671550] |
| neural_fold2_seed17_easy | easy | ranking_at_hurdle_count | 0.238876 | [0.042390, 0.399598] |
| neural_fold2_seed17_easy | hard | full_total | -1.052165 | [-1.701544, -0.292405] |
| neural_fold2_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed17_easy | hard | total | -1.052165 | [-1.701544, -0.292405] |
| neural_fold2_seed17_easy | hard | ranking_at_product_count | 0.976055 | [0.365636, 1.499894] |
| neural_fold2_seed17_easy | hard | coverage_with_hurdle_ranking | -2.028221 | [-2.781790, -1.240912] |
| neural_fold2_seed17_easy | hard | coverage_with_product_ranking | -1.587099 | [-2.281446, -0.805390] |
| neural_fold2_seed17_easy | hard | ranking_at_hurdle_count | 0.534934 | [0.237508, 0.827030] |
| damping097_fold2_seed17_all | all | full_total | -0.211545 | [-0.740546, 0.305238] |
| damping097_fold2_seed17_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | all | total | -0.211545 | [-0.740546, 0.305238] |
| damping097_fold2_seed17_all | all | ranking_at_product_count | -1.402770 | [-1.715613, -1.091543] |
| damping097_fold2_seed17_all | all | coverage_with_hurdle_ranking | 1.191225 | [0.918440, 1.443516] |
| damping097_fold2_seed17_all | all | coverage_with_product_ranking | 1.048424 | [0.857052, 1.253601] |
| damping097_fold2_seed17_all | all | ranking_at_hurdle_count | -1.259970 | [-1.684987, -0.887126] |
| damping097_fold2_seed17_all | easy | full_total | 0.223952 | [-0.162172, 0.675294] |
| damping097_fold2_seed17_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | easy | total | 0.223952 | [-0.162172, 0.675294] |
| damping097_fold2_seed17_all | easy | ranking_at_product_count | 0.398387 | [0.145309, 0.746734] |
| damping097_fold2_seed17_all | easy | coverage_with_hurdle_ranking | -0.174434 | [-0.539455, 0.082150] |
| damping097_fold2_seed17_all | easy | coverage_with_product_ranking | 0.102716 | [-0.108637, 0.277791] |
| damping097_fold2_seed17_all | easy | ranking_at_hurdle_count | 0.121236 | [-0.126065, 0.492727] |
| damping097_fold2_seed17_all | hard | full_total | -0.661845 | [-1.341430, -0.075528] |
| damping097_fold2_seed17_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_all | hard | total | -0.661845 | [-1.341430, -0.075528] |
| damping097_fold2_seed17_all | hard | ranking_at_product_count | -1.825495 | [-2.375738, -1.393252] |
| damping097_fold2_seed17_all | hard | coverage_with_hurdle_ranking | 1.163650 | [0.917009, 1.390223] |
| damping097_fold2_seed17_all | hard | coverage_with_product_ranking | 1.002275 | [0.837855, 1.164310] |
| damping097_fold2_seed17_all | hard | ranking_at_hurdle_count | -1.664120 | [-2.345526, -1.137331] |
| damping097_fold2_seed17_easy | all | full_total | -3.550015 | [-4.088860, -2.871038] |
| damping097_fold2_seed17_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | all | total | -3.550015 | [-4.088860, -2.871038] |
| damping097_fold2_seed17_easy | all | ranking_at_product_count | -0.302284 | [-0.571589, 0.076589] |
| damping097_fold2_seed17_easy | all | coverage_with_hurdle_ranking | -3.247731 | [-3.861141, -2.485253] |
| damping097_fold2_seed17_easy | all | coverage_with_product_ranking | -2.844907 | [-3.293513, -2.309820] |
| damping097_fold2_seed17_easy | all | ranking_at_hurdle_count | -0.705109 | [-0.955147, -0.443740] |
| damping097_fold2_seed17_easy | easy | full_total | 0.820243 | [0.210104, 1.596071] |
| damping097_fold2_seed17_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | easy | total | 0.820243 | [0.210104, 1.596071] |
| damping097_fold2_seed17_easy | easy | ranking_at_product_count | 0.032762 | [-0.109521, 0.184171] |
| damping097_fold2_seed17_easy | easy | coverage_with_hurdle_ranking | 0.787481 | [0.176674, 1.604200] |
| damping097_fold2_seed17_easy | easy | coverage_with_product_ranking | -0.965207 | [-1.630370, -0.175074] |
| damping097_fold2_seed17_easy | easy | ranking_at_hurdle_count | 1.785451 | [1.364219, 2.251073] |
| damping097_fold2_seed17_easy | hard | full_total | -3.875338 | [-4.475504, -3.166276] |
| damping097_fold2_seed17_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed17_easy | hard | total | -3.875338 | [-4.475504, -3.166276] |
| damping097_fold2_seed17_easy | hard | ranking_at_product_count | -0.498687 | [-0.823651, -0.013280] |
| damping097_fold2_seed17_easy | hard | coverage_with_hurdle_ranking | -3.376652 | [-4.120004, -2.497013] |
| damping097_fold2_seed17_easy | hard | coverage_with_product_ranking | -2.994674 | [-3.483407, -2.435296] |
| damping097_fold2_seed17_easy | hard | ranking_at_hurdle_count | -0.880664 | [-1.150585, -0.599775] |
| neural_fold2_seed29_all | all | full_total | -0.104979 | [-0.297232, 0.057674] |
| neural_fold2_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | all | total | -0.104979 | [-0.297232, 0.057674] |
| neural_fold2_seed29_all | all | ranking_at_product_count | -0.206645 | [-0.404307, -0.043586] |
| neural_fold2_seed29_all | all | coverage_with_hurdle_ranking | 0.101666 | [0.045325, 0.154154] |
| neural_fold2_seed29_all | all | coverage_with_product_ranking | 0.151743 | [0.104107, 0.205723] |
| neural_fold2_seed29_all | all | ranking_at_hurdle_count | -0.256722 | [-0.469303, -0.072069] |
| neural_fold2_seed29_all | easy | full_total | 0.577489 | [0.038945, 1.152208] |
| neural_fold2_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | easy | total | 0.577489 | [0.038945, 1.152208] |
| neural_fold2_seed29_all | easy | ranking_at_product_count | 0.144702 | [-0.247349, 0.518909] |
| neural_fold2_seed29_all | easy | coverage_with_hurdle_ranking | 0.432787 | [0.023929, 0.950178] |
| neural_fold2_seed29_all | easy | coverage_with_product_ranking | 0.180332 | [-0.265524, 0.710605] |
| neural_fold2_seed29_all | easy | ranking_at_hurdle_count | 0.397158 | [-0.173939, 0.889701] |
| neural_fold2_seed29_all | hard | full_total | -0.172918 | [-0.329397, -0.039598] |
| neural_fold2_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_all | hard | total | -0.172918 | [-0.329397, -0.039598] |
| neural_fold2_seed29_all | hard | ranking_at_product_count | -0.264664 | [-0.451291, -0.115943] |
| neural_fold2_seed29_all | hard | coverage_with_hurdle_ranking | 0.091746 | [0.016953, 0.163608] |
| neural_fold2_seed29_all | hard | coverage_with_product_ranking | 0.144970 | [0.073352, 0.219857] |
| neural_fold2_seed29_all | hard | ranking_at_hurdle_count | -0.317888 | [-0.531196, -0.124487] |
| neural_fold2_seed29_easy | all | full_total | -1.519024 | [-2.341149, -0.533084] |
| neural_fold2_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | all | total | -1.519024 | [-2.341149, -0.533084] |
| neural_fold2_seed29_easy | all | ranking_at_product_count | 0.689703 | [0.112642, 1.261550] |
| neural_fold2_seed29_easy | all | coverage_with_hurdle_ranking | -2.208727 | [-2.840284, -1.533684] |
| neural_fold2_seed29_easy | all | coverage_with_product_ranking | -2.255431 | [-3.025304, -1.414493] |
| neural_fold2_seed29_easy | all | ranking_at_hurdle_count | 0.736406 | [0.382770, 1.106205] |
| neural_fold2_seed29_easy | easy | full_total | 4.787575 | [0.880327, 8.857691] |
| neural_fold2_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | easy | total | 4.787575 | [0.880327, 8.857691] |
| neural_fold2_seed29_easy | easy | ranking_at_product_count | 4.170807 | [2.076760, 6.234240] |
| neural_fold2_seed29_easy | easy | coverage_with_hurdle_ranking | 0.616768 | [-3.970793, 5.346440] |
| neural_fold2_seed29_easy | easy | coverage_with_product_ranking | 3.924697 | [0.047128, 7.897109] |
| neural_fold2_seed29_easy | easy | ranking_at_hurdle_count | 0.862878 | [0.574872, 1.115944] |
| neural_fold2_seed29_easy | hard | full_total | -1.326871 | [-2.128434, -0.361569] |
| neural_fold2_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed29_easy | hard | total | -1.326871 | [-2.128434, -0.361569] |
| neural_fold2_seed29_easy | hard | ranking_at_product_count | 0.867203 | [0.333178, 1.440158] |
| neural_fold2_seed29_easy | hard | coverage_with_hurdle_ranking | -2.194074 | [-2.895248, -1.462824] |
| neural_fold2_seed29_easy | hard | coverage_with_product_ranking | -2.162162 | [-3.017731, -1.217907] |
| neural_fold2_seed29_easy | hard | ranking_at_hurdle_count | 0.835291 | [0.430962, 1.216467] |
| damping097_fold2_seed29_all | all | full_total | -0.253745 | [-0.923955, 0.292701] |
| damping097_fold2_seed29_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | all | total | -0.253745 | [-0.923955, 0.292701] |
| damping097_fold2_seed29_all | all | ranking_at_product_count | -1.213846 | [-1.589442, -0.879892] |
| damping097_fold2_seed29_all | all | coverage_with_hurdle_ranking | 0.960101 | [0.621169, 1.236435] |
| damping097_fold2_seed29_all | all | coverage_with_product_ranking | 0.816203 | [0.518008, 1.093809] |
| damping097_fold2_seed29_all | all | ranking_at_hurdle_count | -1.069948 | [-1.519360, -0.712855] |
| damping097_fold2_seed29_all | easy | full_total | 0.112032 | [-0.212364, 0.539120] |
| damping097_fold2_seed29_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | easy | total | 0.112032 | [-0.212364, 0.539120] |
| damping097_fold2_seed29_all | easy | ranking_at_product_count | 0.274746 | [0.039627, 0.624296] |
| damping097_fold2_seed29_all | easy | coverage_with_hurdle_ranking | -0.162715 | [-0.438222, 0.006159] |
| damping097_fold2_seed29_all | easy | coverage_with_product_ranking | 0.026424 | [-0.130923, 0.137030] |
| damping097_fold2_seed29_all | easy | ranking_at_hurdle_count | 0.085607 | [-0.134055, 0.453211] |
| damping097_fold2_seed29_all | hard | full_total | -0.704208 | [-1.588775, -0.046567] |
| damping097_fold2_seed29_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_all | hard | total | -0.704208 | [-1.588775, -0.046567] |
| damping097_fold2_seed29_all | hard | ranking_at_product_count | -1.596887 | [-2.219315, -1.128112] |
| damping097_fold2_seed29_all | hard | coverage_with_hurdle_ranking | 0.892679 | [0.564630, 1.166167] |
| damping097_fold2_seed29_all | hard | coverage_with_product_ranking | 0.754551 | [0.489154, 0.993907] |
| damping097_fold2_seed29_all | hard | ranking_at_hurdle_count | -1.458759 | [-2.151513, -0.954631] |
| damping097_fold2_seed29_easy | all | full_total | -3.858384 | [-4.451016, -3.186295] |
| damping097_fold2_seed29_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | all | total | -3.858384 | [-4.451016, -3.186295] |
| damping097_fold2_seed29_easy | all | ranking_at_product_count | -0.499551 | [-0.705156, -0.236177] |
| damping097_fold2_seed29_easy | all | coverage_with_hurdle_ranking | -3.358832 | [-4.000594, -2.571051] |
| damping097_fold2_seed29_easy | all | coverage_with_product_ranking | -3.091738 | [-3.669775, -2.392817] |
| damping097_fold2_seed29_easy | all | ranking_at_hurdle_count | -0.766646 | [-0.947639, -0.573874] |
| damping097_fold2_seed29_easy | easy | full_total | 0.739440 | [0.174408, 1.449712] |
| damping097_fold2_seed29_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | easy | total | 0.739440 | [0.174408, 1.449712] |
| damping097_fold2_seed29_easy | easy | ranking_at_product_count | -0.020037 | [-0.173604, 0.135939] |
| damping097_fold2_seed29_easy | easy | coverage_with_hurdle_ranking | 0.759478 | [0.148091, 1.553299] |
| damping097_fold2_seed29_easy | easy | coverage_with_product_ranking | -0.779154 | [-1.471476, 0.026468] |
| damping097_fold2_seed29_easy | easy | ranking_at_hurdle_count | 1.518595 | [1.187561, 1.860008] |
| damping097_fold2_seed29_easy | hard | full_total | -4.220166 | [-4.860113, -3.553703] |
| damping097_fold2_seed29_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed29_easy | hard | total | -4.220166 | [-4.860113, -3.553703] |
| damping097_fold2_seed29_easy | hard | ranking_at_product_count | -0.761326 | [-1.058539, -0.402388] |
| damping097_fold2_seed29_easy | hard | coverage_with_hurdle_ranking | -3.458839 | [-4.204022, -2.632840] |
| damping097_fold2_seed29_easy | hard | coverage_with_product_ranking | -3.278027 | [-3.922054, -2.540365] |
| damping097_fold2_seed29_easy | hard | ranking_at_hurdle_count | -0.942138 | [-1.150087, -0.688699] |
| neural_fold2_seed43_all | all | full_total | 0.323029 | [0.165690, 0.491844] |
| neural_fold2_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | all | total | 0.323029 | [0.165690, 0.491844] |
| neural_fold2_seed43_all | all | ranking_at_product_count | 0.176428 | [0.076667, 0.274880] |
| neural_fold2_seed43_all | all | coverage_with_hurdle_ranking | 0.146601 | [0.072323, 0.254566] |
| neural_fold2_seed43_all | all | coverage_with_product_ranking | 0.139302 | [0.086937, 0.201451] |
| neural_fold2_seed43_all | all | ranking_at_hurdle_count | 0.183727 | [0.079243, 0.294488] |
| neural_fold2_seed43_all | easy | full_total | 0.370656 | [-0.017994, 0.802067] |
| neural_fold2_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | easy | total | 0.370656 | [-0.017994, 0.802067] |
| neural_fold2_seed43_all | easy | ranking_at_product_count | -0.040792 | [-0.203792, 0.111199] |
| neural_fold2_seed43_all | easy | coverage_with_hurdle_ranking | 0.411448 | [0.017794, 0.797451] |
| neural_fold2_seed43_all | easy | coverage_with_product_ranking | 0.358195 | [0.068044, 0.605837] |
| neural_fold2_seed43_all | easy | ranking_at_hurdle_count | 0.012461 | [-0.250854, 0.311528] |
| neural_fold2_seed43_all | hard | full_total | 0.363618 | [0.174628, 0.556413] |
| neural_fold2_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_all | hard | total | 0.363618 | [0.174628, 0.556413] |
| neural_fold2_seed43_all | hard | ranking_at_product_count | 0.178849 | [0.048060, 0.316728] |
| neural_fold2_seed43_all | hard | coverage_with_hurdle_ranking | 0.184769 | [0.064814, 0.352391] |
| neural_fold2_seed43_all | hard | coverage_with_product_ranking | 0.137786 | [0.077415, 0.198167] |
| neural_fold2_seed43_all | hard | ranking_at_hurdle_count | 0.225832 | [0.093132, 0.362152] |
| neural_fold2_seed43_easy | all | full_total | -1.245604 | [-1.779304, -0.624259] |
| neural_fold2_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | all | total | -1.245604 | [-1.779304, -0.624259] |
| neural_fold2_seed43_easy | all | ranking_at_product_count | 0.269600 | [-0.187855, 0.697434] |
| neural_fold2_seed43_easy | all | coverage_with_hurdle_ranking | -1.515203 | [-2.000267, -1.020380] |
| neural_fold2_seed43_easy | all | coverage_with_product_ranking | -1.708291 | [-2.299769, -1.000500] |
| neural_fold2_seed43_easy | all | ranking_at_hurdle_count | 0.462687 | [0.244543, 0.681007] |
| neural_fold2_seed43_easy | easy | full_total | 4.313788 | [0.629570, 8.182941] |
| neural_fold2_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | easy | total | 4.313788 | [0.629570, 8.182941] |
| neural_fold2_seed43_easy | easy | ranking_at_product_count | 4.868794 | [2.970867, 6.559673] |
| neural_fold2_seed43_easy | easy | coverage_with_hurdle_ranking | -0.555006 | [-4.074237, 2.986376] |
| neural_fold2_seed43_easy | easy | coverage_with_product_ranking | 4.137520 | [0.547074, 7.901605] |
| neural_fold2_seed43_easy | easy | ranking_at_hurdle_count | 0.176268 | [-0.094305, 0.431302] |
| neural_fold2_seed43_easy | hard | full_total | -0.942894 | [-1.391776, -0.383996] |
| neural_fold2_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| neural_fold2_seed43_easy | hard | total | -0.942894 | [-1.391776, -0.383996] |
| neural_fold2_seed43_easy | hard | ranking_at_product_count | 0.524482 | [0.065532, 0.975036] |
| neural_fold2_seed43_easy | hard | coverage_with_hurdle_ranking | -1.467377 | [-2.001994, -0.921792] |
| neural_fold2_seed43_easy | hard | coverage_with_product_ranking | -1.533304 | [-2.152130, -0.811463] |
| neural_fold2_seed43_easy | hard | ranking_at_hurdle_count | 0.590410 | [0.308192, 0.862283] |
| damping097_fold2_seed43_all | all | full_total | -0.034282 | [-0.473199, 0.395075] |
| damping097_fold2_seed43_all | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | all | total | -0.034282 | [-0.473199, 0.395075] |
| damping097_fold2_seed43_all | all | ranking_at_product_count | -1.203890 | [-1.447758, -0.957374] |
| damping097_fold2_seed43_all | all | coverage_with_hurdle_ranking | 1.169608 | [0.881243, 1.416355] |
| damping097_fold2_seed43_all | all | coverage_with_product_ranking | 1.094435 | [0.885339, 1.278317] |
| damping097_fold2_seed43_all | all | ranking_at_hurdle_count | -1.128717 | [-1.496196, -0.813939] |
| damping097_fold2_seed43_all | easy | full_total | 0.150339 | [-0.088396, 0.394304] |
| damping097_fold2_seed43_all | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | easy | total | 0.150339 | [-0.088396, 0.394304] |
| damping097_fold2_seed43_all | easy | ranking_at_product_count | 0.243960 | [0.132543, 0.382087] |
| damping097_fold2_seed43_all | easy | coverage_with_hurdle_ranking | -0.093621 | [-0.384551, 0.106702] |
| damping097_fold2_seed43_all | easy | coverage_with_product_ranking | -0.111330 | [-0.471794, 0.190953] |
| damping097_fold2_seed43_all | easy | ranking_at_hurdle_count | 0.261669 | [0.066159, 0.590482] |
| damping097_fold2_seed43_all | hard | full_total | -0.468057 | [-1.018806, 0.033319] |
| damping097_fold2_seed43_all | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_all | hard | total | -0.468057 | [-1.018806, 0.033319] |
| damping097_fold2_seed43_all | hard | ranking_at_product_count | -1.564882 | [-1.967593, -1.210709] |
| damping097_fold2_seed43_all | hard | coverage_with_hurdle_ranking | 1.096824 | [0.825793, 1.350302] |
| damping097_fold2_seed43_all | hard | coverage_with_product_ranking | 1.074178 | [0.892893, 1.213545] |
| damping097_fold2_seed43_all | hard | ranking_at_hurdle_count | -1.542235 | [-2.094907, -1.078931] |
| damping097_fold2_seed43_easy | all | full_total | -3.213873 | [-3.877408, -2.469190] |
| damping097_fold2_seed43_easy | all | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | all | total | -3.213873 | [-3.877408, -2.469190] |
| damping097_fold2_seed43_easy | all | ranking_at_product_count | -0.474906 | [-0.831513, 0.001575] |
| damping097_fold2_seed43_easy | all | coverage_with_hurdle_ranking | -2.738967 | [-3.542754, -1.864871] |
| damping097_fold2_seed43_easy | all | coverage_with_product_ranking | -2.354050 | [-2.885355, -1.772950] |
| damping097_fold2_seed43_easy | all | ranking_at_hurdle_count | -0.859823 | [-1.137951, -0.602988] |
| damping097_fold2_seed43_easy | easy | full_total | 0.642183 | [0.171508, 1.219969] |
| damping097_fold2_seed43_easy | easy | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | easy | total | 0.642183 | [0.171508, 1.219969] |
| damping097_fold2_seed43_easy | easy | ranking_at_product_count | 0.027568 | [-0.144816, 0.206031] |
| damping097_fold2_seed43_easy | easy | coverage_with_hurdle_ranking | 0.614615 | [0.111566, 1.269914] |
| damping097_fold2_seed43_easy | easy | coverage_with_product_ranking | -1.137395 | [-1.798007, -0.441490] |
| damping097_fold2_seed43_easy | easy | ranking_at_hurdle_count | 1.779577 | [1.310396, 2.213341] |
| damping097_fold2_seed43_easy | hard | full_total | -3.567527 | [-4.239263, -2.850442] |
| damping097_fold2_seed43_easy | hard | support_difference | 0.000000 | [0.000000, 0.000000] |
| damping097_fold2_seed43_easy | hard | total | -3.567527 | [-4.239263, -2.850442] |
| damping097_fold2_seed43_easy | hard | ranking_at_product_count | -0.848386 | [-1.308832, -0.216197] |
| damping097_fold2_seed43_easy | hard | coverage_with_hurdle_ranking | -2.719141 | [-3.658703, -1.726094] |
| damping097_fold2_seed43_easy | hard | coverage_with_product_ranking | -2.525553 | [-3.080419, -1.924306] |
| damping097_fold2_seed43_easy | hard | ranking_at_hurdle_count | -1.041974 | [-1.357068, -0.762087] |

Detector-track image pixels, obs8/pred12 rawstride12. Conditional, dependent, unadjusted locality CIs.
Not seconds, metric, human gold, physical safety, independent confirmation, true 3D or foundation.
No deployment promotion, Stage5C or SMC. Initial support mismatch is preserved in the parent amendment.
