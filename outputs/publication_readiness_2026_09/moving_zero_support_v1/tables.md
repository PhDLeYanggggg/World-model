# Moving Zero-CV Support: Complete Diagnostic Tables

Fresh distances and raw-annotation checks; frozen verified forecasts/forests. No new training or policy.
Four exposed source sites, three seeds, obs8/pred12 stride12 annotation pixels.

## Seven Cases, Three Scoped Tracks

| Row | Site / track | Last step px | Past CV max residual px | Sampled future CV ADE px | Dense raw future CV ADE px | Past generated / 8 | Future generated / 12 | Past rows bracketed by later control |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 14612 | coupa/video2:87 | 16.500 | 5.500 | 0.000 | 0.8125 | 8 | 12 | 1 |
| 14613 | coupa/video2:87 | 16.500 | 5.500 | 0.000 | 0.8125 | 8 | 12 | 2 |
| 14614 | coupa/video2:87 | 16.500 | 5.500 | 0.000 | 0.8125 | 8 | 12 | 3 |
| 14615 | coupa/video2:87 | 16.500 | 3.000 | 0.000 | 0.8125 | 8 | 12 | 4 |
| 14616 | coupa/video2:87 | 16.500 | 3.000 | 0.000 | 0.8125 | 8 | 12 | 5 |
| 125641 | hyang/video2:93 | 7.000 | 4.123 | 0.000 | 0.5000 | 8 | 12 | 2 |
| 129359 | hyang/video2:158 | 6.000 | 10.000 | 0.000 | 0.6250 | 7 | 12 | 0 |

Dense raw-frame errors are a provenance diagnostic, not a replacement main metric.
Generated flags and following controls are retrospective labels, not inference features.
Five coupa rows overlap on one track; the three tracks are not three independent datasets.

## All Case/Feature/Action/Seed Comparisons

| View | Action | Features | Row | Closest RMS feature distance | Nearest zero-event rank | Source rank fraction | Exact matches | Zero events / 512 | Tracks / 512 | Mean relative gain / 512 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coupa_seed17 | damped_velocity_005 | history | 14612 | 0.0038305937 | 7921 | 0.084934 | 0 | 0 | 316 | -0.131278 |
| coupa_seed17 | damped_velocity_005 | history | 14613 | 0.0028394235 | 7052 | 0.075616 | 0 | 0 | 319 | -0.229037 |
| coupa_seed17 | damped_velocity_005 | history | 14614 | 0.0045491541 | 8706 | 0.093351 | 0 | 0 | 278 | -0.310378 |
| coupa_seed17 | damped_velocity_005 | history | 14615 | 2.5513475e-12 | 11109 | 0.119117 | 0 | 0 | 263 | -0.338898 |
| coupa_seed17 | damped_velocity_005 | history | 14616 | 2.5513475e-12 | 12720 | 0.136391 | 0 | 0 | 238 | -0.400678 |
| coupa_seed17 | damped_velocity_005 | full_risk | 14612 | 0.0061670557 | 42151 | 0.451968 | 0 | 0 | 128 | -0.143311 |
| coupa_seed17 | damped_velocity_005 | full_risk | 14613 | 0.0073632239 | 42182 | 0.452301 | 0 | 0 | 126 | -0.155360 |
| coupa_seed17 | damped_velocity_005 | full_risk | 14614 | 0.0064391195 | 42361 | 0.454220 | 0 | 0 | 126 | -0.160893 |
| coupa_seed17 | damped_velocity_005 | full_risk | 14615 | 0.0056334661 | 42579 | 0.456557 | 0 | 0 | 127 | -0.166384 |
| coupa_seed17 | damped_velocity_005 | full_risk | 14616 | 0.0047416152 | 42770 | 0.458605 | 0 | 0 | 127 | -0.175635 |
| coupa_seed17 | transformer | history | 14612 | 0.0038305937 | 7921 | 0.084934 | 0 | 0 | 316 | 0.350158 |
| coupa_seed17 | transformer | history | 14613 | 0.0028394235 | 7052 | 0.075616 | 0 | 0 | 319 | 0.238173 |
| coupa_seed17 | transformer | history | 14614 | 0.0045491541 | 8706 | 0.093351 | 0 | 0 | 278 | 0.106448 |
| coupa_seed17 | transformer | history | 14615 | 2.5513475e-12 | 11109 | 0.119117 | 0 | 0 | 263 | 0.030828 |
| coupa_seed17 | transformer | history | 14616 | 2.5513475e-12 | 12720 | 0.136391 | 0 | 0 | 238 | -0.039644 |
| coupa_seed17 | transformer | full_risk | 14612 | 0.024838881 | 31711 | 0.340024 | 0 | 0 | 132 | 0.240906 |
| coupa_seed17 | transformer | full_risk | 14613 | 0.025705487 | 30874 | 0.331049 | 0 | 0 | 132 | 0.223888 |
| coupa_seed17 | transformer | full_risk | 14614 | 0.02357459 | 30043 | 0.322139 | 0 | 0 | 130 | 0.221010 |
| coupa_seed17 | transformer | full_risk | 14615 | 0.024209163 | 27268 | 0.292384 | 0 | 0 | 128 | 0.211684 |
| coupa_seed17 | transformer | full_risk | 14616 | 0.025382958 | 26617 | 0.285403 | 0 | 0 | 127 | 0.200632 |
| coupa_seed17 | eqmotion | history | 14612 | 0.0038305937 | 7921 | 0.084934 | 0 | 0 | 316 | 0.294560 |
| coupa_seed17 | eqmotion | history | 14613 | 0.0028394235 | 7052 | 0.075616 | 0 | 0 | 319 | 0.204384 |
| coupa_seed17 | eqmotion | history | 14614 | 0.0045491541 | 8706 | 0.093351 | 0 | 0 | 278 | 0.069582 |
| coupa_seed17 | eqmotion | history | 14615 | 2.5513475e-12 | 11109 | 0.119117 | 0 | 0 | 263 | -0.026211 |
| coupa_seed17 | eqmotion | history | 14616 | 2.5513475e-12 | 12720 | 0.136391 | 0 | 0 | 238 | -0.078777 |
| coupa_seed17 | eqmotion | full_risk | 14612 | 0.0054849578 | 34225 | 0.366981 | 0 | 0 | 133 | 0.325937 |
| coupa_seed17 | eqmotion | full_risk | 14613 | 0.005232988 | 27198 | 0.291633 | 0 | 0 | 132 | 0.271871 |
| coupa_seed17 | eqmotion | full_risk | 14614 | 0.0067500392 | 27234 | 0.292019 | 0 | 0 | 131 | 0.269296 |
| coupa_seed17 | eqmotion | full_risk | 14615 | 0.0060054794 | 22566 | 0.241966 | 0 | 0 | 124 | 0.260955 |
| coupa_seed17 | eqmotion | full_risk | 14616 | 0.006124352 | 23320 | 0.250051 | 0 | 0 | 124 | 0.263431 |
| coupa_seed29 | damped_velocity_005 | history | 14612 | 0.0038305937 | 7905 | 0.084880 | 0 | 0 | 313 | -0.134028 |
| coupa_seed29 | damped_velocity_005 | history | 14613 | 0.0028394235 | 7030 | 0.075485 | 0 | 0 | 321 | -0.217869 |
| coupa_seed29 | damped_velocity_005 | history | 14614 | 0.0045491541 | 8688 | 0.093288 | 0 | 0 | 281 | -0.316224 |
| coupa_seed29 | damped_velocity_005 | history | 14615 | 2.5513475e-12 | 11090 | 0.119080 | 0 | 0 | 266 | -0.345244 |
| coupa_seed29 | damped_velocity_005 | history | 14616 | 2.5513475e-12 | 12706 | 0.136431 | 0 | 0 | 244 | -0.395518 |
| coupa_seed29 | damped_velocity_005 | full_risk | 14612 | 0.0061670557 | 42071 | 0.451740 | 0 | 0 | 128 | -0.143485 |
| coupa_seed29 | damped_velocity_005 | full_risk | 14613 | 0.0073632239 | 42100 | 0.452051 | 0 | 0 | 126 | -0.155534 |
| coupa_seed29 | damped_velocity_005 | full_risk | 14614 | 0.0064391195 | 42279 | 0.453973 | 0 | 0 | 125 | -0.161067 |
| coupa_seed29 | damped_velocity_005 | full_risk | 14615 | 0.0056334661 | 42495 | 0.456293 | 0 | 0 | 126 | -0.166558 |
| coupa_seed29 | damped_velocity_005 | full_risk | 14616 | 0.0047416152 | 42687 | 0.458354 | 0 | 0 | 126 | -0.175809 |
| coupa_seed29 | transformer | history | 14612 | 0.0038305937 | 7905 | 0.084880 | 0 | 0 | 313 | 0.318618 |
| coupa_seed29 | transformer | history | 14613 | 0.0028394235 | 7030 | 0.075485 | 0 | 0 | 321 | 0.216728 |
| coupa_seed29 | transformer | history | 14614 | 0.0045491541 | 8688 | 0.093288 | 0 | 0 | 281 | 0.088258 |
| coupa_seed29 | transformer | history | 14615 | 2.5513475e-12 | 11090 | 0.119080 | 0 | 0 | 266 | 0.041408 |
| coupa_seed29 | transformer | history | 14616 | 2.5513475e-12 | 12706 | 0.136431 | 0 | 0 | 244 | -0.045931 |
| coupa_seed29 | transformer | full_risk | 14612 | 0.018194122 | 32710 | 0.351226 | 0 | 0 | 131 | 0.145380 |
| coupa_seed29 | transformer | full_risk | 14613 | 0.02271027 | 32305 | 0.346877 | 0 | 0 | 131 | 0.136587 |
| coupa_seed29 | transformer | full_risk | 14614 | 0.024883502 | 30486 | 0.327345 | 0 | 0 | 134 | 0.122741 |
| coupa_seed29 | transformer | full_risk | 14615 | 0.015946576 | 27355 | 0.293726 | 0 | 0 | 131 | 0.124965 |
| coupa_seed29 | transformer | full_risk | 14616 | 0.015785152 | 25954 | 0.278683 | 0 | 0 | 130 | 0.109008 |
| coupa_seed29 | eqmotion | history | 14612 | 0.0038305937 | 7905 | 0.084880 | 0 | 0 | 313 | 0.288636 |
| coupa_seed29 | eqmotion | history | 14613 | 0.0028394235 | 7030 | 0.075485 | 0 | 0 | 321 | 0.202264 |
| coupa_seed29 | eqmotion | history | 14614 | 0.0045491541 | 8688 | 0.093288 | 0 | 0 | 281 | 0.080876 |
| coupa_seed29 | eqmotion | history | 14615 | 2.5513475e-12 | 11090 | 0.119080 | 0 | 0 | 266 | -0.003327 |
| coupa_seed29 | eqmotion | history | 14616 | 2.5513475e-12 | 12706 | 0.136431 | 0 | 0 | 244 | -0.075331 |
| coupa_seed29 | eqmotion | full_risk | 14612 | 0.0069751126 | 34755 | 0.373184 | 0 | 0 | 135 | 0.326061 |
| coupa_seed29 | eqmotion | full_risk | 14613 | 0.0047295933 | 25265 | 0.271285 | 0 | 0 | 133 | 0.253922 |
| coupa_seed29 | eqmotion | full_risk | 14614 | 0.0043701262 | 24825 | 0.266560 | 0 | 0 | 133 | 0.252432 |
| coupa_seed29 | eqmotion | full_risk | 14615 | 0.005617017 | 21634 | 0.232296 | 0 | 0 | 125 | 0.254132 |
| coupa_seed29 | eqmotion | full_risk | 14616 | 0.0051437814 | 22232 | 0.238718 | 0 | 0 | 125 | 0.254228 |
| coupa_seed43 | damped_velocity_005 | history | 14612 | 0.0038305937 | 7912 | 0.084850 | 0 | 0 | 311 | -0.134300 |
| coupa_seed43 | damped_velocity_005 | history | 14613 | 0.0028394235 | 7030 | 0.075391 | 0 | 0 | 322 | -0.219363 |
| coupa_seed43 | damped_velocity_005 | history | 14614 | 0.0045491541 | 8700 | 0.093301 | 0 | 0 | 279 | -0.311989 |
| coupa_seed43 | damped_velocity_005 | history | 14615 | 2.5513475e-12 | 11114 | 0.119189 | 0 | 0 | 262 | -0.336621 |
| coupa_seed43 | damped_velocity_005 | history | 14616 | 2.5513475e-12 | 12716 | 0.136369 | 0 | 0 | 238 | -0.393134 |
| coupa_seed43 | damped_velocity_005 | full_risk | 14612 | 0.0061670557 | 42147 | 0.451993 | 0 | 0 | 127 | -0.144267 |
| coupa_seed43 | damped_velocity_005 | full_risk | 14613 | 0.0073632239 | 42182 | 0.452368 | 0 | 0 | 125 | -0.155177 |
| coupa_seed43 | damped_velocity_005 | full_risk | 14614 | 0.0064391195 | 42367 | 0.454352 | 0 | 0 | 126 | -0.158299 |
| coupa_seed43 | damped_velocity_005 | full_risk | 14615 | 0.0056334661 | 42580 | 0.456637 | 0 | 0 | 126 | -0.165540 |
| coupa_seed43 | damped_velocity_005 | full_risk | 14616 | 0.0047416152 | 42774 | 0.458717 | 0 | 0 | 126 | -0.174514 |
| coupa_seed43 | transformer | history | 14612 | 0.0038305937 | 7912 | 0.084850 | 0 | 0 | 311 | 0.266373 |
| coupa_seed43 | transformer | history | 14613 | 0.0028394235 | 7030 | 0.075391 | 0 | 0 | 322 | 0.167315 |
| coupa_seed43 | transformer | history | 14614 | 0.0045491541 | 8700 | 0.093301 | 0 | 0 | 279 | 0.015415 |
| coupa_seed43 | transformer | history | 14615 | 2.5513475e-12 | 11114 | 0.119189 | 0 | 0 | 262 | -0.016984 |
| coupa_seed43 | transformer | history | 14616 | 2.5513475e-12 | 12716 | 0.136369 | 0 | 0 | 238 | -0.093150 |
| coupa_seed43 | transformer | full_risk | 14612 | 0.033786492 | 40489 | 0.434212 | 0 | 0 | 133 | 0.174805 |
| coupa_seed43 | transformer | full_risk | 14613 | 0.031155678 | 40052 | 0.429526 | 0 | 0 | 134 | 0.168773 |
| coupa_seed43 | transformer | full_risk | 14614 | 0.0293374 | 38749 | 0.415552 | 0 | 0 | 135 | 0.163338 |
| coupa_seed43 | transformer | full_risk | 14615 | 0.027470843 | 37131 | 0.398200 | 0 | 0 | 134 | 0.157796 |
| coupa_seed43 | transformer | full_risk | 14616 | 0.029531319 | 35325 | 0.378833 | 0 | 0 | 134 | 0.155085 |
| coupa_seed43 | eqmotion | history | 14612 | 0.0038305937 | 7912 | 0.084850 | 0 | 0 | 311 | 0.294596 |
| coupa_seed43 | eqmotion | history | 14613 | 0.0028394235 | 7030 | 0.075391 | 0 | 0 | 322 | 0.215456 |
| coupa_seed43 | eqmotion | history | 14614 | 0.0045491541 | 8700 | 0.093301 | 0 | 0 | 279 | 0.068232 |
| coupa_seed43 | eqmotion | history | 14615 | 2.5513475e-12 | 11114 | 0.119189 | 0 | 0 | 262 | -0.028428 |
| coupa_seed43 | eqmotion | history | 14616 | 2.5513475e-12 | 12716 | 0.136369 | 0 | 0 | 238 | -0.102403 |
| coupa_seed43 | eqmotion | full_risk | 14612 | 0.0049847345 | 29191 | 0.313050 | 0 | 0 | 132 | 0.349175 |
| coupa_seed43 | eqmotion | full_risk | 14613 | 0.0043840471 | 22260 | 0.238721 | 0 | 0 | 131 | 0.273599 |
| coupa_seed43 | eqmotion | full_risk | 14614 | 0.0044663004 | 22059 | 0.236565 | 0 | 0 | 129 | 0.269490 |
| coupa_seed43 | eqmotion | full_risk | 14615 | 0.0066355213 | 21025 | 0.225476 | 0 | 0 | 125 | 0.257229 |
| coupa_seed43 | eqmotion | full_risk | 14616 | 0.0059209932 | 21460 | 0.230141 | 0 | 0 | 125 | 0.256666 |
| hyang_seed17 | damped_velocity_005 | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | -0.068338 |
| hyang_seed17 | damped_velocity_005 | history | 129359 | 0.013805218 | 4517 | 0.097280 | 0 | 0 | 275 | 0.415490 |
| hyang_seed17 | damped_velocity_005 | full_risk | 125641 | 0.010346572 | 40355 | 0.869102 | 0 | 0 | 101 | -0.129119 |
| hyang_seed17 | damped_velocity_005 | full_risk | 129359 | 0.065136272 | 40298 | 0.867874 | 0 | 0 | 277 | -0.073998 |
| hyang_seed17 | transformer | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | 0.334440 |
| hyang_seed17 | transformer | history | 129359 | 0.013805218 | 4517 | 0.097280 | 0 | 0 | 275 | 0.566808 |
| hyang_seed17 | transformer | full_risk | 125641 | 0.017042273 | 40708 | 0.876704 | 0 | 0 | 172 | 0.046356 |
| hyang_seed17 | transformer | full_risk | 129359 | 0.076525182 | 40630 | 0.875024 | 0 | 0 | 274 | 0.139206 |
| hyang_seed17 | eqmotion | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | 0.350364 |
| hyang_seed17 | eqmotion | history | 129359 | 0.013805218 | 4517 | 0.097280 | 0 | 0 | 275 | 0.490081 |
| hyang_seed17 | eqmotion | full_risk | 125641 | 0.01044354 | 40398 | 0.870028 | 0 | 0 | 202 | 0.068145 |
| hyang_seed17 | eqmotion | full_risk | 129359 | 0.079151186 | 40240 | 0.866625 | 0 | 0 | 277 | 0.246778 |
| hyang_seed29 | damped_velocity_005 | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | -0.068338 |
| hyang_seed29 | damped_velocity_005 | history | 129359 | 0.013805218 | 4517 | 0.097280 | 0 | 0 | 275 | 0.415490 |
| hyang_seed29 | damped_velocity_005 | full_risk | 125641 | 0.010346572 | 40354 | 0.869080 | 0 | 0 | 101 | -0.129119 |
| hyang_seed29 | damped_velocity_005 | full_risk | 129359 | 0.065136272 | 40297 | 0.867853 | 0 | 0 | 277 | -0.073998 |
| hyang_seed29 | transformer | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | 0.331644 |
| hyang_seed29 | transformer | history | 129359 | 0.013805218 | 4517 | 0.097280 | 0 | 0 | 275 | 0.567472 |
| hyang_seed29 | transformer | full_risk | 125641 | 0.01715375 | 40689 | 0.876295 | 0 | 0 | 166 | -0.014167 |
| hyang_seed29 | transformer | full_risk | 129359 | 0.085415265 | 40657 | 0.875606 | 0 | 0 | 272 | 0.142059 |
| hyang_seed29 | eqmotion | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | 0.341855 |
| hyang_seed29 | eqmotion | history | 129359 | 0.013805218 | 4517 | 0.097280 | 0 | 0 | 275 | 0.488214 |
| hyang_seed29 | eqmotion | full_risk | 125641 | 0.0089510126 | 40300 | 0.867917 | 0 | 0 | 225 | 0.143421 |
| hyang_seed29 | eqmotion | full_risk | 129359 | 0.081525969 | 40220 | 0.866194 | 0 | 0 | 277 | 0.228041 |
| hyang_seed43 | damped_velocity_005 | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | -0.068338 |
| hyang_seed43 | damped_velocity_005 | history | 129359 | 0.013805218 | 4515 | 0.097237 | 0 | 0 | 275 | 0.418308 |
| hyang_seed43 | damped_velocity_005 | full_risk | 125641 | 0.010346572 | 40354 | 0.869080 | 0 | 0 | 101 | -0.129119 |
| hyang_seed43 | damped_velocity_005 | full_risk | 129359 | 0.065136272 | 40297 | 0.867853 | 0 | 0 | 277 | -0.073998 |
| hyang_seed43 | transformer | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | 0.289666 |
| hyang_seed43 | transformer | history | 129359 | 0.013805218 | 4515 | 0.097237 | 0 | 0 | 275 | 0.553032 |
| hyang_seed43 | transformer | full_risk | 125641 | 0.01798367 | 40713 | 0.876812 | 0 | 0 | 176 | 0.095061 |
| hyang_seed43 | transformer | full_risk | 129359 | 0.058577415 | 40501 | 0.872246 | 0 | 0 | 272 | 0.115464 |
| hyang_seed43 | eqmotion | history | 125641 | 0.011774076 | 1330 | 0.028643 | 0 | 0 | 263 | 0.349050 |
| hyang_seed43 | eqmotion | history | 129359 | 0.013805218 | 4515 | 0.097237 | 0 | 0 | 275 | 0.486274 |
| hyang_seed43 | eqmotion | full_risk | 125641 | 0.010460267 | 40233 | 0.866474 | 0 | 0 | 210 | 0.117607 |
| hyang_seed43 | eqmotion | full_risk | 129359 | 0.080576379 | 40211 | 0.866000 | 0 | 0 | 277 | 0.248702 |

## Outcome-Blind Control Comparison

32 fixed metadata-hash controls per site, reused across seeds/actions. The 384 records per action/arm are not 384 independent sites or new tracks.
At-least-as-distant fractions compare each case only with its 32 same-view controls; they are descriptive, not calibrated p-values or thresholds.

| Action | Features | Case fraction of controls at least as distant, min / median / max |
|---|---|---:|
| damped_velocity_005 | history | 0.7188 / 0.9062 / 0.9375 |
| damped_velocity_005 | full_risk | 0.0938 / 0.7812 / 0.9062 |
| transformer | history | 0.7188 / 0.9062 / 0.9375 |
| transformer | full_risk | 0.0938 / 0.5625 / 0.8750 |
| eqmotion | history | 0.7188 / 0.9062 / 0.9375 |
| eqmotion | full_risk | 0.0938 / 0.9062 / 0.9375 |

All control quantiles, neighborhood counts at 32/128/512, source counts and identities remain in analysis.json.
Near contexts are not identical observations; these results do not prove Bayes irreducibility.
No deployment, external readout, independent calibration or Stage5C/SMC execution.
