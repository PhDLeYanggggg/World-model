# Reference Protection Results

## Material Passport
Fresh 72 Torch continuations / 144,000 additional updates and 360 new readout views; 216 old views cached_verified.
All six source A/B assignments, three seeds, full/motion-only forecast pairs. C is historically opened development.
No favorable seed, checkpoint, threshold or source assignment is selected.

## Policies

| Pair / policy | All ADE vs raw (%) | Hard vs raw (%) | All vs R (%) | Worst net easy degradation (%) | Risk /18 | Net easy /18 | Switch fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| full / reference | -4.62412 to -2.90493 | -5.27571 to -2.80735 | +0.00000 to +0.00000 | +0.00000 to +0.00000 | 18 | 18 | +0.00000 to +0.00000 |
| full / raw_neural | +0.00000 to +0.00000 | +0.00000 to +0.00000 | +2.81490 to +4.37497 | +0.00000 to +0.00000 | 5 | 18 | +0.31521 to +0.43763 |
| full / raw_ridge | -3.09329 to +0.13989 | -3.23178 to +0.45188 | +0.45784 to +4.41538 | +0.00000 to +0.00000 | 12 | 18 | +0.03570 to +0.35368 |
| full / mean_dual | -4.03905 to -0.97992 | -4.73168 to -1.04035 | +0.09256 to +2.66995 | +0.00000 to +0.00000 | 16 | 18 | +0.07290 to +0.29693 |
| full / mean_joint | -0.75681 to +0.85588 | -0.54216 to +1.02611 | +2.60860 to +4.41676 | +0.00000 to +0.00000 | 6 | 18 | +0.26871 to +0.42555 |
| full / sampling_joint | -0.70490 to +0.89179 | -0.69284 to +1.21649 | +2.20926 to +4.51687 | +0.00000 to +0.00000 | 4 | 18 | +0.30136 to +0.43595 |
| full / continued_all | -1.16785 to -0.02388 | -1.66830 to -0.03347 | +2.29483 to +4.33828 | +0.00000 to +0.00000 | 12 | 18 | +0.20633 to +0.39884 |
| full / continued_dual | -3.85345 to -0.55673 | -4.73132 to -0.61373 | +0.06836 to +3.10093 | +0.00000 to +0.00000 | 15 | 18 | +0.04772 to +0.32465 |
| full / continued_scene | -3.25051 to -0.10645 | -3.63578 to -0.08415 | +0.64439 to +3.96145 | +0.00000 to +0.00000 | 14 | 18 | +0.05910 to +0.39443 |
| full / continued_joint | -0.75472 to +1.04790 | -0.32492 to +1.25688 | +2.58232 to +4.60120 | +0.00000 to +0.00000 | 4 | 18 | +0.27708 to +0.42623 |
| full / continued_hash_matched | -1.61772 to +0.92664 | -1.60665 to +1.10966 | +2.20036 to +4.48414 | +0.00000 to +0.00000 | 0 | 18 | +0.27708 to +0.42623 |
| full / protected_all | -1.06797 to -0.04530 | -1.40138 to -0.06713 | +2.29310 to +4.18421 | +0.00000 to +0.00000 | 9 | 18 | +0.21133 to +0.39682 |
| full / protected_dual | -3.52154 to -0.44869 | -4.41948 to -0.52605 | +0.38281 to +3.53216 | +0.00000 to +0.00000 | 14 | 18 | +0.07701 to +0.35255 |
| full / protected_scene | -2.78683 to -0.16726 | -3.05732 to -0.10826 | +1.08474 to +4.08927 | +0.00000 to +0.00000 | 11 | 18 | +0.09460 to +0.39462 |
| full / protected_joint | -0.28581 to +1.02848 | -0.26406 to +1.24484 | +2.62293 to +4.58260 | +0.00000 to +0.00000 | 5 | 18 | +0.30151 to +0.42766 |
| full / protected_hash_matched | -0.86154 to +0.89442 | -0.78494 to +1.07523 | +2.40015 to +4.45329 | +0.00000 to +0.00000 | 0 | 18 | +0.30151 to +0.42766 |
| motion_only / reference | -4.06577 to -2.87137 | -5.31206 to -2.73188 | +0.00000 to +0.00000 | +0.00000 to +0.00000 | 18 | 18 | +0.00000 to +0.00000 |
| motion_only / raw_neural | +0.00000 to +0.00000 | +0.00000 to +0.00000 | +2.78156 to +3.86793 | +0.00000 to +1.86021 | 6 | 18 | +0.19615 to +0.35264 |
| motion_only / raw_ridge | -1.96597 to +0.06876 | -2.17820 to +0.21859 | +1.37961 to +3.89013 | +0.00000 to +1.85734 | 12 | 18 | +0.08253 to +0.31753 |
| motion_only / mean_dual | -2.64001 to -0.61500 | -3.48937 to -0.77730 | +0.28638 to +2.62441 | +0.00000 to +0.83516 | 17 | 18 | +0.06328 to +0.28270 |
| motion_only / mean_joint | -0.58800 to +0.26173 | -0.67157 to +0.37903 | +2.27160 to +3.77560 | +0.00000 to +0.83570 | 12 | 18 | +0.19817 to +0.32952 |
| motion_only / sampling_joint | -0.64020 to +0.43518 | -0.57576 to +0.61826 | +2.22094 to +3.88323 | +0.00000 to +1.83096 | 8 | 18 | +0.20571 to +0.33344 |
| motion_only / continued_all | -0.84065 to -0.01472 | -0.98108 to +0.07756 | +2.34652 to +3.82434 | +0.00000 to +1.46715 | 12 | 18 | +0.15756 to +0.33946 |
| motion_only / continued_dual | -2.73391 to -0.38125 | -2.86020 to -0.40275 | +0.22684 to +3.28265 | +0.00000 to +1.38045 | 13 | 18 | +0.10938 to +0.30755 |
| motion_only / continued_scene | -0.88015 to +0.02418 | -1.31596 to +0.05595 | +2.09461 to +3.72491 | +0.00000 to +1.50503 | 12 | 18 | +0.18740 to +0.33316 |
| motion_only / continued_joint | -0.61296 to +0.37069 | -0.67961 to +0.59112 | +2.26313 to +3.89930 | +0.00000 to +2.01190 | 9 | 17 | +0.21172 to +0.34367 |
| motion_only / continued_hash_matched | -0.63069 to +0.36607 | -0.82107 to +0.53677 | +2.25969 to +3.90487 | +0.00000 to +1.39025 | 8 | 18 | +0.21172 to +0.34367 |
| motion_only / protected_all | -0.83367 to -0.04795 | -0.96197 to +0.04636 | +2.30553 to +3.81164 | +0.00000 to +1.46715 | 12 | 18 | +0.15159 to +0.33840 |
| motion_only / protected_dual | -2.43493 to -0.30523 | -2.59234 to -0.29753 | +0.51653 to +3.47151 | +0.00000 to +1.46715 | 12 | 18 | +0.12373 to +0.30817 |
| motion_only / protected_scene | -0.73091 to +0.02584 | -1.16370 to +0.07969 | +2.20831 to +3.82929 | +0.00000 to +1.86021 | 12 | 18 | +0.18854 to +0.33225 |
| motion_only / protected_joint | -0.57975 to +0.37467 | -0.65429 to +0.53292 | +2.31275 to +3.93618 | +0.00000 to +1.86021 | 6 | 18 | +0.21273 to +0.34412 |
| motion_only / protected_hash_matched | -0.58054 to +0.37018 | -0.64072 to +0.54566 | +2.31178 to +3.93294 | +0.00000 to +1.86021 | 7 | 18 | +0.21273 to +0.34412 |

Complete risk requires all/easy positive harm <=2%, net easy <=2%, and no zero-CV harm in every supported C locality.
Reference-only abstention is not learned improvement. A frozen denominator is not a calibrated denominator.

## Three-Seed Contrasts

| Pair / A-B roles | All-ADE contrast | Point (%) | 95% locality-bootstrap CI (%) |
|---|---|---:|---:|
| full / producer0_controller1 | continued_joint_vs_mean_joint__all | +0.20911 | +0.06848 to +0.47591 |
| full / producer0_controller1 | protected_all_vs_continued_all__all | -0.06152 | -0.16411 to +0.00429 |
| full / producer0_controller1 | protected_dual_vs_continued_dual__all | +0.61925 | +0.27727 to +0.95925 |
| full / producer0_controller1 | protected_joint_vs_continued_joint__all | +0.01020 | -0.08443 to +0.09539 |
| full / producer0_controller1 | protected_joint_vs_mean_joint__all | +0.21913 | -0.01611 to +0.57094 |
| full / producer0_controller1 | protected_joint_vs_protected_dual__all | +0.96252 | +0.25194 to +1.52317 |
| full / producer0_controller1 | protected_joint_vs_protected_hash_matched__all | +0.03910 | -0.04536 to +0.14247 |
| full / producer0_controller1 | protected_joint_vs_raw_neural__all | +0.04184 | -0.21971 to +0.31637 |
| full / producer0_controller1 | protected_joint_vs_raw_ridge__all | -0.04061 | -0.22848 to +0.15824 |
| full / producer0_controller1 | protected_joint_vs_sampling_joint__all | +0.17312 | +0.02059 to +0.44769 |
| full / producer0_controller1 | protected_scene_vs_continued_scene__all | +0.02146 | -0.03533 to +0.10380 |
| full / producer0_controller2 | continued_joint_vs_mean_joint__all | -0.00822 | -0.05465 to +0.03401 |
| full / producer0_controller2 | protected_all_vs_continued_all__all | -0.00845 | -0.04558 to +0.02868 |
| full / producer0_controller2 | protected_dual_vs_continued_dual__all | +0.21912 | +0.13201 to +0.30623 |
| full / producer0_controller2 | protected_joint_vs_continued_joint__all | +0.00924 | -0.04674 to +0.06764 |
| full / producer0_controller2 | protected_joint_vs_mean_joint__all | +0.00110 | -0.03335 to +0.03524 |
| full / producer0_controller2 | protected_joint_vs_protected_dual__all | +1.98247 | +1.56495 to +2.39999 |
| full / producer0_controller2 | protected_joint_vs_protected_hash_matched__all | +0.10246 | +0.04408 to +0.15264 |
| full / producer0_controller2 | protected_joint_vs_raw_neural__all | +0.32016 | -0.20805 to +0.71153 |
| full / producer0_controller2 | protected_joint_vs_raw_ridge__all | +2.46015 | +2.25821 to +2.79951 |
| full / producer0_controller2 | protected_joint_vs_sampling_joint__all | -0.01123 | -0.08005 to +0.05762 |
| full / producer0_controller2 | protected_scene_vs_continued_scene__all | +0.10967 | +0.07070 to +0.15561 |
| full / producer1_controller0 | continued_joint_vs_mean_joint__all | +0.52283 | +0.33315 to +0.89632 |
| full / producer1_controller0 | protected_all_vs_continued_all__all | +0.07169 | -0.07756 to +0.19161 |
| full / producer1_controller0 | protected_dual_vs_continued_dual__all | +0.44709 | +0.07295 to +0.78525 |
| full / producer1_controller0 | protected_joint_vs_continued_joint__all | +0.22672 | -0.05147 to +0.54229 |
| full / producer1_controller0 | protected_joint_vs_mean_joint__all | +0.74900 | +0.28631 to +1.43686 |
| full / producer1_controller0 | protected_joint_vs_protected_dual__all | +2.73727 | +1.01887 to +3.89190 |
| full / producer1_controller0 | protected_joint_vs_protected_hash_matched__all | +0.66093 | +0.21003 to +1.05865 |
| full / producer1_controller0 | protected_joint_vs_raw_neural__all | +0.12849 | -0.08507 to +0.37851 |
| full / producer1_controller0 | protected_joint_vs_raw_ridge__all | +0.83317 | +0.12550 to +1.47269 |
| full / producer1_controller0 | protected_joint_vs_sampling_joint__all | +0.26071 | +0.00569 to +0.75672 |
| full / producer1_controller0 | protected_scene_vs_continued_scene__all | +0.24588 | +0.05960 to +0.46815 |
| full / producer1_controller2 | continued_joint_vs_mean_joint__all | +0.07300 | -0.07002 to +0.19690 |
| full / producer1_controller2 | protected_all_vs_continued_all__all | -0.05206 | -0.08130 to -0.02283 |
| full / producer1_controller2 | protected_dual_vs_continued_dual__all | +0.21959 | +0.16905 to +0.29110 |
| full / producer1_controller2 | protected_joint_vs_continued_joint__all | -0.05427 | -0.12733 to +0.01880 |
| full / producer1_controller2 | protected_joint_vs_mean_joint__all | +0.01879 | -0.17815 to +0.21573 |
| full / producer1_controller2 | protected_joint_vs_protected_dual__all | +2.26011 | +1.32171 to +2.97181 |
| full / producer1_controller2 | protected_joint_vs_protected_hash_matched__all | +0.24509 | +0.05803 to +0.44450 |
| full / producer1_controller2 | protected_joint_vs_raw_neural__all | +0.62301 | +0.00071 to +1.18183 |
| full / producer1_controller2 | protected_joint_vs_raw_ridge__all | +3.39819 | +2.54552 to +4.33017 |
| full / producer1_controller2 | protected_joint_vs_sampling_joint__all | -0.04844 | -0.20208 to +0.10520 |
| full / producer1_controller2 | protected_scene_vs_continued_scene__all | -0.05566 | -0.11615 to +0.00777 |
| full / producer2_controller0 | continued_joint_vs_mean_joint__all | +0.08204 | -0.05705 to +0.27918 |
| full / producer2_controller0 | protected_all_vs_continued_all__all | +0.03147 | -0.06467 to +0.11337 |
| full / producer2_controller0 | protected_dual_vs_continued_dual__all | +0.11451 | +0.00470 to +0.23457 |
| full / producer2_controller0 | protected_joint_vs_continued_joint__all | +0.00359 | -0.09543 to +0.09793 |
| full / producer2_controller0 | protected_joint_vs_mean_joint__all | +0.08613 | +0.02600 to +0.19014 |
| full / producer2_controller0 | protected_joint_vs_protected_dual__all | +0.53176 | +0.28801 to +0.72699 |
| full / producer2_controller0 | protected_joint_vs_protected_hash_matched__all | +0.21462 | +0.10168 to +0.33207 |
| full / producer2_controller0 | protected_joint_vs_raw_neural__all | -0.17132 | -0.41134 to -0.04076 |
| full / producer2_controller0 | protected_joint_vs_raw_ridge__all | -0.07681 | -0.35320 to +0.17020 |
| full / producer2_controller0 | protected_joint_vs_sampling_joint__all | +0.25326 | +0.08787 to +0.49950 |
| full / producer2_controller0 | protected_scene_vs_continued_scene__all | +0.03738 | -0.04236 to +0.11859 |
| full / producer2_controller1 | continued_joint_vs_mean_joint__all | +0.10411 | +0.04351 to +0.16470 |
| full / producer2_controller1 | protected_all_vs_continued_all__all | +0.02259 | -0.02882 to +0.06156 |
| full / producer2_controller1 | protected_dual_vs_continued_dual__all | +0.19377 | +0.16350 to +0.24886 |
| full / producer2_controller1 | protected_joint_vs_continued_joint__all | +0.01270 | -0.02949 to +0.05362 |
| full / producer2_controller1 | protected_joint_vs_mean_joint__all | +0.11676 | +0.03366 to +0.20334 |
| full / producer2_controller1 | protected_joint_vs_protected_dual__all | +0.78388 | +0.54931 to +1.10442 |
| full / producer2_controller1 | protected_joint_vs_protected_hash_matched__all | +0.18113 | +0.05605 to +0.31338 |
| full / producer2_controller1 | protected_joint_vs_raw_neural__all | -0.06158 | -0.08536 to -0.03780 |
| full / producer2_controller1 | protected_joint_vs_raw_ridge__all | -0.09108 | -0.15189 to -0.03027 |
| full / producer2_controller1 | protected_joint_vs_sampling_joint__all | +0.17962 | +0.05595 to +0.30328 |
| full / producer2_controller1 | protected_scene_vs_continued_scene__all | +0.06355 | +0.01554 to +0.11114 |
| motion_only / producer0_controller1 | continued_joint_vs_mean_joint__all | +0.08978 | -0.07289 to +0.27459 |
| motion_only / producer0_controller1 | protected_all_vs_continued_all__all | -0.01161 | -0.02583 to +0.00261 |
| motion_only / producer0_controller1 | protected_dual_vs_continued_dual__all | +0.21274 | +0.11108 to +0.30822 |
| motion_only / producer0_controller1 | protected_joint_vs_continued_joint__all | +0.04118 | +0.00339 to +0.07897 |
| motion_only / producer0_controller1 | protected_joint_vs_mean_joint__all | +0.13095 | +0.00457 to +0.31726 |
| motion_only / producer0_controller1 | protected_joint_vs_protected_dual__all | +0.60365 | +0.20533 to +0.89398 |
| motion_only / producer0_controller1 | protected_joint_vs_protected_hash_matched__all | +0.04230 | -0.00396 to +0.10383 |
| motion_only / producer0_controller1 | protected_joint_vs_raw_neural__all | +0.03174 | -0.08940 to +0.17421 |
| motion_only / producer0_controller1 | protected_joint_vs_raw_ridge__all | -0.00318 | -0.08806 to +0.06815 |
| motion_only / producer0_controller1 | protected_joint_vs_sampling_joint__all | +0.03668 | -0.00718 to +0.06062 |
| motion_only / producer0_controller1 | protected_scene_vs_continued_scene__all | +0.07044 | +0.02245 to +0.11981 |
| motion_only / producer0_controller2 | continued_joint_vs_mean_joint__all | +0.24883 | +0.09869 to +0.39897 |
| motion_only / producer0_controller2 | protected_all_vs_continued_all__all | -0.05265 | -0.08251 to -0.02331 |
| motion_only / producer0_controller2 | protected_dual_vs_continued_dual__all | +0.14218 | +0.02182 to +0.32518 |
| motion_only / producer0_controller2 | protected_joint_vs_continued_joint__all | +0.02164 | +0.01616 to +0.02773 |
| motion_only / producer0_controller2 | protected_joint_vs_mean_joint__all | +0.27044 | +0.11556 to +0.42532 |
| motion_only / producer0_controller2 | protected_joint_vs_protected_dual__all | +1.16295 | +0.81060 to +1.54413 |
| motion_only / producer0_controller2 | protected_joint_vs_protected_hash_matched__all | +0.04824 | +0.00270 to +0.10238 |
| motion_only / producer0_controller2 | protected_joint_vs_raw_neural__all | +0.09400 | -0.31669 to +0.38118 |
| motion_only / producer0_controller2 | protected_joint_vs_raw_ridge__all | +1.70686 | +1.10146 to +2.38666 |
| motion_only / producer0_controller2 | protected_joint_vs_sampling_joint__all | +0.05559 | +0.00771 to +0.10346 |
| motion_only / producer0_controller2 | protected_scene_vs_continued_scene__all | +0.01407 | -0.00351 to +0.03924 |
| motion_only / producer1_controller0 | continued_joint_vs_mean_joint__all | +0.06305 | +0.00173 to +0.10152 |
| motion_only / producer1_controller0 | protected_all_vs_continued_all__all | +0.03656 | +0.01298 to +0.06485 |
| motion_only / producer1_controller0 | protected_dual_vs_continued_dual__all | +0.35086 | +0.03426 to +0.64141 |
| motion_only / producer1_controller0 | protected_joint_vs_continued_joint__all | +0.00612 | -0.01342 to +0.02566 |
| motion_only / producer1_controller0 | protected_joint_vs_mean_joint__all | +0.06917 | -0.00338 to +0.11154 |
| motion_only / producer1_controller0 | protected_joint_vs_protected_dual__all | +0.55510 | +0.20182 to +0.90838 |
| motion_only / producer1_controller0 | protected_joint_vs_protected_hash_matched__all | +0.08447 | +0.00374 to +0.15458 |
| motion_only / producer1_controller0 | protected_joint_vs_raw_neural__all | -0.20143 | -0.39855 to -0.00431 |
| motion_only / producer1_controller0 | protected_joint_vs_raw_ridge__all | +0.05708 | -0.07316 to +0.16945 |
| motion_only / producer1_controller0 | protected_joint_vs_sampling_joint__all | -0.12794 | -0.24104 to -0.03421 |
| motion_only / producer1_controller0 | protected_scene_vs_continued_scene__all | +0.11279 | +0.03029 to +0.19530 |
| motion_only / producer1_controller2 | continued_joint_vs_mean_joint__all | +0.03055 | -0.03000 to +0.12129 |
| motion_only / producer1_controller2 | protected_all_vs_continued_all__all | -0.05788 | -0.09676 to -0.01901 |
| motion_only / producer1_controller2 | protected_dual_vs_continued_dual__all | +0.17960 | +0.12912 to +0.23007 |
| motion_only / producer1_controller2 | protected_joint_vs_continued_joint__all | +0.00505 | -0.02340 to +0.03332 |
| motion_only / producer1_controller2 | protected_joint_vs_mean_joint__all | +0.03564 | -0.02726 to +0.12571 |
| motion_only / producer1_controller2 | protected_joint_vs_protected_dual__all | +1.35841 | +0.79620 to +1.79312 |
| motion_only / producer1_controller2 | protected_joint_vs_protected_hash_matched__all | +0.01475 | -0.04440 to +0.07389 |
| motion_only / producer1_controller2 | protected_joint_vs_raw_neural__all | +0.24764 | -0.05483 to +0.42585 |
| motion_only / producer1_controller2 | protected_joint_vs_raw_ridge__all | +2.00212 | +1.51482 to +2.69532 |
| motion_only / producer1_controller2 | protected_joint_vs_sampling_joint__all | +0.10919 | +0.00924 to +0.19392 |
| motion_only / producer1_controller2 | protected_scene_vs_continued_scene__all | +0.05741 | +0.02321 to +0.09168 |
| motion_only / producer2_controller0 | continued_joint_vs_mean_joint__all | -0.00914 | -0.03893 to +0.01644 |
| motion_only / producer2_controller0 | protected_all_vs_continued_all__all | +0.03319 | +0.01896 to +0.04540 |
| motion_only / producer2_controller0 | protected_dual_vs_continued_dual__all | +0.40301 | +0.26068 to +0.54117 |
| motion_only / producer2_controller0 | protected_joint_vs_continued_joint__all | +0.08569 | +0.02468 to +0.17493 |
| motion_only / producer2_controller0 | protected_joint_vs_mean_joint__all | +0.07653 | -0.01258 to +0.19393 |
| motion_only / producer2_controller0 | protected_joint_vs_protected_dual__all | +1.51919 | +0.97639 to +2.06199 |
| motion_only / producer2_controller0 | protected_joint_vs_protected_hash_matched__all | -0.00521 | -0.04412 to +0.04033 |
| motion_only / producer2_controller0 | protected_joint_vs_raw_neural__all | -0.45760 | -0.99081 to -0.10300 |
| motion_only / producer2_controller0 | protected_joint_vs_raw_ridge__all | -0.43305 | -0.93911 to -0.07555 |
| motion_only / producer2_controller0 | protected_joint_vs_sampling_joint__all | +0.06689 | +0.02253 to +0.12655 |
| motion_only / producer2_controller0 | protected_scene_vs_continued_scene__all | +0.12441 | +0.06553 to +0.18297 |
| motion_only / producer2_controller1 | continued_joint_vs_mean_joint__all | +0.11142 | +0.05661 to +0.16623 |
| motion_only / producer2_controller1 | protected_all_vs_continued_all__all | -0.00484 | -0.05882 to +0.07108 |
| motion_only / producer2_controller1 | protected_dual_vs_continued_dual__all | +0.12956 | -0.00293 to +0.24923 |
| motion_only / producer2_controller1 | protected_joint_vs_continued_joint__all | +0.01490 | -0.01909 to +0.04889 |
| motion_only / producer2_controller1 | protected_joint_vs_mean_joint__all | +0.12627 | +0.03752 to +0.21501 |
| motion_only / producer2_controller1 | protected_joint_vs_protected_dual__all | +0.47093 | +0.31758 to +0.65520 |
| motion_only / producer2_controller1 | protected_joint_vs_protected_hash_matched__all | +0.02518 | +0.00529 to +0.04161 |
| motion_only / producer2_controller1 | protected_joint_vs_raw_neural__all | -0.11943 | -0.20706 to -0.04133 |
| motion_only / producer2_controller1 | protected_joint_vs_raw_ridge__all | -0.07136 | -0.13037 to -0.01031 |
| motion_only / producer2_controller1 | protected_joint_vs_sampling_joint__all | -0.01310 | -0.02503 to -0.00118 |
| motion_only / producer2_controller1 | protected_scene_vs_continued_scene__all | -0.05050 | -0.10060 to -0.00617 |

Three seeds are averaged within locality, then four C localities are resampled 3,000 times.
Overlapping source roles and unadjusted multiple contrasts are not independent confirmatory evidence.
The count-matched hash control does not match realized risk. Allocation is greedy, not optimal or collision-aware.
Protected inference carries a second frozen network; equal update count is not equal total storage or inference cost.
Obs8/pred12 annotation steps at raw stride12; image pixels, detector-derived labels.
No metric/seconds, human-gold, physical-safety, true3D, foundation or submission-ready claim.
No selection/calibration/confirmation access. No deployment. Stage5C and SMC remain off.
