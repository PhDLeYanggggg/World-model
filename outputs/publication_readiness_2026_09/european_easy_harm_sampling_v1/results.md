# Importance-Corrected Easy-Harm Sampling

Fresh 36 Torch fits / 72,000 updates and 180 new policy views; 324 old views cached_verified.
The 504 views cover three seeds and six ordered source assignments in each of two forecast pairs.
Source C is excluded from the current A/B fitted chain but historically opened development.

| Pair / policy | All ADE vs raw neural (%) | Hard vs raw (%) | All vs R (%) | Worst easy degradation (%) | Risk /18 | Net easy /18 | Switch fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| full / reference | -4.62412 to -2.90493 | -5.27571 to -2.80735 | +0.00000 to +0.00000 | +0.00000 to +0.00000 | 18 | 18 | +0.00000 to +0.00000 |
| full / raw_neural | +0.00000 to +0.00000 | +0.00000 to +0.00000 | +2.81490 to +4.37497 | +0.00000 to +0.00000 | 5 | 18 | +0.31521 to +0.43763 |
| full / raw_ridge | -3.09329 to +0.13989 | -3.23178 to +0.45188 | +0.45784 to +4.41538 | +0.00000 to +0.00000 | 12 | 18 | +0.03570 to +0.35368 |
| full / mean_all | -1.50734 to -0.10971 | -2.15342 to -0.09821 | +1.91855 to +3.88612 | +0.00000 to +0.00000 | 10 | 18 | +0.20721 to +0.37202 |
| full / mean_dual | -4.03905 to -0.97992 | -4.73168 to -1.04035 | +0.09256 to +2.66995 | +0.00000 to +0.00000 | 16 | 18 | +0.07290 to +0.29693 |
| full / mean_scene | -3.27787 to -0.28909 | -3.77762 to -0.21861 | +0.64530 to +3.94162 | +0.00000 to +0.00000 | 15 | 18 | +0.05302 to +0.40419 |
| full / mean_joint | -0.75681 to +0.85588 | -0.54216 to +1.02611 | +2.60860 to +4.41676 | +0.00000 to +0.00000 | 6 | 18 | +0.26871 to +0.42555 |
| full / selected_dual | -4.12548 to -1.25262 | -4.89265 to -1.35013 | +0.07737 to +2.38396 | +0.00000 to +0.00000 | 17 | 18 | +0.06575 to +0.26736 |
| full / selected_joint | -1.49111 to +0.78578 | -1.57796 to +0.87031 | +2.31920 to +4.34929 | +0.00000 to +0.00000 | 9 | 18 | +0.23063 to +0.41200 |
| full / corrected_all | -1.67884 to -0.10045 | -2.17326 to -0.10599 | +1.69501 to +3.80982 | +0.00000 to +0.00000 | 7 | 18 | +0.23752 to +0.37352 |
| full / corrected_dual | -4.12600 to -0.67153 | -4.73170 to -0.97929 | +0.08598 to +3.12920 | +0.00000 to +0.00000 | 14 | 18 | +0.06112 to +0.28966 |
| full / corrected_scene | -3.37272 to -0.29121 | -3.96515 to -0.07005 | +0.66582 to +3.90859 | +0.00000 to +0.00000 | 11 | 18 | +0.07809 to +0.39879 |
| full / corrected_joint | -0.70490 to +0.89179 | -0.69284 to +1.21649 | +2.20926 to +4.51687 | +0.00000 to +0.00000 | 4 | 18 | +0.30136 to +0.43595 |
| full / corrected_hash_matched | -1.44076 to +0.79562 | -1.79333 to +1.16109 | +2.16946 to +4.34023 | +0.00000 to +0.00000 | 1 | 18 | +0.30136 to +0.43595 |
| motion_only / reference | -4.06577 to -2.87137 | -5.31206 to -2.73188 | +0.00000 to +0.00000 | +0.00000 to +0.00000 | 18 | 18 | +0.00000 to +0.00000 |
| motion_only / raw_neural | +0.00000 to +0.00000 | +0.00000 to +0.00000 | +2.78156 to +3.86793 | +0.00000 to +1.86021 | 6 | 18 | +0.19615 to +0.35264 |
| motion_only / raw_ridge | -1.96597 to +0.06876 | -2.17820 to +0.21859 | +1.37961 to +3.89013 | +0.00000 to +1.85734 | 12 | 18 | +0.08253 to +0.31753 |
| motion_only / mean_all | -0.93755 to -0.05108 | -1.07369 to -0.03224 | +2.13888 to +3.70645 | +0.00000 to +1.46715 | 14 | 18 | +0.14464 to +0.33387 |
| motion_only / mean_dual | -2.64001 to -0.61500 | -3.48937 to -0.77730 | +0.28638 to +2.62441 | +0.00000 to +0.83516 | 17 | 18 | +0.06328 to +0.28270 |
| motion_only / mean_scene | -1.46761 to -0.01421 | -1.56294 to -0.00744 | +2.04530 to +3.38071 | +0.00000 to +0.81910 | 13 | 18 | +0.17149 to +0.31184 |
| motion_only / mean_joint | -0.58800 to +0.26173 | -0.67157 to +0.37903 | +2.27160 to +3.77560 | +0.00000 to +0.83570 | 12 | 18 | +0.19817 to +0.32952 |
| motion_only / selected_dual | -3.83604 to -1.22770 | -4.81712 to -1.45148 | +0.06068 to +2.03291 | +0.00000 to +0.00000 | 18 | 18 | +0.01381 to +0.19720 |
| motion_only / selected_joint | -1.17961 to +0.12204 | -1.58761 to +0.17214 | +2.11881 to +3.36083 | +0.00000 to +0.00000 | 15 | 18 | +0.19238 to +0.30687 |
| motion_only / corrected_all | -1.05439 to -0.11634 | -1.17154 to -0.10721 | +2.14077 to +3.62713 | +0.00000 to +1.46715 | 12 | 18 | +0.15857 to +0.32191 |
| motion_only / corrected_dual | -2.30317 to -0.30373 | -2.56475 to -0.40496 | +0.61086 to +3.52586 | +0.00000 to +1.50491 | 14 | 18 | +0.07647 to +0.30233 |
| motion_only / corrected_scene | -1.07471 to -0.04825 | -1.21546 to +0.03452 | +1.99303 to +3.70338 | +0.00000 to +1.83096 | 10 | 18 | +0.17898 to +0.32405 |
| motion_only / corrected_joint | -0.64020 to +0.43518 | -0.57576 to +0.61826 | +2.22094 to +3.88323 | +0.00000 to +1.83096 | 8 | 18 | +0.20571 to +0.33344 |
| motion_only / corrected_hash_matched | -0.63534 to +0.40026 | -0.58689 to +0.61941 | +2.22593 to +3.88170 | +0.00000 to +1.83096 | 7 | 18 | +0.20571 to +0.33344 |

Complete risk requires both positive-harm ratios <=2%, net easy degradation <=2%,
and no zero-CV harm, in every supported C locality. Net easy passing is not complete risk passing.
Reference-only abstention is not learned improvement. No deployment selection is made.

## Three-Seed Comparisons

| Pair / A-B roles | All-ADE contrast | Point (%) | 95% locality-bootstrap CI (%) |
|---|---|---:|---:|
| full / producer0_controller1 | corrected_all_vs_mean_all__all | -0.04341 | -0.16163 to +0.05392 |
| full / producer0_controller1 | corrected_dual_vs_mean_dual__all | +0.53759 | +0.21208 to +0.86309 |
| full / producer0_controller1 | corrected_joint_vs_corrected_dual__all | +2.00847 | +0.73655 to +3.08817 |
| full / producer0_controller1 | corrected_joint_vs_corrected_hash_matched__all | +0.09480 | +0.01097 to +0.17862 |
| full / producer0_controller1 | corrected_joint_vs_mean_joint__all | +0.04530 | -0.03076 to +0.11512 |
| full / producer0_controller1 | corrected_joint_vs_raw_neural__all | -0.13360 | -0.56324 to +0.25244 |
| full / producer0_controller1 | corrected_joint_vs_raw_ridge__all | -0.21625 | -0.62518 to +0.10879 |
| full / producer0_controller1 | corrected_scene_vs_mean_scene__all | +0.08002 | +0.02224 to +0.13781 |
| full / producer0_controller2 | corrected_all_vs_mean_all__all | -0.00470 | -0.09444 to +0.07456 |
| full / producer0_controller2 | corrected_dual_vs_mean_dual__all | -0.05224 | -0.12924 to +0.02475 |
| full / producer0_controller2 | corrected_joint_vs_corrected_dual__all | +2.27643 | +1.85985 to +2.69300 |
| full / producer0_controller2 | corrected_joint_vs_corrected_hash_matched__all | +0.14953 | +0.11232 to +0.20010 |
| full / producer0_controller2 | corrected_joint_vs_mean_joint__all | +0.01210 | -0.06530 to +0.08305 |
| full / producer0_controller2 | corrected_joint_vs_raw_neural__all | +0.33105 | -0.19366 to +0.78735 |
| full / producer0_controller2 | corrected_joint_vs_raw_ridge__all | +2.47044 | +2.28849 to +2.76610 |
| full / producer0_controller2 | corrected_scene_vs_mean_scene__all | -0.18411 | -0.44132 to -0.02848 |
| full / producer1_controller0 | corrected_all_vs_mean_all__all | +0.00142 | -0.24293 to +0.17703 |
| full / producer1_controller0 | corrected_dual_vs_mean_dual__all | +0.94315 | +0.42640 to +1.37014 |
| full / producer1_controller0 | corrected_joint_vs_corrected_dual__all | +2.62870 | +1.05971 to +3.78529 |
| full / producer1_controller0 | corrected_joint_vs_corrected_hash_matched__all | +0.62265 | +0.25846 to +1.01069 |
| full / producer1_controller0 | corrected_joint_vs_mean_joint__all | +0.48651 | +0.26573 to +0.67770 |
| full / producer1_controller0 | corrected_joint_vs_raw_neural__all | -0.13778 | -0.39509 to +0.08703 |
| full / producer1_controller0 | corrected_joint_vs_raw_ridge__all | +0.56893 | +0.00628 to +1.00413 |
| full / producer1_controller0 | corrected_scene_vs_mean_scene__all | +0.75704 | +0.29328 to +1.02933 |
| full / producer1_controller2 | corrected_all_vs_mean_all__all | +0.06607 | +0.00639 to +0.12576 |
| full / producer1_controller2 | corrected_dual_vs_mean_dual__all | +0.02885 | -0.02485 to +0.06968 |
| full / producer1_controller2 | corrected_joint_vs_corrected_dual__all | +2.62064 | +1.51292 to +3.37554 |
| full / producer1_controller2 | corrected_joint_vs_corrected_hash_matched__all | +0.16965 | +0.01222 to +0.34378 |
| full / producer1_controller2 | corrected_joint_vs_mean_joint__all | +0.06710 | +0.01215 to +0.12754 |
| full / producer1_controller2 | corrected_joint_vs_raw_neural__all | +0.67072 | +0.00487 to +1.12051 |
| full / producer1_controller2 | corrected_joint_vs_raw_ridge__all | +3.44487 | +2.50826 to +4.31217 |
| full / producer1_controller2 | corrected_scene_vs_mean_scene__all | -0.00864 | -0.12823 to +0.11094 |
| full / producer2_controller0 | corrected_all_vs_mean_all__all | -0.19953 | -0.44546 to -0.04692 |
| full / producer2_controller0 | corrected_dual_vs_mean_dual__all | +0.29323 | +0.18688 to +0.39958 |
| full / producer2_controller0 | corrected_joint_vs_corrected_dual__all | +0.84463 | +0.51224 to +1.17615 |
| full / producer2_controller0 | corrected_joint_vs_corrected_hash_matched__all | +0.19937 | +0.10863 to +0.31387 |
| full / producer2_controller0 | corrected_joint_vs_mean_joint__all | -0.16911 | -0.31421 to -0.04992 |
| full / producer2_controller0 | corrected_joint_vs_raw_neural__all | -0.42722 | -0.91932 to -0.12986 |
| full / producer2_controller0 | corrected_joint_vs_raw_ridge__all | -0.33243 | -0.83129 to +0.00461 |
| full / producer2_controller0 | corrected_scene_vs_mean_scene__all | -0.05897 | -0.17450 to +0.06549 |
| full / producer2_controller1 | corrected_all_vs_mean_all__all | -0.03804 | -0.15463 to +0.05689 |
| full / producer2_controller1 | corrected_dual_vs_mean_dual__all | +0.21577 | -0.04893 to +0.40966 |
| full / producer2_controller1 | corrected_joint_vs_corrected_dual__all | +1.20182 | +0.92200 to +1.65973 |
| full / producer2_controller1 | corrected_joint_vs_corrected_hash_matched__all | +0.20120 | +0.05185 to +0.36636 |
| full / producer2_controller1 | corrected_joint_vs_mean_joint__all | -0.06301 | -0.10372 to -0.02230 |
| full / producer2_controller1 | corrected_joint_vs_raw_neural__all | -0.24176 | -0.35653 to -0.12698 |
| full / producer2_controller1 | corrected_joint_vs_raw_ridge__all | -0.27128 | -0.37644 to -0.16612 |
| full / producer2_controller1 | corrected_scene_vs_mean_scene__all | -0.02105 | -0.12165 to +0.05501 |
| motion_only / producer0_controller1 | corrected_all_vs_mean_all__all | -0.07556 | -0.16070 to +0.00591 |
| motion_only / producer0_controller1 | corrected_dual_vs_mean_dual__all | +1.06612 | +0.59929 to +1.59645 |
| motion_only / producer0_controller1 | corrected_joint_vs_corrected_dual__all | +1.09509 | +0.43072 to +1.68631 |
| motion_only / producer0_controller1 | corrected_joint_vs_corrected_hash_matched__all | +0.05778 | +0.02216 to +0.09907 |
| motion_only / producer0_controller1 | corrected_joint_vs_mean_joint__all | +0.09431 | +0.00405 to +0.26103 |
| motion_only / producer0_controller1 | corrected_joint_vs_raw_neural__all | -0.00499 | -0.13264 to +0.12420 |
| motion_only / producer0_controller1 | corrected_joint_vs_raw_ridge__all | -0.03992 | -0.13166 to +0.03154 |
| motion_only / producer0_controller1 | corrected_scene_vs_mean_scene__all | +0.24751 | +0.08666 to +0.42894 |
| motion_only / producer0_controller2 | corrected_all_vs_mean_all__all | +0.13303 | +0.03898 to +0.25114 |
| motion_only / producer0_controller2 | corrected_dual_vs_mean_dual__all | +0.78618 | +0.53731 to +1.07169 |
| motion_only / producer0_controller2 | corrected_joint_vs_corrected_dual__all | +1.22155 | +0.77425 to +1.67113 |
| motion_only / producer0_controller2 | corrected_joint_vs_corrected_hash_matched__all | +0.03047 | -0.02263 to +0.08815 |
| motion_only / producer0_controller2 | corrected_joint_vs_mean_joint__all | +0.21491 | +0.04333 to +0.38641 |
| motion_only / producer0_controller2 | corrected_joint_vs_raw_neural__all | +0.03820 | -0.39247 to +0.36162 |
| motion_only / producer0_controller2 | corrected_joint_vs_raw_ridge__all | +1.65181 | +1.02912 to +2.36494 |
| motion_only / producer0_controller2 | corrected_scene_vs_mean_scene__all | +0.32659 | +0.07242 to +0.58904 |
| motion_only / producer1_controller0 | corrected_all_vs_mean_all__all | +0.05044 | -0.02455 to +0.11496 |
| motion_only / producer1_controller0 | corrected_dual_vs_mean_dual__all | +1.70202 | +0.79384 to +2.51483 |
| motion_only / producer1_controller0 | corrected_joint_vs_corrected_dual__all | +0.27847 | +0.14208 to +0.40257 |
| motion_only / producer1_controller0 | corrected_joint_vs_corrected_hash_matched__all | +0.09469 | +0.02865 to +0.16073 |
| motion_only / producer1_controller0 | corrected_joint_vs_mean_joint__all | +0.19680 | +0.13547 to +0.25813 |
| motion_only / producer1_controller0 | corrected_joint_vs_raw_neural__all | -0.07339 | -0.22274 to +0.09644 |
| motion_only / producer1_controller0 | corrected_joint_vs_raw_ridge__all | +0.18462 | +0.01581 to +0.35342 |
| motion_only / producer1_controller0 | corrected_scene_vs_mean_scene__all | +0.64974 | +0.35168 to +0.95301 |
| motion_only / producer1_controller2 | corrected_all_vs_mean_all__all | -0.13588 | -0.22294 to -0.03225 |
| motion_only / producer1_controller2 | corrected_dual_vs_mean_dual__all | +0.09194 | -0.02474 to +0.24184 |
| motion_only / producer1_controller2 | corrected_joint_vs_corrected_dual__all | +1.27513 | +0.79819 to +1.64481 |
| motion_only / producer1_controller2 | corrected_joint_vs_corrected_hash_matched__all | +0.05882 | +0.00969 to +0.10795 |
| motion_only / producer1_controller2 | corrected_joint_vs_mean_joint__all | -0.07431 | -0.18648 to +0.11558 |
| motion_only / producer1_controller2 | corrected_joint_vs_raw_neural__all | +0.13808 | -0.06933 to +0.26234 |
| motion_only / producer1_controller2 | corrected_joint_vs_raw_ridge__all | +1.89479 | +1.46156 to +2.49736 |
| motion_only / producer1_controller2 | corrected_scene_vs_mean_scene__all | -0.29395 | -0.50156 to +0.06337 |
| motion_only / producer2_controller0 | corrected_all_vs_mean_all__all | -0.08098 | -0.12990 to -0.03207 |
| motion_only / producer2_controller0 | corrected_dual_vs_mean_dual__all | +0.17016 | -0.03525 to +0.37556 |
| motion_only / producer2_controller0 | corrected_joint_vs_corrected_dual__all | +1.70479 | +1.13953 to +2.27006 |
| motion_only / producer2_controller0 | corrected_joint_vs_corrected_hash_matched__all | -0.00141 | -0.06031 to +0.05539 |
| motion_only / producer2_controller0 | corrected_joint_vs_mean_joint__all | +0.00957 | -0.06437 to +0.06931 |
| motion_only / producer2_controller0 | corrected_joint_vs_raw_neural__all | -0.52526 | -1.11938 to -0.12557 |
| motion_only / producer2_controller0 | corrected_joint_vs_raw_ridge__all | -0.50068 | -1.06762 to -0.09812 |
| motion_only / producer2_controller0 | corrected_scene_vs_mean_scene__all | +0.00869 | -0.05512 to +0.05231 |
| motion_only / producer2_controller1 | corrected_all_vs_mean_all__all | -0.03076 | -0.11939 to +0.05354 |
| motion_only / producer2_controller1 | corrected_dual_vs_mean_dual__all | +0.71811 | +0.43379 to +1.00707 |
| motion_only / producer2_controller1 | corrected_joint_vs_corrected_dual__all | +0.34439 | +0.26899 to +0.44034 |
| motion_only / producer2_controller1 | corrected_joint_vs_corrected_hash_matched__all | +0.00211 | -0.02473 to +0.02803 |
| motion_only / producer2_controller1 | corrected_joint_vs_mean_joint__all | +0.13927 | +0.04677 to +0.23178 |
| motion_only / producer2_controller1 | corrected_joint_vs_raw_neural__all | -0.10630 | -0.18138 to -0.03209 |
| motion_only / producer2_controller1 | corrected_joint_vs_raw_ridge__all | -0.05824 | -0.10874 to +0.00626 |
| motion_only / producer2_controller1 | corrected_scene_vs_mean_scene__all | +0.19072 | +0.09034 to +0.29109 |

Seeds are averaged within locality, then four C localities are resampled 3,000 times.
Overlapping source-role views are not independent replications. No multiplicity-adjusted discovery.
The exact expected-loss identity does not make Adam updates unbiased, nor guarantee lower variance.
Greedy query allocation is not collision-aware or optimal. Matched hash controls query counts, not risk.

8 observed / 12 predicted annotation steps, raw stride 12; image pixels, detector-derived labels.
No metric, seconds-level, human-gold, physical-safety, true3D, foundation or submission-ready claim.
Selection, reserved calibration and confirmation are not read this round. Stage5C and SMC remain off.
