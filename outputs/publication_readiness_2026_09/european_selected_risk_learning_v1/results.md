# Source-Only Selected-Risk Learning

## Material Passport
Fresh 72 Torch fits (144,000 updates), frozen source-C readout; cached_verified forecasts and old heads.
Four held C localities per setting. Three seeds, overlapping source-role views, not independent confirmation.

| Pair / policy | All ADE vs raw neural (%) | Hard vs raw (%) | All vs R (%) | Worst easy degradation (%) | Risk passes /18 | Easy passes /18 | Switch rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| full / reference | -4.62412 to -2.90493 | -5.27571 to -2.80735 | +0.00000 to +0.00000 | +0.00000 to +0.00000 | 18 | 18 | +0.00000 to +0.00000 |
| full / raw_neural | +0.00000 to +0.00000 | +0.00000 to +0.00000 | +2.81490 to +4.37497 | +0.00000 to +0.00000 | 5 | 18 | +0.31521 to +0.43763 |
| full / raw_ridge | -3.09329 to +0.13989 | -3.23178 to +0.45188 | +0.45784 to +4.41538 | +0.00000 to +0.00000 | 12 | 18 | +0.03570 to +0.35368 |
| full / mean_all | -1.50734 to -0.10971 | -2.15342 to -0.09821 | +1.91855 to +3.88612 | +0.00000 to +0.00000 | 10 | 18 | +0.20721 to +0.37202 |
| full / mean_dual | -4.03905 to -0.97992 | -4.73168 to -1.04035 | +0.09256 to +2.66995 | +0.00000 to +0.00000 | 16 | 18 | +0.07290 to +0.29693 |
| full / mean_scene | -3.27787 to -0.28909 | -3.77762 to -0.21861 | +0.64530 to +3.94162 | +0.00000 to +0.00000 | 15 | 18 | +0.05302 to +0.40419 |
| full / mean_joint | -0.75681 to +0.85588 | -0.54216 to +1.02611 | +2.60860 to +4.41676 | +0.00000 to +0.00000 | 6 | 18 | +0.26871 to +0.42555 |
| full / selected_all | -1.73443 to -0.30116 | -2.45010 to -0.27905 | +1.88274 to +3.63251 | +0.00000 to +0.00000 | 12 | 18 | +0.21784 to +0.36166 |
| full / selected_dual | -4.12548 to -1.25262 | -4.89265 to -1.35013 | +0.07737 to +2.38396 | +0.00000 to +0.00000 | 17 | 18 | +0.06575 to +0.26736 |
| full / selected_scene | -3.61660 to -0.33138 | -4.48521 to -0.41289 | +0.29364 to +3.46987 | +0.00000 to +0.00000 | 15 | 18 | +0.04087 to +0.35874 |
| full / selected_joint | -1.49111 to +0.78578 | -1.57796 to +0.87031 | +2.31920 to +4.34929 | +0.00000 to +0.00000 | 9 | 18 | +0.23063 to +0.41200 |
| full / selected_hash_matched | -2.20455 to +0.76438 | -2.68967 to +0.90332 | +1.64062 to +4.32840 | +0.00000 to +0.00000 | 4 | 18 | +0.23063 to +0.41200 |
| motion_only / reference | -4.06577 to -2.87137 | -5.31206 to -2.73188 | +0.00000 to +0.00000 | +0.00000 to +0.00000 | 18 | 18 | +0.00000 to +0.00000 |
| motion_only / raw_neural | +0.00000 to +0.00000 | +0.00000 to +0.00000 | +2.78156 to +3.86793 | +0.00000 to +1.86021 | 6 | 18 | +0.19615 to +0.35264 |
| motion_only / raw_ridge | -1.96597 to +0.06876 | -2.17820 to +0.21859 | +1.37961 to +3.89013 | +0.00000 to +1.85734 | 12 | 18 | +0.08253 to +0.31753 |
| motion_only / mean_all | -0.93755 to -0.05108 | -1.07369 to -0.03224 | +2.13888 to +3.70645 | +0.00000 to +1.46715 | 14 | 18 | +0.14464 to +0.33387 |
| motion_only / mean_dual | -2.64001 to -0.61500 | -3.48937 to -0.77730 | +0.28638 to +2.62441 | +0.00000 to +0.83516 | 17 | 18 | +0.06328 to +0.28270 |
| motion_only / mean_scene | -1.46761 to -0.01421 | -1.56294 to -0.00744 | +2.04530 to +3.38071 | +0.00000 to +0.81910 | 13 | 18 | +0.17149 to +0.31184 |
| motion_only / mean_joint | -0.58800 to +0.26173 | -0.67157 to +0.37903 | +2.27160 to +3.77560 | +0.00000 to +0.83570 | 12 | 18 | +0.19817 to +0.32952 |
| motion_only / selected_all | -1.21071 to -0.22003 | -1.35228 to -0.23875 | +1.87516 to +3.33033 | +0.00000 to +1.46715 | 15 | 18 | +0.14818 to +0.31915 |
| motion_only / selected_dual | -3.83604 to -1.22770 | -4.81712 to -1.45148 | +0.06068 to +2.03291 | +0.00000 to +0.00000 | 18 | 18 | +0.01381 to +0.19720 |
| motion_only / selected_scene | -2.33216 to -0.25830 | -2.88581 to -0.28759 | +1.54702 to +2.96950 | +0.00000 to +0.00000 | 17 | 18 | +0.13017 to +0.28562 |
| motion_only / selected_joint | -1.17961 to +0.12204 | -1.58761 to +0.17214 | +2.11881 to +3.36083 | +0.00000 to +0.00000 | 15 | 18 | +0.19238 to +0.30687 |
| motion_only / selected_hash_matched | -1.39877 to +0.08578 | -1.85926 to +0.14119 | +2.07052 to +3.30085 | +0.00000 to +0.00000 | 15 | 18 | +0.19238 to +0.30687 |

Policy ranges include every seed and source assignment. Switch rates are fractions, not percentages.
Complete risk checks require both positive-harm events, net easy <=2%, and no zero-CV harm in every C locality.
Reference-only risk is structural abstention, not positive learned contribution.

## Three-Seed Contrasts

| Pair / A-B role | All-ADE contrast | Point (%) | 95% locality-bootstrap interval (%) |
|---|---|---:|---:|
| full / producer0_controller1 | mean_all_vs_raw_neural__all | -0.42544 | -0.58939 to -0.29961 |
| full / producer0_controller1 | mean_joint_vs_mean_dual__all | +2.49762 | +0.94164 to +3.84450 |
| full / producer0_controller1 | selected_all_vs_mean_all__all | -0.24512 | -0.41588 to -0.08344 |
| full / producer0_controller1 | selected_dual_vs_mean_dual__all | -0.73575 | -1.02244 to -0.37784 |
| full / producer0_controller1 | selected_dual_vs_raw_neural__all | -3.52875 | -5.11472 to -1.76745 |
| full / producer0_controller1 | selected_joint_vs_mean_joint__all | -0.36142 | -0.62899 to -0.16415 |
| full / producer0_controller1 | selected_joint_vs_raw_neural__all | -0.54238 | -1.34635 to +0.10165 |
| full / producer0_controller1 | selected_joint_vs_selected_dual__all | +2.85705 | +1.05285 to +4.34553 |
| full / producer0_controller1 | selected_joint_vs_selected_hash_matched__all | +0.16967 | +0.07668 to +0.26265 |
| full / producer0_controller1 | selected_scene_vs_mean_scene__all | -0.66294 | -1.04421 to -0.28167 |
| full / producer0_controller2 | mean_all_vs_raw_neural__all | -1.01525 | -1.29619 to -0.85437 |
| full / producer0_controller2 | mean_joint_vs_mean_dual__all | +2.21585 | +1.79852 to +2.66177 |
| full / producer0_controller2 | selected_all_vs_mean_all__all | -0.17736 | -0.38192 to +0.02720 |
| full / producer0_controller2 | selected_dual_vs_mean_dual__all | -0.47990 | -0.87464 to -0.08515 |
| full / producer0_controller2 | selected_dual_vs_raw_neural__all | -2.43066 | -2.84382 to -1.85145 |
| full / producer0_controller2 | selected_joint_vs_mean_joint__all | -0.41340 | -0.69311 to -0.13369 |
| full / producer0_controller2 | selected_joint_vs_raw_neural__all | -0.09363 | -0.86774 to +0.34527 |
| full / producer0_controller2 | selected_joint_vs_selected_dual__all | +2.27850 | +1.73193 to +2.82507 |
| full / producer0_controller2 | selected_joint_vs_selected_hash_matched__all | +0.02144 | -0.05053 to +0.08543 |
| full / producer0_controller2 | selected_scene_vs_mean_scene__all | -0.24218 | -0.59363 to +0.10928 |
| full / producer1_controller0 | mean_all_vs_raw_neural__all | -1.32386 | -1.89380 to -0.53818 |
| full / producer1_controller0 | mean_joint_vs_mean_dual__all | +3.08536 | +1.20778 to +4.45253 |
| full / producer1_controller0 | selected_all_vs_mean_all__all | -0.16531 | -0.30059 to +0.05594 |
| full / producer1_controller0 | selected_dual_vs_mean_dual__all | -0.07572 | -0.08835 to -0.06572 |
| full / producer1_controller0 | selected_dual_vs_raw_neural__all | -3.94447 | -5.51043 to -1.84509 |
| full / producer1_controller0 | selected_joint_vs_mean_joint__all | -0.49970 | -0.72452 to -0.25316 |
| full / producer1_controller0 | selected_joint_vs_raw_neural__all | -1.13158 | -1.79778 to -0.54701 |
| full / producer1_controller0 | selected_joint_vs_selected_dual__all | +2.67844 | +1.00639 to +3.94440 |
| full / producer1_controller0 | selected_joint_vs_selected_hash_matched__all | +0.81789 | +0.27629 to +1.39242 |
| full / producer1_controller0 | selected_scene_vs_mean_scene__all | -0.31510 | -0.40957 to -0.20173 |
| full / producer1_controller2 | mean_all_vs_raw_neural__all | -1.05389 | -1.18810 to -0.79747 |
| full / producer1_controller2 | mean_joint_vs_mean_dual__all | +2.58328 | +1.45630 to +3.34533 |
| full / producer1_controller2 | selected_all_vs_mean_all__all | -0.10508 | -0.26979 to +0.03110 |
| full / producer1_controller2 | selected_dual_vs_mean_dual__all | -0.26180 | -0.33241 to -0.18206 |
| full / producer1_controller2 | selected_dual_vs_raw_neural__all | -2.30781 | -2.69906 to -1.91657 |
| full / producer1_controller2 | selected_joint_vs_mean_joint__all | -0.28965 | -0.44540 to -0.15588 |
| full / producer1_controller2 | selected_joint_vs_raw_neural__all | +0.31484 | -0.55507 to +0.84521 |
| full / producer1_controller2 | selected_joint_vs_selected_dual__all | +2.55639 | +1.35461 to +3.32611 |
| full / producer1_controller2 | selected_joint_vs_selected_hash_matched__all | +0.08206 | -0.07902 to +0.26336 |
| full / producer1_controller2 | selected_scene_vs_mean_scene__all | -0.15126 | -0.35507 to +0.00060 |
| full / producer2_controller0 | mean_all_vs_raw_neural__all | -0.60624 | -0.89247 to -0.31615 |
| full / producer2_controller0 | mean_joint_vs_mean_dual__all | +1.30182 | +0.87684 to +1.72681 |
| full / producer2_controller0 | selected_all_vs_mean_all__all | -0.01199 | -0.06620 to +0.03040 |
| full / producer2_controller0 | selected_dual_vs_mean_dual__all | -0.40161 | -0.58422 to -0.21899 |
| full / producer2_controller0 | selected_dual_vs_raw_neural__all | -1.99314 | -2.78325 to -1.20304 |
| full / producer2_controller0 | selected_joint_vs_mean_joint__all | -0.03236 | -0.07468 to +0.00012 |
| full / producer2_controller0 | selected_joint_vs_raw_neural__all | -0.29059 | -0.67341 to -0.07419 |
| full / producer2_controller0 | selected_joint_vs_selected_dual__all | +1.66388 | +1.08320 to +2.24457 |
| full / producer2_controller0 | selected_joint_vs_selected_hash_matched__all | +0.26382 | +0.13226 to +0.38262 |
| full / producer2_controller0 | selected_scene_vs_mean_scene__all | -0.04123 | -0.13099 to +0.01724 |
| full / producer2_controller1 | mean_all_vs_raw_neural__all | -0.29611 | -0.42708 to -0.14753 |
| full / producer2_controller1 | mean_joint_vs_mean_dual__all | +1.47760 | +1.02133 to +2.07715 |
| full / producer2_controller1 | selected_all_vs_mean_all__all | -0.12599 | -0.18861 to -0.07742 |
| full / producer2_controller1 | selected_dual_vs_mean_dual__all | -0.08076 | -0.17969 to +0.00175 |
| full / producer2_controller1 | selected_dual_vs_raw_neural__all | -1.76584 | -2.24861 to -1.40529 |
| full / producer2_controller1 | selected_joint_vs_mean_joint__all | -0.06969 | -0.11057 to -0.04473 |
| full / producer2_controller1 | selected_joint_vs_raw_neural__all | -0.24840 | -0.35704 to -0.14611 |
| full / producer2_controller1 | selected_joint_vs_selected_dual__all | +1.48823 | +1.08306 to +2.07053 |
| full / producer2_controller1 | selected_joint_vs_selected_hash_matched__all | +0.20776 | +0.05891 to +0.35662 |
| full / producer2_controller1 | selected_scene_vs_mean_scene__all | -0.06382 | -0.08154 to -0.04462 |
| motion_only / producer0_controller1 | mean_all_vs_raw_neural__all | -0.10979 | -0.19562 to -0.02396 |
| motion_only / producer0_controller1 | mean_joint_vs_mean_dual__all | +2.05682 | +0.98794 to +3.18237 |
| motion_only / producer0_controller1 | selected_all_vs_mean_all__all | -0.35172 | -0.54564 to -0.18203 |
| motion_only / producer0_controller1 | selected_dual_vs_mean_dual__all | -1.32653 | -2.19293 to -0.48560 |
| motion_only / producer0_controller1 | selected_dual_vs_raw_neural__all | -3.57736 | -5.25603 to -1.75144 |
| motion_only / producer0_controller1 | selected_joint_vs_mean_joint__all | -0.70779 | -1.23750 to -0.28466 |
| motion_only / producer0_controller1 | selected_joint_vs_raw_neural__all | -0.80946 | -1.63790 to -0.16751 |
| motion_only / producer0_controller1 | selected_joint_vs_selected_dual__all | +2.64492 | +0.83184 to +4.15866 |
| motion_only / producer0_controller1 | selected_joint_vs_selected_hash_matched__all | +0.16784 | +0.07024 to +0.26809 |
| motion_only / producer0_controller1 | selected_scene_vs_mean_scene__all | -0.98235 | -1.52101 to -0.44368 |
| motion_only / producer0_controller2 | mean_all_vs_raw_neural__all | -0.84267 | -1.05413 to -0.63121 |
| motion_only / producer0_controller2 | mean_joint_vs_mean_dual__all | +1.78754 | +1.28611 to +2.36448 |
| motion_only / producer0_controller2 | selected_all_vs_mean_all__all | -0.29601 | -0.45833 to -0.18689 |
| motion_only / producer0_controller2 | selected_dual_vs_mean_dual__all | -0.61525 | -0.93935 to -0.29115 |
| motion_only / producer0_controller2 | selected_dual_vs_raw_neural__all | -2.63122 | -3.05705 to -2.05910 |
| motion_only / producer0_controller2 | selected_joint_vs_mean_joint__all | -0.12971 | -0.25449 to -0.04682 |
| motion_only / producer0_controller2 | selected_joint_vs_raw_neural__all | -0.30804 | -0.96199 to +0.07281 |
| motion_only / producer0_controller2 | selected_joint_vs_selected_dual__all | +2.25946 | +1.73984 to +2.77908 |
| motion_only / producer0_controller2 | selected_joint_vs_selected_hash_matched__all | +0.01033 | -0.05618 to +0.07597 |
| motion_only / producer0_controller2 | selected_scene_vs_mean_scene__all | -0.14199 | -0.27953 to -0.00444 |
| motion_only / producer1_controller0 | mean_all_vs_raw_neural__all | -0.37197 | -0.58910 to -0.13332 |
| motion_only / producer1_controller0 | mean_joint_vs_mean_dual__all | +1.78074 | +0.69877 to +2.75884 |
| motion_only / producer1_controller0 | selected_all_vs_mean_all__all | -0.15062 | -0.30226 to +0.01425 |
| motion_only / producer1_controller0 | selected_dual_vs_mean_dual__all | -1.49361 | -2.42581 to -0.56140 |
| motion_only / producer1_controller0 | selected_dual_vs_raw_neural__all | -3.62923 | -5.01172 to -1.84575 |
| motion_only / producer1_controller0 | selected_joint_vs_mean_joint__all | -0.29383 | -0.65442 to +0.01319 |
| motion_only / producer1_controller0 | selected_joint_vs_raw_neural__all | -0.56485 | -0.75798 to -0.39500 |
| motion_only / producer1_controller0 | selected_joint_vs_selected_dual__all | +2.93165 | +1.32149 to +4.25519 |
| motion_only / producer1_controller0 | selected_joint_vs_selected_hash_matched__all | +0.33609 | +0.15704 to +0.52957 |
| motion_only / producer1_controller0 | selected_scene_vs_mean_scene__all | -0.77596 | -1.30191 to -0.25783 |
| motion_only / producer1_controller2 | mean_all_vs_raw_neural__all | -0.55355 | -0.66396 to -0.44315 |
| motion_only / producer1_controller2 | mean_joint_vs_mean_dual__all | +1.43981 | +0.92397 to +1.86836 |
| motion_only / producer1_controller2 | selected_all_vs_mean_all__all | -0.40803 | -0.61721 to -0.22901 |
| motion_only / producer1_controller2 | selected_dual_vs_mean_dual__all | -0.92182 | -1.26780 to -0.57584 |
| motion_only / producer1_controller2 | selected_dual_vs_raw_neural__all | -2.18492 | -2.78711 to -1.58274 |
| motion_only / producer1_controller2 | selected_joint_vs_mean_joint__all | -0.36489 | -0.59586 to -0.13392 |
| motion_only / producer1_controller2 | selected_joint_vs_raw_neural__all | -0.15296 | -0.72356 to +0.26566 |
| motion_only / producer1_controller2 | selected_joint_vs_selected_dual__all | +1.97958 | +1.13147 to +2.79156 |
| motion_only / producer1_controller2 | selected_joint_vs_selected_hash_matched__all | +0.01463 | -0.02441 to +0.07160 |
| motion_only / producer1_controller2 | selected_scene_vs_mean_scene__all | -0.39274 | -0.62706 to -0.15842 |
| motion_only / producer2_controller0 | mean_all_vs_raw_neural__all | -0.30283 | -0.38767 to -0.14962 |
| motion_only / producer2_controller0 | mean_joint_vs_mean_dual__all | +1.86323 | +1.36532 to +2.36113 |
| motion_only / producer2_controller0 | selected_all_vs_mean_all__all | -0.06802 | -0.13348 to -0.00298 |
| motion_only / producer2_controller0 | selected_dual_vs_mean_dual__all | -0.22781 | -0.33713 to -0.13329 |
| motion_only / producer2_controller0 | selected_dual_vs_raw_neural__all | -2.68243 | -3.76242 to -1.78970 |
| motion_only / producer2_controller0 | selected_joint_vs_mean_joint__all | -0.07519 | -0.11113 to -0.03905 |
| motion_only / producer2_controller0 | selected_joint_vs_raw_neural__all | -0.61076 | -1.29191 to -0.23619 |
| motion_only / producer2_controller0 | selected_joint_vs_selected_dual__all | +2.01247 | +1.47120 to +2.55373 |
| motion_only / producer2_controller0 | selected_joint_vs_selected_hash_matched__all | +0.03416 | -0.01047 to +0.08922 |
| motion_only / producer2_controller0 | selected_scene_vs_mean_scene__all | -0.13006 | -0.18692 to -0.06915 |
| motion_only / producer2_controller1 | mean_all_vs_raw_neural__all | -0.26717 | -0.36608 to -0.16903 |
| motion_only / producer2_controller1 | mean_joint_vs_mean_dual__all | +0.92226 | +0.61951 to +1.34146 |
| motion_only / producer2_controller1 | selected_all_vs_mean_all__all | -0.13829 | -0.17781 to -0.09876 |
| motion_only / producer2_controller1 | selected_dual_vs_mean_dual__all | -1.12645 | -1.51358 to -0.73932 |
| motion_only / producer2_controller1 | selected_dual_vs_raw_neural__all | -2.32192 | -2.96802 to -1.67583 |
| motion_only / producer2_controller1 | selected_joint_vs_mean_joint__all | -0.26731 | -0.40451 to -0.13011 |
| motion_only / producer2_controller1 | selected_joint_vs_raw_neural__all | -0.51453 | -0.81996 to -0.20911 |
| motion_only / producer2_controller1 | selected_joint_vs_selected_dual__all | +1.76129 | +1.09503 to +2.42756 |
| motion_only / producer2_controller1 | selected_joint_vs_selected_hash_matched__all | +0.01573 | -0.02289 to +0.04785 |
| motion_only / producer2_controller1 | selected_scene_vs_mean_scene__all | -0.27651 | -0.38838 to -0.17683 |

Seed means are computed within locality before 3,000 resamples. No multiplicity-adjusted discovery,
independent risk certificate or held-selection/confirmation claim. Source C was historically opened development.
Support-bin diagnostics in group JSON never select a threshold or reject a forecast. The hash control
matches query counts, not risk budgets. Scene-query allocation is greedy, not optimal or collision-aware.

No change to deployment. Image pixels, annotation steps, detector-derived labels. No metric/seconds,
human-gold, physical-safety, true3D or foundation claim. Stage5C and SMC are off.
