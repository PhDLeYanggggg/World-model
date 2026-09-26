# Harm Tail Crossfit Results

## Material Passport
144 fresh Torch heads / 288,000 updates; 36 source-role/seed/pair groups. No new trajectory or policy result.
Inner B event cuts use three training localities; original B/C use their unchanged whole-B event cut.
Do not interpret those populations as matched event definitions. No held-out or C model selection.

| Pair / subset / population | Views | Weak support | Median event rate | Moment AUROC | AP/prevalence | Moment top10 harm share | Envelope top10 share | Harm coverage | Oracle top1 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full / all / inner_held_B | 72 | 0 | 0.02228 | 0.79177 | 2.43729 | 0.44608 | 0.09375 | 0.66266 | 0.90773 |
| full / all / original_fit_B | 72 | 1 | 0.02405 | 0.78741 | 2.48955 | 0.51077 | 0.09804 | 0.77836 | 0.89247 |
| full / all / original_held_C | 72 | 1 | 0.02405 | 0.78350 | 2.45039 | 0.48490 | 0.09804 | 0.80621 | 0.89247 |
| full / envelope_positive / inner_held_B | 72 | 0 | 0.05353 | 0.48578 | 1.08796 | 0.27396 | 0.00000 | 0.66266 | 0.67868 |
| full / envelope_positive / original_fit_B | 72 | 1 | 0.05558 | 0.51794 | 1.12422 | 0.34866 | 0.00246 | 0.77836 | 0.66753 |
| full / envelope_positive / original_held_C | 72 | 1 | 0.05558 | 0.47512 | 0.97880 | 0.22655 | 0.00246 | 0.80621 | 0.66753 |
| motion_only / all / inner_held_B | 72 | 30 | 0.00403 | 0.87301 | 5.26018 | 0.43241 | 0.25797 | 1.07677 | 1.00000 |
| motion_only / all / original_fit_B | 72 | 39 | 0.00346 | 0.88503 | 5.91918 | 0.58158 | 0.28547 | 1.02777 | 1.00000 |
| motion_only / all / original_held_C | 72 | 39 | 0.00346 | 0.88991 | 5.65815 | 0.48508 | 0.28547 | 1.06812 | 1.00000 |
| motion_only / envelope_positive / inner_held_B | 72 | 30 | 0.01540 | 0.55792 | 1.47664 | 0.14858 | 0.00000 | 1.07677 | 0.97653 |
| motion_only / envelope_positive / original_fit_B | 72 | 39 | 0.01470 | 0.60063 | 1.58890 | 0.24638 | 0.00000 | 1.02777 | 0.97016 |
| motion_only / envelope_positive / original_held_C | 72 | 39 | 0.01470 | 0.59154 | 1.47474 | 0.16335 | 0.00000 | 1.06812 | 0.97016 |

## Paired Tail-Retrieval Contrast

Difference in harm captured by the highest-scoring 10%: neural score minus envelope, percentage points.
Three-seed locality averages, then 3,000 resamples of four localities. Not independent confirmation.

| Pair / assignment | Subset / score | Point (pp) | 95% CI (pp) |
|---|---|---:|---:|
| full / producer0_controller1 | all__moment_vs_envelope | 19.58380 | [12.977385862132548, 28.27225381997436] |
| full / producer0_controller1 | all__fraction_vs_envelope | 26.07799 | [10.038347845753497, 42.11764051378145] |
| full / producer0_controller1 | envelope_positive__moment_vs_envelope | 10.00270 | [-0.44179449131476844, 20.44718802181434] |
| full / producer0_controller1 | envelope_positive__fraction_vs_envelope | 12.76360 | [4.948295316435228, 19.724377646710586] |
| full / producer0_controller2 | all__moment_vs_envelope | 8.10677 | [-9.700704748291464, 26.907876555873365] |
| full / producer0_controller2 | all__fraction_vs_envelope | 8.95023 | [-17.87140682829146, 35.77187382830863] |
| full / producer0_controller2 | envelope_positive__moment_vs_envelope | 11.65070 | [-0.4411264734798108, 23.742521208145714] |
| full / producer0_controller2 | envelope_positive__fraction_vs_envelope | 1.20155 | [-14.13731603783038, 17.703990499477666] |
| full / producer1_controller0 | all__moment_vs_envelope | 25.81743 | [18.08565102795468, 35.61827971831096] |
| full / producer1_controller0 | all__fraction_vs_envelope | 31.22219 | [6.767082254226702, 55.67730692859266] |
| full / producer1_controller0 | envelope_positive__moment_vs_envelope | 20.17985 | [14.186121817131191, 26.173581014497188] |
| full / producer1_controller0 | envelope_positive__fraction_vs_envelope | 18.44857 | [4.694407050553464, 32.20273487280697] |
| full / producer1_controller2 | all__moment_vs_envelope | 12.03672 | [-2.3268218984914233, 26.400253867772747] |
| full / producer1_controller2 | all__fraction_vs_envelope | 17.73021 | [-8.926549724876315, 44.386961660615405] |
| full / producer1_controller2 | envelope_positive__moment_vs_envelope | 4.48144 | [-1.9486457969732243, 9.435749543042304] |
| full / producer1_controller2 | envelope_positive__fraction_vs_envelope | 4.50218 | [-8.187011529859474, 17.191372013243807] |
| full / producer2_controller0 | all__moment_vs_envelope | 59.09907 | [47.60389795033528, 69.8773743329673] |
| full / producer2_controller0 | all__fraction_vs_envelope | 57.08157 | [40.73201100695272, 73.43113068009997] |
| full / producer2_controller0 | envelope_positive__moment_vs_envelope | 44.01438 | [36.375255627415804, 52.377176156617395] |
| full / producer2_controller0 | envelope_positive__fraction_vs_envelope | 35.92447 | [33.09968245009976, 38.74926343385299] |
| full / producer2_controller1 | all__moment_vs_envelope | 75.76959 | [60.43694244826199, 90.86992942131104] |
| full / producer2_controller1 | all__fraction_vs_envelope | 75.45130 | [57.30599001857426, 91.55458667477464] |
| full / producer2_controller1 | envelope_positive__moment_vs_envelope | 59.38345 | [42.12307780827876, 76.64382346936661] |
| full / producer2_controller1 | envelope_positive__fraction_vs_envelope | 52.39415 | [32.62563817503059, 72.1626546475932] |
| motion_only / producer0_controller1 | all__moment_vs_envelope | 29.51299 | [4.1851520782296685, 72.04442471341955] |
| motion_only / producer0_controller1 | all__fraction_vs_envelope | 45.17371 | [13.644194663671165, 76.7032291165054] |
| motion_only / producer0_controller1 | envelope_positive__moment_vs_envelope | 15.15627 | [4.979258034349152, 27.213381028373234] |
| motion_only / producer0_controller1 | envelope_positive__fraction_vs_envelope | 15.57695 | [5.269095822003128, 27.380868995638867] |
| motion_only / producer0_controller2 | all__moment_vs_envelope | 13.49389 | [-2.039617092402539, 30.97224360382259] |
| motion_only / producer0_controller2 | all__fraction_vs_envelope | 15.97034 | [-4.013353440175496, 35.95403855929659] |
| motion_only / producer0_controller2 | envelope_positive__moment_vs_envelope | 21.82209 | [6.150549216761507, 37.493633316405315] |
| motion_only / producer0_controller2 | envelope_positive__fraction_vs_envelope | 18.74752 | [5.013072699269771, 32.48196432631022] |
| motion_only / producer1_controller0 | all__moment_vs_envelope | 6.84799 | [-21.945176520675822, 35.64114722559983] |
| motion_only / producer1_controller0 | all__fraction_vs_envelope | 5.27657 | [-15.976516157747593, 26.529647882940417] |
| motion_only / producer1_controller0 | envelope_positive__moment_vs_envelope | 18.63072 | [6.397256004466481, 29.657378196704855] |
| motion_only / producer1_controller0 | envelope_positive__fraction_vs_envelope | 14.78046 | [5.434160705138301, 21.62566714843972] |
| motion_only / producer1_controller2 | all__moment_vs_envelope | 21.06966 | [3.3123297896918977, 40.729740465782164] |
| motion_only / producer1_controller2 | all__fraction_vs_envelope | 14.18215 | [-3.4497987033668327, 41.34736993027732] |
| motion_only / producer1_controller2 | envelope_positive__moment_vs_envelope | 5.07645 | [1.9342745010584115, 7.737832701376886] |
| motion_only / producer1_controller2 | envelope_positive__fraction_vs_envelope | 5.30588 | [0.1048666233178755, 11.825327811747803] |
| motion_only / producer2_controller0 | all__moment_vs_envelope | -2.06951 | [-36.540299242722334, 23.350480237785607] |
| motion_only / producer2_controller0 | all__fraction_vs_envelope | -22.60991 | [-46.0561934867481, 6.475761199206749] |
| motion_only / producer2_controller0 | envelope_positive__moment_vs_envelope | 12.72477 | [4.704088479458198, 19.262928142356984] |
| motion_only / producer2_controller0 | envelope_positive__fraction_vs_envelope | 2.86813 | [-28.16341811847692, 22.545710840490152] |
| motion_only / producer2_controller1 | all__moment_vs_envelope | -5.59555 | [-23.00659351261691, 9.753229471273134] |
| motion_only / producer2_controller1 | all__fraction_vs_envelope | -5.96522 | [-26.93114465522846, 11.630252309015182] |
| motion_only / producer2_controller1 | envelope_positive__moment_vs_envelope | 8.55700 | [0.0, 19.179815544288864] |
| motion_only / producer2_controller1 | envelope_positive__fraction_vs_envelope | 8.68260 | [1.0609387058435666, 21.603394319314308] |

Expected harm scores are not event probabilities. AUROC/AP assess ranking, not calibration.
Top10 shares are evaluation diagnostics, never a label-aware deployment rule. No policy or tolerance changes.
Overlapping windows/roles and fold-varying event cuts prohibit pooling all rows as independent evidence.
Image pixels, annotation steps, detector-derived labels; no metric/seconds, human-gold, true3D or foundation claim.
Selection/calibration/confirmation remain closed this round. Stage5C and SMC remain off.
