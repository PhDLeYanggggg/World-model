# Frozen Factor Attribution

## Material Passport
Fresh offline attribution from cached_verified models. No new fitting, policy, threshold or independent evaluation.
Label-assisted substitutions require future-derived labels and are NOT deployable models.

| Pair / source roles | Current cost gain % | E-assisted gain % | H-assisted gain % | E-assisted 95% CI |
|---|---:|---:|---:|---:|
| full / producer0_controller1 | -5.777 | -172.455 | -34.499 | [-526.1040395297512, 6.60071511889151] |
| full / producer0_controller2 | 3.812 | -123.896 | -23.858 | [-414.10157969968833, 37.10099910139451] |
| full / producer1_controller0 | -88.839 | -81.285 | -13.042 | [-216.494927225646, 12.706840577281998] |
| full / producer1_controller2 | -1.863 | 2.830 | -43.844 | [-3.6204760337499007, 9.618601988941244] |
| full / producer2_controller0 | -12.361 | -26.906 | -136.798 | [-43.709882666135925, -10.101348541825844] |
| full / producer2_controller1 | -6.165 | 8.527 | -34.385 | [-0.6805534575422882, 17.734146382010465] |
| motion_only / producer0_controller1 | 2.018 | -1.501 | -433.717 | [-19.68207793028492, 13.457550533833286] |
| motion_only / producer0_controller2 | 22.824 | -143.884 | 21.151 | [-464.6197090459051, 29.93747250147854] |
| motion_only / producer1_controller0 | -159.326 | -677.112 | -300.855 | [-2018.6985075036869, 2.564232337362243] |
| motion_only / producer1_controller2 | -2.138 | -15.903 | -66.143 | [-56.63892743772249, 5.438189925201188] |
| motion_only / producer2_controller0 | -42.388 | -274.544 | -235.529 | [-740.9370022532156, 1.602457594771923] |
| motion_only / producer2_controller1 | 11.119 | 11.135 | -202.564 | [-12.740902052448824, 43.50340818137431] |

Gains compare easy-harm MSE to the original cost model, not ADE/FDE.
Three seeds averaged per locality; 3,000 resamples of four localities. Repeated source roles/windows are dependent and exploratory.
The signed cross term can be negative. Squared-term/MSE ratios are not independent causal shares or probabilities.
No change to the parent failed model gate. Pixels/annotation steps; no metric/seconds, gold, physical-safety, true3D or foundation claim. Stage5C/SMC off.
