# Support-Fractional Harm Results

## Material Passport
144 fresh Torch heads / 288,000 updates; matched cached mean heads; no new forecast or policy.
Previously opened source development. Three-site easy definitions vary across folds.

## Population Diagnostics

| Pair / subset / arm | Dependent views | Weak support | Event AUROC | AP/prevalence | Top10 harm share | Harm coverage |
|---|---:|---:|---:|---:|---:|---:|
| full / all / mean | 72 | 0 | 0.79177 | 2.43729 | 0.44608 | 0.66266 |
| full / all / fractional | 72 | 0 | 0.82512 | 3.87864 | 0.50050 | 0.89098 |
| full / envelope_positive / mean | 72 | 0 | 0.48578 | 1.08796 | 0.27396 | 0.66266 |
| full / envelope_positive / fractional | 72 | 0 | 0.61971 | 1.66631 | 0.30539 | 0.89098 |
| motion_only / all / mean | 72 | 30 | 0.87301 | 5.26018 | 0.43241 | 1.07677 |
| motion_only / all / fractional | 72 | 30 | 0.87500 | 5.19430 | 0.45862 | 1.18235 |
| motion_only / envelope_positive / mean | 72 | 30 | 0.55792 | 1.47664 | 0.14858 | 1.07677 |
| motion_only / envelope_positive / fractional | 72 | 30 | 0.55858 | 1.44127 | 0.13015 | 1.18235 |

## Paired Contrasts

All signs favor the fractional auxiliary when positive. MSE gains are percentages; tail differences are pp.
Coverage contrast is reduction in absolute log(predicted/actual harm), not percentage coverage.
Three seeds averaged within locality, then 3,000 resamples of four localities; exploratory, no multiplicity correction.

| Pair / source roles | Contrast | Point | 95% CI |
|---|---|---:|---:|
| full / producer0_controller1 | all__harm_MSE_gain_percent | -1.60466 | [-5.29103370141725, 0.6455178322885321] |
| full / producer0_controller1 | all__top10_gain_pp | 8.90896 | [1.9721504954305837, 15.84576487008907] |
| full / producer0_controller1 | all__AUROC_delta | 0.07022 | [0.0576581769680539, 0.08643029878938951] |
| full / producer0_controller1 | all__coverage_log_error_reduction | 0.60322 | [0.08690423124928873, 1.108277218822538] |
| full / producer0_controller1 | envelope_positive__harm_MSE_gain_percent | -1.60466 | [-5.291033701417251, 0.6455178322885355] |
| full / producer0_controller1 | envelope_positive__top10_gain_pp | 9.31530 | [4.531336308075727, 11.909568417979248] |
| full / producer0_controller1 | envelope_positive__AUROC_delta | 0.15660 | [0.12334910177974771, 0.20346497987586792] |
| full / producer0_controller1 | envelope_positive__coverage_log_error_reduction | 0.60322 | [0.0869042312492885, 1.108277218822538] |
| full / producer0_controller1 | all__D_all_MSE_gain_percent | -0.03365 | [-0.9764861116407306, 0.9631962463081395] |
| full / producer0_controller1 | all__H_all_MSE_gain_percent | 3.97300 | [1.7743995676936635, 7.684510197289632] |
| full / producer0_controller1 | all__D_easy_MSE_gain_percent | -0.18365 | [-0.5845652469318884, 0.1375563293687225] |
| full / producer0_controller1 | all__H_easy_MSE_gain_percent | -1.60466 | [-5.2910337014173345, 0.6455178322884836] |
| full / producer0_controller2 | all__harm_MSE_gain_percent | -25.74101 | [-51.91966342634327, -0.06610983410742843] |
| full / producer0_controller2 | all__top10_gain_pp | 3.47762 | [-4.188734677459424, 10.68996595192201] |
| full / producer0_controller2 | all__AUROC_delta | 0.02118 | [0.00998720406350642, 0.029187633661640934] |
| full / producer0_controller2 | all__coverage_log_error_reduction | 0.66453 | [-0.37518037812759913, 1.7042367974341017] |
| full / producer0_controller2 | envelope_positive__harm_MSE_gain_percent | -25.74101 | [-51.919663426343256, -0.06610983410741936] |
| full / producer0_controller2 | envelope_positive__top10_gain_pp | -5.27517 | [-22.191071642222397, 6.822673981825384] |
| full / producer0_controller2 | envelope_positive__AUROC_delta | 0.12349 | [0.02170080187935389, 0.29052355717205786] |
| full / producer0_controller2 | envelope_positive__coverage_log_error_reduction | 0.66453 | [-0.37518037812759897, 1.7042367974341013] |
| full / producer0_controller2 | all__D_all_MSE_gain_percent | 0.59387 | [-1.3073251569859408, 3.3005683380601303] |
| full / producer0_controller2 | all__H_all_MSE_gain_percent | 3.35223 | [-0.03440537340976285, 8.338985582837061] |
| full / producer0_controller2 | all__D_easy_MSE_gain_percent | -0.23943 | [-0.627474576405952, 0.3157288482680448] |
| full / producer0_controller2 | all__H_easy_MSE_gain_percent | -25.74101 | [-51.9196634263436, -0.06610983410780705] |
| full / producer1_controller0 | all__harm_MSE_gain_percent | -0.34665 | [-2.6488711514302494, 2.0804387641688806] |
| full / producer1_controller0 | all__top10_gain_pp | 10.07719 | [0.060161674374476615, 22.415721848765767] |
| full / producer1_controller0 | all__AUROC_delta | 0.03356 | [0.022632292896439345, 0.045852434765276166] |
| full / producer1_controller0 | all__coverage_log_error_reduction | 0.02463 | [-0.12454439731594229, 0.17380409085226875] |
| full / producer1_controller0 | envelope_positive__harm_MSE_gain_percent | -0.34665 | [-2.648871151430265, 2.080438764168887] |
| full / producer1_controller0 | envelope_positive__top10_gain_pp | 3.16727 | [-0.3379362127860113, 6.900869468311914] |
| full / producer1_controller0 | envelope_positive__AUROC_delta | 0.07734 | [0.06037411347814774, 0.10172832229053054] |
| full / producer1_controller0 | envelope_positive__coverage_log_error_reduction | 0.02463 | [-0.1245443973159423, 0.17380409085226878] |
| full / producer1_controller0 | all__D_all_MSE_gain_percent | -0.18120 | [-0.9049826842047526, 0.6525114478778771] |
| full / producer1_controller0 | all__H_all_MSE_gain_percent | -3.42513 | [-7.629591605909566, 0.5939685302732535] |
| full / producer1_controller0 | all__D_easy_MSE_gain_percent | -0.08257 | [-0.4118808322049665, 0.30866144550156804] |
| full / producer1_controller0 | all__H_easy_MSE_gain_percent | -0.34665 | [-2.6488711514302827, 2.0804387641687847] |
| full / producer1_controller2 | all__harm_MSE_gain_percent | -6.08104 | [-17.01257668508461, 0.7889239240493788] |
| full / producer1_controller2 | all__top10_gain_pp | 4.58732 | [-2.8677481327845964, 11.948025245574732] |
| full / producer1_controller2 | all__AUROC_delta | 0.04769 | [0.028795051097912413, 0.058148645184177075] |
| full / producer1_controller2 | all__coverage_log_error_reduction | 1.23306 | [0.27289537819004295, 2.2815905034929567] |
| full / producer1_controller2 | envelope_positive__harm_MSE_gain_percent | -6.08104 | [-17.012576685084593, 0.7889239240493858] |
| full / producer1_controller2 | envelope_positive__top10_gain_pp | 4.41273 | [-0.30260319794818386, 12.02664821372844] |
| full / producer1_controller2 | envelope_positive__AUROC_delta | 0.13928 | [0.11545972574798931, 0.1724065399374746] |
| full / producer1_controller2 | envelope_positive__coverage_log_error_reduction | 1.23306 | [0.2728953781900431, 2.2815905034929567] |
| full / producer1_controller2 | all__D_all_MSE_gain_percent | -0.75828 | [-0.9535661330458627, -0.5249373934467233] |
| full / producer1_controller2 | all__H_all_MSE_gain_percent | 3.94480 | [-1.2649410740529663, 13.413613908441222] |
| full / producer1_controller2 | all__D_easy_MSE_gain_percent | -0.35442 | [-0.9399261309080049, 0.23109310141101608] |
| full / producer1_controller2 | all__H_easy_MSE_gain_percent | -6.08104 | [-17.0125766850848, 0.788923924049232] |
| full / producer2_controller0 | all__harm_MSE_gain_percent | 0.78337 | [-0.2181598129614374, 1.903618702965069] |
| full / producer2_controller0 | all__top10_gain_pp | 0.99577 | [-0.9106342099004627, 2.9021728484739127] |
| full / producer2_controller0 | all__AUROC_delta | -0.00040 | [-0.010312839700431165, 0.006287277596611426] |
| full / producer2_controller0 | all__coverage_log_error_reduction | 0.07833 | [0.02426595445654556, 0.15878437133577766] |
| full / producer2_controller0 | envelope_positive__harm_MSE_gain_percent | 0.78337 | [-0.21815981296143983, 1.9036187029650797] |
| full / producer2_controller0 | envelope_positive__top10_gain_pp | 0.81418 | [-0.337434027956691, 1.9658014171899858] |
| full / producer2_controller0 | envelope_positive__AUROC_delta | 0.00587 | [-0.02279090156953406, 0.030755071506126996] |
| full / producer2_controller0 | envelope_positive__coverage_log_error_reduction | 0.07833 | [0.02426595445654578, 0.1587843713357778] |
| full / producer2_controller0 | all__D_all_MSE_gain_percent | -1.49766 | [-3.108300942435815, -0.28413168262935595] |
| full / producer2_controller0 | all__H_all_MSE_gain_percent | -4.42089 | [-10.972199042624382, -0.3461178170947279] |
| full / producer2_controller0 | all__D_easy_MSE_gain_percent | 0.45332 | [0.1321408649919807, 0.7744899909434382] |
| full / producer2_controller0 | all__H_easy_MSE_gain_percent | 0.78337 | [-0.21815981296133902, 1.9036187029651614] |
| full / producer2_controller1 | all__harm_MSE_gain_percent | -20.32179 | [-42.17896137308985, 1.5353764256507731] |
| full / producer2_controller1 | all__top10_gain_pp | 3.32915 | [2.194801866859163, 5.45880292200937] |
| full / producer2_controller1 | all__AUROC_delta | 0.00285 | [0.000982533234444425, 0.005321796998592644] |
| full / producer2_controller1 | all__coverage_log_error_reduction | 0.03434 | [-0.015611921555074553, 0.08428893538896258] |
| full / producer2_controller1 | envelope_positive__harm_MSE_gain_percent | -20.32179 | [-42.178961373089855, 1.535376425650772] |
| full / producer2_controller1 | envelope_positive__top10_gain_pp | 1.50402 | [-0.414666226246848, 2.9914849023046726] |
| full / producer2_controller1 | envelope_positive__AUROC_delta | 0.00788 | [0.002917979218161236, 0.01500551306629195] |
| full / producer2_controller1 | envelope_positive__coverage_log_error_reduction | 0.03434 | [-0.015611921555074553, 0.08428893538896255] |
| full / producer2_controller1 | all__D_all_MSE_gain_percent | -0.43653 | [-1.7598738549308104, 1.194482389831935] |
| full / producer2_controller1 | all__H_all_MSE_gain_percent | -4.10364 | [-24.57330533562221, 8.200826219284902] |
| full / producer2_controller1 | all__D_easy_MSE_gain_percent | -0.04663 | [-0.25483541729623205, 0.20463355546119627] |
| full / producer2_controller1 | all__H_easy_MSE_gain_percent | -20.32179 | [-42.17896137308995, 1.5353764256506657] |
| motion_only / producer0_controller1 | all__harm_MSE_gain_percent | 2.04011 | [-1.6973889357860819, 7.7636304592070955] |
| motion_only / producer0_controller1 | all__top10_gain_pp | 1.56523 | [-0.9762400680981665, 4.34673084788859] |
| motion_only / producer0_controller1 | all__AUROC_delta | 0.00127 | [-0.009111716464221912, 0.01165011651430357] |
| motion_only / producer0_controller1 | all__coverage_log_error_reduction | 0.09620 | [-0.15559720847751912, 0.3479935147552219] |
| motion_only / producer0_controller1 | envelope_positive__harm_MSE_gain_percent | 2.04011 | [-1.6973889357860523, 7.763630459207096] |
| motion_only / producer0_controller1 | envelope_positive__top10_gain_pp | -0.07533 | [-0.7610367460880822, 0.6327250405965326] |
| motion_only / producer0_controller1 | envelope_positive__AUROC_delta | 0.00821 | [-0.03442456850233414, 0.05074402134054566] |
| motion_only / producer0_controller1 | envelope_positive__coverage_log_error_reduction | 0.09620 | [-0.155597208477519, 0.347993514755222] |
| motion_only / producer0_controller1 | all__D_all_MSE_gain_percent | -0.48846 | [-1.1314774630032156, 0.07541448769736528] |
| motion_only / producer0_controller1 | all__H_all_MSE_gain_percent | 8.80858 | [0.8156118602473303, 17.708062444079616] |
| motion_only / producer0_controller1 | all__D_easy_MSE_gain_percent | -0.19398 | [-0.38310871794492507, -0.0048510035967323645] |
| motion_only / producer0_controller1 | all__H_easy_MSE_gain_percent | 2.04011 | [-1.6973889357861474, 7.7636304592070715] |
| motion_only / producer0_controller2 | all__harm_MSE_gain_percent | -50.34640 | [-158.98855303109326, 8.167359996509138] |
| motion_only / producer0_controller2 | all__top10_gain_pp | 4.61760 | [-0.9101840338650622, 14.849273606512378] |
| motion_only / producer0_controller2 | all__AUROC_delta | 0.00045 | [-0.004147145029933898, 0.005422945873961925] |
| motion_only / producer0_controller2 | all__coverage_log_error_reduction | 0.35765 | [-0.296581346634594, 1.0043965958623597] |
| motion_only / producer0_controller2 | envelope_positive__harm_MSE_gain_percent | -50.34640 | [-158.9885530310933, 8.167359996509145] |
| motion_only / producer0_controller2 | envelope_positive__top10_gain_pp | -5.80256 | [-16.895963247817193, 0.17757248246666313] |
| motion_only / producer0_controller2 | envelope_positive__AUROC_delta | 0.00489 | [-0.009494774832317893, 0.01863106794967031] |
| motion_only / producer0_controller2 | envelope_positive__coverage_log_error_reduction | 0.35765 | [-0.2965813466345941, 1.0043965958623597] |
| motion_only / producer0_controller2 | all__D_all_MSE_gain_percent | -0.89904 | [-1.8824072725857945, -0.25579804837004727] |
| motion_only / producer0_controller2 | all__H_all_MSE_gain_percent | -7.63364 | [-27.564887515449804, 4.8748721185608765] |
| motion_only / producer0_controller2 | all__D_easy_MSE_gain_percent | 0.04173 | [-0.7686688891030581, 1.3572315521616851] |
| motion_only / producer0_controller2 | all__H_easy_MSE_gain_percent | -50.34640 | [-158.9885530310931, 8.167359996508686] |
| motion_only / producer1_controller0 | all__harm_MSE_gain_percent | -23.84791 | [-69.28491553236353, 0.12753246428208448] |
| motion_only / producer1_controller0 | all__top10_gain_pp | 6.16813 | [0.2867028634005666, 14.752218188440901] |
| motion_only / producer1_controller0 | all__AUROC_delta | 0.00227 | [-0.0029930803876003316, 0.005298308287682037] |
| motion_only / producer1_controller0 | all__coverage_log_error_reduction | -0.10982 | [-0.4175814328321127, 0.20244306093045852] |
| motion_only / producer1_controller0 | envelope_positive__harm_MSE_gain_percent | -23.84791 | [-69.28491553236351, 0.12753246428209056] |
| motion_only / producer1_controller0 | envelope_positive__top10_gain_pp | 1.73151 | [-0.7359341750797839, 6.158681116063757] |
| motion_only / producer1_controller0 | envelope_positive__AUROC_delta | 0.01212 | [-0.008029174170536177, 0.027854651247256203] |
| motion_only / producer1_controller0 | envelope_positive__coverage_log_error_reduction | -0.10982 | [-0.4175814328321127, 0.20244306093045839] |
| motion_only / producer1_controller0 | all__D_all_MSE_gain_percent | 0.17575 | [-0.03197705587427751, 0.3232971794708607] |
| motion_only / producer1_controller0 | all__H_all_MSE_gain_percent | -1.22824 | [-4.353774019154774, 1.3385921057535242] |
| motion_only / producer1_controller0 | all__D_easy_MSE_gain_percent | 0.05277 | [-0.5445770526895808, 0.650118842878] |
| motion_only / producer1_controller0 | all__H_easy_MSE_gain_percent | -23.84791 | [-69.28491553236363, 0.12753246428183618] |
| motion_only / producer1_controller2 | all__harm_MSE_gain_percent | -31.98958 | [-95.370815781705, 0.038674084971429726] |
| motion_only / producer1_controller2 | all__top10_gain_pp | -3.07376 | [-13.463405487070492, 3.583047338543813] |
| motion_only / producer1_controller2 | all__AUROC_delta | -0.00101 | [-0.008113619100327846, 0.0039158265259754965] |
| motion_only / producer1_controller2 | all__coverage_log_error_reduction | 0.74966 | [-0.48320167316929274, 2.0975423131764863] |
| motion_only / producer1_controller2 | envelope_positive__harm_MSE_gain_percent | -31.98958 | [-95.37081578170499, 0.03867408497142886] |
| motion_only / producer1_controller2 | envelope_positive__top10_gain_pp | 0.72304 | [-0.7181692328600829, 2.1642513287419414] |
| motion_only / producer1_controller2 | envelope_positive__AUROC_delta | -0.00093 | [-0.021886667867429765, 0.014619034276782635] |
| motion_only / producer1_controller2 | envelope_positive__coverage_log_error_reduction | 0.74966 | [-0.48320167316929274, 2.0975423131764863] |
| motion_only / producer1_controller2 | all__D_all_MSE_gain_percent | 0.76243 | [-0.011798617810359471, 2.0162420237811194] |
| motion_only / producer1_controller2 | all__H_all_MSE_gain_percent | -1.55260 | [-7.090492214180321, 5.149259754217355] |
| motion_only / producer1_controller2 | all__D_easy_MSE_gain_percent | 0.28580 | [0.14136171814608497, 0.44302818718792114] |
| motion_only / producer1_controller2 | all__H_easy_MSE_gain_percent | -31.98958 | [-95.37081578170498, 0.038674084971506296] |
| motion_only / producer2_controller0 | all__harm_MSE_gain_percent | -13.07924 | [-39.70615827427939, 0.38484640046838153] |
| motion_only / producer2_controller0 | all__top10_gain_pp | -5.12446 | [-10.261308452944395, -1.1609874092138488] |
| motion_only / producer2_controller0 | all__AUROC_delta | -0.00415 | [-0.011294924794624175, 0.0016854947203516213] |
| motion_only / producer2_controller0 | all__coverage_log_error_reduction | 0.04031 | [-0.18848125883836186, 0.3340571755537018] |
| motion_only / producer2_controller0 | envelope_positive__harm_MSE_gain_percent | -13.07924 | [-39.70615827427936, 0.38484640046837587] |
| motion_only / producer2_controller0 | envelope_positive__top10_gain_pp | -0.43708 | [-2.704843614367368, 1.3935988601708664] |
| motion_only / producer2_controller0 | envelope_positive__AUROC_delta | -0.00637 | [-0.02738979316158141, 0.017068093323712022] |
| motion_only / producer2_controller0 | envelope_positive__coverage_log_error_reduction | 0.04031 | [-0.18848125883836192, 0.3340571755537016] |
| motion_only / producer2_controller0 | all__D_all_MSE_gain_percent | -0.09299 | [-0.906746974236357, 0.5414168525134319] |
| motion_only / producer2_controller0 | all__H_all_MSE_gain_percent | 1.96677 | [-1.1082304072376266, 5.538224624583238] |
| motion_only / producer2_controller0 | all__D_easy_MSE_gain_percent | 0.00732 | [-0.09830095039713022, 0.11212341443780155] |
| motion_only / producer2_controller0 | all__H_easy_MSE_gain_percent | -13.07924 | [-39.70615827427931, 0.3848464004683885] |
| motion_only / producer2_controller1 | all__harm_MSE_gain_percent | 1.68145 | [0.04853983837996388, 3.314358805281527] |
| motion_only / producer2_controller1 | all__top10_gain_pp | -0.46869 | [-1.9923237253605723, 0.7777557395070427] |
| motion_only / producer2_controller1 | all__AUROC_delta | -0.00859 | [-0.01742286042062845, 0.0005480980136535751] |
| motion_only / producer2_controller1 | all__coverage_log_error_reduction | 0.14041 | [-0.021366440326536857, 0.40799358730455737] |
| motion_only / producer2_controller1 | envelope_positive__harm_MSE_gain_percent | 1.68145 | [0.04853983837998474, 3.3143588052815143] |
| motion_only / producer2_controller1 | envelope_positive__top10_gain_pp | 1.58470 | [0.0, 4.543783740251768] |
| motion_only / producer2_controller1 | envelope_positive__AUROC_delta | -0.02445 | [-0.050348952492672586, 0.0033458208597460107] |
| motion_only / producer2_controller1 | envelope_positive__coverage_log_error_reduction | 0.14041 | [-0.02136644032653682, 0.4079935873045575] |
| motion_only / producer2_controller1 | all__D_all_MSE_gain_percent | -0.54317 | [-1.4399408112047867, 0.15218279606545093] |
| motion_only / producer2_controller1 | all__H_all_MSE_gain_percent | 2.18080 | [-0.3742393299079, 6.245377153772632] |
| motion_only / producer2_controller1 | all__D_easy_MSE_gain_percent | 0.18361 | [-0.07503789091464017, 0.40696176787848937] |
| motion_only / producer2_controller1 | all__H_easy_MSE_gain_percent | 1.68145 | [0.04853983838000371, 3.3143588052814934] |

## Gates

- training_complete: true
- matched_initialization_draws_steps: true
- locality_exclusion: true
- primary_six_positive: false
- tail_and_coverage_guards: true
- development_advance_gate: false
- new_policy_evaluated: false
- independent_confirmation: false
- deployment_changed: false
- submission_ready: false
- stage5c_executed: false
- smc_enabled: false

The fractional score is expected severity fraction, not a failure probability or risk certificate.
Image pixels, annotation steps, detector-derived labels; no metric/seconds, human-gold, true3D or foundation claim.
Reserved calibration/confirmation remain closed. No deployment. Stage5C and SMC remain off.
