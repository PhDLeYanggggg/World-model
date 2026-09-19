# Native Motion: Resolution And Window Evidence

Source: fresh_run native decoding, motion extraction and fixed logistic fitting; cached_verified source population and old controls. No model selection or deployment.

## Scope And Cost

All 15,430 stationary-history queries across 29 recordings and four explored SDD sites are retained. This is the approved offline supplied-annotation source diagnostic, not the full primary benchmark, independent confirmation or external transfer. Eight observed and twelve predicted steps use a stride of 12 raw frames. No seconds, metric, true-3D or foundation claim. Bookstore, main and outer populations remain unscored.

- Native past crops: 25,300; exact old-pixel reductions: 25,300.
- Native cache: 889.66 MiB; decode/crop time: 354.50 s.
- Pair measurements: 95,560; flow time: 136.66 s.
- Logistic models: 64; fitting time: 111.87 s; max iterations: 1462.
- Native arm64 CPU, four compute threads, single-process data loading; completed fits checkpointed separately.
- No new neural trajectory training in this comparison. The four measurements do not establish body-motion truth.

## Measurement Support

| Variant | Box support | Surround support | Mean box magnitude (annotation px/past pair) |
| --- | ---: | ---: | ---: |
| lowpass_w45 | 98.874% | 99.958% | 0.716446 |
| lowpass_w15 | 98.741% | 99.959% | 0.783998 |
| native_w45 | 99.668% | 99.968% | 0.577954 |
| native_w15 | 99.595% | 99.968% | 0.603708 |

Support is a coverage/consistency proxy, not verified motion accuracy. Nominal window extent is matched; pixel lattice, polynomial support and pyramid operation are not perfectly isolated. All 23,890 lowpass_w45 pair features match the previous control exactly. Quality controls include flow-consistency and support flags; the motion contrast measures the added vectors and magnitudes, not all motion-derived information versus none.

## Complete Probability Results

The supervision labels are exact raw future coordinates, used only for loss/evaluation. There are 728 strictly-greater-than-10-pixel cases. Earlier 739-label results are not substituted as controls; all new lowpass and native probes use the same exact labels. No post-outcome relabeling, threshold search or probe selection.

| Target | Variant | Input | Equal-site Brier | Log loss | AUROC | AUPRC | Sites beating prevalence Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| any_nonzero | lowpass_w45 | quality | 0.25233524 | 0.72330116 | 0.58432882 | 0.46165919 | 2/4 |
| any_nonzero | lowpass_w45 | motion | 0.25605307 | 0.73480561 | 0.57907642 | 0.44986787 | 2/4 |
| any_nonzero | lowpass_w15 | quality | 0.24954146 | 0.71701184 | 0.58452159 | 0.46165947 | 3/4 |
| any_nonzero | lowpass_w15 | motion | 0.25493984 | 0.73098729 | 0.57932578 | 0.45204212 | 2/4 |
| any_nonzero | native_w45 | quality | 0.25187323 | 0.71197439 | 0.58294240 | 0.45622903 | 2/4 |
| any_nonzero | native_w45 | motion | 0.25582168 | 0.72597922 | 0.57914930 | 0.44970327 | 2/4 |
| any_nonzero | native_w15 | quality | 0.24861353 | 0.73859204 | 0.58297510 | 0.45895339 | 3/4 |
| any_nonzero | native_w15 | motion | 0.25171832 | 0.74596884 | 0.58127048 | 0.45514174 | 3/4 |
| over10_annotation_pixels | lowpass_w45 | quality | 0.04915677 | 0.22757157 | 0.47977941 | 0.05288021 | 0/4 |
| over10_annotation_pixels | lowpass_w45 | motion | 0.05064843 | 0.23247583 | 0.49227666 | 0.05526015 | 0/4 |
| over10_annotation_pixels | lowpass_w15 | quality | 0.04952397 | 0.22844752 | 0.47806899 | 0.05221665 | 0/4 |
| over10_annotation_pixels | lowpass_w15 | motion | 0.05079264 | 0.23335757 | 0.48466748 | 0.05145687 | 0/4 |
| over10_annotation_pixels | native_w45 | quality | 0.05020215 | 0.22885879 | 0.48520827 | 0.05089411 | 0/4 |
| over10_annotation_pixels | native_w45 | motion | 0.05167968 | 0.23840045 | 0.49455748 | 0.05159906 | 0/4 |
| over10_annotation_pixels | native_w15 | quality | 0.05023804 | 0.23615277 | 0.47657251 | 0.05184890 | 0/4 |
| over10_annotation_pixels | native_w15 | motion | 0.05156939 | 0.24147588 | 0.48258867 | 0.05010804 | 0/4 |

## Paired Contrasts

Positive values favor the first/new representation; Brier/log-loss values are reductions. All contrasts are reported, not a selected winner. There are 2,000 paired site resamples, four already-explored sites with shared training folds. These are unadjusted exploratory intervals, not independent confirmation, multiplicity-controlled discovery or risk calibration.

| Target | Contrast | Metric | Difference | Conditional 95% CI |
| --- | --- | --- | ---: | --- |
| any_nonzero | lowpass_w45_motion_vs_quality | brier | -0.00371783 | [-0.00889899, 0.00146334] |
| any_nonzero | lowpass_w45_motion_vs_quality | log_loss | -0.01150445 | [-0.02464735, 0.00158783] |
| any_nonzero | lowpass_w45_motion_vs_quality | auroc | -0.00525241 | [-0.01344008, 0.00293527] |
| any_nonzero | lowpass_w45_motion_vs_quality | auprc | -0.01179131 | [-0.02875057, 0.00514782] |
| over10_annotation_pixels | lowpass_w45_motion_vs_quality | brier | -0.00149165 | [-0.00343704, -0.00037831] |
| over10_annotation_pixels | lowpass_w45_motion_vs_quality | log_loss | -0.00490426 | [-0.01191079, -0.00042825] |
| over10_annotation_pixels | lowpass_w45_motion_vs_quality | auroc | 0.01249725 | [0.00731799, 0.02056406] |
| over10_annotation_pixels | lowpass_w45_motion_vs_quality | auprc | 0.00237994 | [0.00002205, 0.00542583] |
| any_nonzero | lowpass_w15_motion_vs_quality | brier | -0.00539838 | [-0.00890348, -0.00102220] |
| any_nonzero | lowpass_w15_motion_vs_quality | log_loss | -0.01397544 | [-0.02403080, -0.00244959] |
| any_nonzero | lowpass_w15_motion_vs_quality | auroc | -0.00519581 | [-0.01349385, 0.00308898] |
| any_nonzero | lowpass_w15_motion_vs_quality | auprc | -0.00961736 | [-0.02340085, 0.00407513] |
| over10_annotation_pixels | lowpass_w15_motion_vs_quality | brier | -0.00126867 | [-0.00245887, -0.00029800] |
| over10_annotation_pixels | lowpass_w15_motion_vs_quality | log_loss | -0.00491006 | [-0.01046369, -0.00043813] |
| over10_annotation_pixels | lowpass_w15_motion_vs_quality | auroc | 0.00659849 | [0.00058483, 0.01238100] |
| over10_annotation_pixels | lowpass_w15_motion_vs_quality | auprc | -0.00075978 | [-0.00620316, 0.00380081] |
| any_nonzero | native_w45_motion_vs_quality | brier | -0.00394845 | [-0.00826951, 0.00037261] |
| any_nonzero | native_w45_motion_vs_quality | log_loss | -0.01400483 | [-0.02676918, -0.00131013] |
| any_nonzero | native_w45_motion_vs_quality | auroc | -0.00379310 | [-0.00952321, 0.00300324] |
| any_nonzero | native_w45_motion_vs_quality | auprc | -0.00652577 | [-0.01485167, 0.00102354] |
| over10_annotation_pixels | native_w45_motion_vs_quality | brier | -0.00147753 | [-0.00225837, -0.00061725] |
| over10_annotation_pixels | native_w45_motion_vs_quality | log_loss | -0.00954167 | [-0.02253140, 0.00013065] |
| over10_annotation_pixels | native_w45_motion_vs_quality | auroc | 0.00934921 | [0.00065861, 0.01803980] |
| over10_annotation_pixels | native_w45_motion_vs_quality | auprc | 0.00070495 | [-0.00084898, 0.00356972] |
| any_nonzero | native_w15_motion_vs_quality | brier | -0.00310479 | [-0.00526163, 0.00023694] |
| any_nonzero | native_w15_motion_vs_quality | log_loss | -0.00737679 | [-0.01159383, 0.00007815] |
| any_nonzero | native_w15_motion_vs_quality | auroc | -0.00170462 | [-0.01156058, 0.00492572] |
| any_nonzero | native_w15_motion_vs_quality | auprc | -0.00381165 | [-0.00867884, 0.00242329] |
| over10_annotation_pixels | native_w15_motion_vs_quality | brier | -0.00133134 | [-0.00257617, -0.00028425] |
| over10_annotation_pixels | native_w15_motion_vs_quality | log_loss | -0.00532310 | [-0.01441933, 0.00037219] |
| over10_annotation_pixels | native_w15_motion_vs_quality | auroc | 0.00601616 | [0.00006247, 0.01206868] |
| over10_annotation_pixels | native_w15_motion_vs_quality | auprc | -0.00174086 | [-0.00876146, 0.00305658] |
| any_nonzero | native_minus_lowpass_w45 | brier | 0.00023139 | [-0.00185098, 0.00166227] |
| any_nonzero | native_minus_lowpass_w45 | log_loss | 0.00882639 | [-0.00637771, 0.03554012] |
| any_nonzero | native_minus_lowpass_w45 | auroc | 0.00007288 | [-0.00872132, 0.00907190] |
| any_nonzero | native_minus_lowpass_w45 | auprc | -0.00016461 | [-0.00935203, 0.01041533] |
| over10_annotation_pixels | native_minus_lowpass_w45 | brier | -0.00103125 | [-0.00158620, -0.00062479] |
| over10_annotation_pixels | native_minus_lowpass_w45 | log_loss | -0.00592462 | [-0.01411573, 0.00214865] |
| over10_annotation_pixels | native_minus_lowpass_w45 | auroc | 0.00228081 | [-0.01355008, 0.02506070] |
| over10_annotation_pixels | native_minus_lowpass_w45 | auprc | -0.00366109 | [-0.00647071, -0.00142890] |
| any_nonzero | native_minus_lowpass_w15 | brier | 0.00322152 | [-0.00146618, 0.00790923] |
| any_nonzero | native_minus_lowpass_w15 | log_loss | -0.01498155 | [-0.10290530, 0.04879008] |
| any_nonzero | native_minus_lowpass_w15 | auroc | 0.00194470 | [-0.01093019, 0.01489897] |
| any_nonzero | native_minus_lowpass_w15 | auprc | 0.00309963 | [-0.00950204, 0.01611114] |
| over10_annotation_pixels | native_minus_lowpass_w15 | brier | -0.00077675 | [-0.00237574, 0.00015940] |
| over10_annotation_pixels | native_minus_lowpass_w15 | log_loss | -0.00811830 | [-0.01722449, 0.00098788] |
| over10_annotation_pixels | native_minus_lowpass_w15 | auroc | -0.00207880 | [-0.00553768, 0.00247196] |
| over10_annotation_pixels | native_minus_lowpass_w15 | auprc | -0.00134883 | [-0.00265142, -0.00004624] |
| any_nonzero | w15_minus_w45_lowpass | brier | 0.00111323 | [-0.00182734, 0.00405379] |
| any_nonzero | w15_minus_w45_lowpass | log_loss | 0.00381832 | [-0.00409058, 0.01309574] |
| any_nonzero | w15_minus_w45_lowpass | auroc | 0.00024936 | [-0.00350981, 0.00636260] |
| any_nonzero | w15_minus_w45_lowpass | auprc | 0.00217424 | [0.00041836, 0.00527476] |
| over10_annotation_pixels | w15_minus_w45_lowpass | brier | -0.00014421 | [-0.00080415, 0.00051573] |
| over10_annotation_pixels | w15_minus_w45_lowpass | log_loss | -0.00088174 | [-0.00218132, 0.00049223] |
| over10_annotation_pixels | w15_minus_w45_lowpass | auroc | -0.00760919 | [-0.01213550, -0.00308287] |
| over10_annotation_pixels | w15_minus_w45_lowpass | auprc | -0.00380328 | [-0.00944063, -0.00043031] |
| any_nonzero | w15_minus_w45_native | brier | 0.00410336 | [-0.00122913, 0.00943585] |
| any_nonzero | w15_minus_w45_native | log_loss | -0.01998962 | [-0.09797598, 0.03126104] |
| any_nonzero | w15_minus_w45_native | auroc | 0.00212118 | [-0.00664465, 0.01126349] |
| any_nonzero | w15_minus_w45_native | auprc | 0.00543847 | [-0.00405530, 0.01675519] |
| over10_annotation_pixels | w15_minus_w45_native | brier | 0.00011029 | [-0.00112057, 0.00154622] |
| over10_annotation_pixels | w15_minus_w45_native | log_loss | -0.00307542 | [-0.02133047, 0.00843337] |
| over10_annotation_pixels | w15_minus_w45_native | auroc | -0.01196880 | [-0.03095464, 0.00505987] |
| over10_annotation_pixels | w15_minus_w45_native | auprc | -0.00149102 | [-0.00856148, 0.00300179] |

## Per-Site Readout

| Site | Target | Variant | Input | Brier | Prevalence Brier | AUROC | Positives/rows |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| coupa | any_nonzero | lowpass_w45 | quality | 0.29797404 | 0.28548541 | 0.571870 | 2546/4250 |
| coupa | over10_annotation_pixels | lowpass_w45 | quality | 0.03214200 | 0.03162165 | 0.397678 | 137/4250 |
| coupa | any_nonzero | lowpass_w45 | motion | 0.29600501 | 0.28548541 | 0.571582 | 2546/4250 |
| coupa | over10_annotation_pixels | lowpass_w45 | motion | 0.03264339 | 0.03162165 | 0.407299 | 137/4250 |
| coupa | any_nonzero | lowpass_w15 | quality | 0.28512096 | 0.28548541 | 0.581058 | 2546/4250 |
| coupa | over10_annotation_pixels | lowpass_w15 | quality | 0.03225469 | 0.03162165 | 0.382144 | 137/4250 |
| coupa | any_nonzero | lowpass_w15 | motion | 0.29036148 | 0.28548541 | 0.580974 | 2546/4250 |
| coupa | over10_annotation_pixels | lowpass_w15 | motion | 0.03227331 | 0.03162165 | 0.396407 | 137/4250 |
| coupa | any_nonzero | native_w45 | quality | 0.29541516 | 0.28548541 | 0.573291 | 2546/4250 |
| coupa | over10_annotation_pixels | native_w45 | quality | 0.03274499 | 0.03162165 | 0.385323 | 137/4250 |
| coupa | any_nonzero | native_w45 | motion | 0.29550084 | 0.28548541 | 0.561790 | 2546/4250 |
| coupa | over10_annotation_pixels | native_w45 | motion | 0.03449302 | 0.03162165 | 0.406697 | 137/4250 |
| coupa | any_nonzero | native_w15 | quality | 0.27862162 | 0.28548541 | 0.582689 | 2546/4250 |
| coupa | over10_annotation_pixels | native_w15 | quality | 0.03242399 | 0.03162165 | 0.383692 | 137/4250 |
| coupa | any_nonzero | native_w15 | motion | 0.28302020 | 0.28548541 | 0.566296 | 2546/4250 |
| coupa | over10_annotation_pixels | native_w15 | motion | 0.03234583 | 0.03162165 | 0.389979 | 137/4250 |
| deathCircle | any_nonzero | lowpass_w45 | quality | 0.22624428 | 0.24174490 | 0.605826 | 987/3054 |
| deathCircle | over10_annotation_pixels | lowpass_w45 | quality | 0.07090261 | 0.06293389 | 0.473166 | 204/3054 |
| deathCircle | any_nonzero | lowpass_w45 | motion | 0.23485978 | 0.24174490 | 0.594732 | 987/3054 |
| deathCircle | over10_annotation_pixels | lowpass_w45 | motion | 0.07531574 | 0.06293389 | 0.497386 | 204/3054 |
| deathCircle | any_nonzero | lowpass_w15 | quality | 0.22553326 | 0.24174490 | 0.606548 | 987/3054 |
| deathCircle | over10_annotation_pixels | lowpass_w15 | quality | 0.07157518 | 0.06293389 | 0.477081 | 204/3054 |
| deathCircle | any_nonzero | lowpass_w15 | motion | 0.23564036 | 0.24174490 | 0.592008 | 987/3054 |
| deathCircle | over10_annotation_pixels | lowpass_w15 | motion | 0.07465435 | 0.06293389 | 0.484006 | 204/3054 |
| deathCircle | any_nonzero | native_w45 | quality | 0.22669016 | 0.24174490 | 0.605366 | 987/3054 |
| deathCircle | over10_annotation_pixels | native_w45 | quality | 0.07372952 | 0.06293389 | 0.477644 | 204/3054 |
| deathCircle | any_nonzero | native_w45 | motion | 0.23355426 | 0.24174490 | 0.597821 | 987/3054 |
| deathCircle | over10_annotation_pixels | native_w45 | motion | 0.07628537 | 0.06293389 | 0.480518 | 204/3054 |
| deathCircle | any_nonzero | native_w15 | quality | 0.22298243 | 0.24174490 | 0.610241 | 987/3054 |
| deathCircle | over10_annotation_pixels | native_w15 | quality | 0.07458598 | 0.06293389 | 0.474985 | 204/3054 |
| deathCircle | any_nonzero | native_w15 | motion | 0.22716319 | 0.24174490 | 0.613177 | 987/3054 |
| deathCircle | over10_annotation_pixels | native_w15 | motion | 0.07779784 | 0.06293389 | 0.480531 | 204/3054 |
| gates | any_nonzero | lowpass_w45 | quality | 0.20854567 | 0.23521530 | 0.639638 | 420/1662 |
| gates | over10_annotation_pixels | lowpass_w45 | quality | 0.04107512 | 0.03820429 | 0.506598 | 66/1662 |
| gates | any_nonzero | lowpass_w45 | motion | 0.21772815 | 0.23521530 | 0.623852 | 420/1662 |
| gates | over10_annotation_pixels | lowpass_w45 | motion | 0.04187200 | 0.03820429 | 0.513148 | 66/1662 |
| gates | any_nonzero | lowpass_w15 | quality | 0.20793965 | 0.23521530 | 0.632087 | 420/1662 |
| gates | over10_annotation_pixels | lowpass_w15 | quality | 0.04150330 | 0.03820429 | 0.513091 | 66/1662 |
| gates | any_nonzero | lowpass_w15 | motion | 0.21526410 | 0.23521530 | 0.619667 | 420/1662 |
| gates | over10_annotation_pixels | lowpass_w15 | motion | 0.04290281 | 0.03820429 | 0.511563 | 66/1662 |
| gates | any_nonzero | native_w45 | quality | 0.20603422 | 0.23521530 | 0.641143 | 420/1662 |
| gates | over10_annotation_pixels | native_w45 | quality | 0.04214153 | 0.03820429 | 0.535268 | 66/1662 |
| gates | any_nonzero | native_w45 | motion | 0.21570913 | 0.23521530 | 0.638498 | 420/1662 |
| gates | over10_annotation_pixels | native_w45 | motion | 0.04238184 | 0.03820429 | 0.549973 | 66/1662 |
| gates | any_nonzero | native_w15 | quality | 0.21088227 | 0.23521530 | 0.628031 | 420/1662 |
| gates | over10_annotation_pixels | native_w15 | quality | 0.04109349 | 0.03820429 | 0.510709 | 66/1662 |
| gates | any_nonzero | native_w15 | motion | 0.21650419 | 0.23521530 | 0.628136 | 420/1662 |
| gates | over10_annotation_pixels | native_w15 | motion | 0.04263851 | 0.03820429 | 0.508696 | 66/1662 |
| hyang | any_nonzero | lowpass_w45 | quality | 0.27657697 | 0.24762327 | 0.519981 | 2911/6464 |
| hyang | over10_annotation_pixels | lowpass_w45 | quality | 0.05250737 | 0.04721177 | 0.541675 | 321/6464 |
| hyang | any_nonzero | lowpass_w45 | motion | 0.27561933 | 0.24762327 | 0.526140 | 2911/6464 |
| hyang | over10_annotation_pixels | lowpass_w45 | motion | 0.05276259 | 0.04721177 | 0.551274 | 321/6464 |
| hyang | any_nonzero | lowpass_w15 | quality | 0.27957197 | 0.24762327 | 0.518393 | 2911/6464 |
| hyang | over10_annotation_pixels | lowpass_w15 | quality | 0.05276270 | 0.04721177 | 0.539959 | 321/6464 |
| hyang | any_nonzero | lowpass_w15 | motion | 0.27849342 | 0.24762327 | 0.524655 | 2911/6464 |
| hyang | over10_annotation_pixels | lowpass_w15 | motion | 0.05334008 | 0.04721177 | 0.546694 | 321/6464 |
| hyang | any_nonzero | native_w45 | quality | 0.27935337 | 0.24762327 | 0.511969 | 2911/6464 |
| hyang | over10_annotation_pixels | native_w45 | quality | 0.05219256 | 0.04721177 | 0.542599 | 321/6464 |
| hyang | any_nonzero | native_w45 | motion | 0.27852247 | 0.24762327 | 0.518489 | 2911/6464 |
| hyang | over10_annotation_pixels | native_w45 | motion | 0.05355847 | 0.04721177 | 0.541042 | 321/6464 |
| hyang | any_nonzero | native_w15 | quality | 0.28196780 | 0.24762327 | 0.510940 | 2911/6464 |
| hyang | over10_annotation_pixels | native_w15 | quality | 0.05284871 | 0.04721177 | 0.536905 | 321/6464 |
| hyang | any_nonzero | native_w15 | motion | 0.28018569 | 0.24762327 | 0.517472 | 2911/6464 |
| hyang | over10_annotation_pixels | native_w15 | motion | 0.05349536 | 0.04721177 | 0.551148 | 321/6464 |

## Verification And Boundaries

95,560 exact pair replays; 64 exact coefficient replays; 128 future-label poison queries; 48 prohibited training-role rejections. 486 artifacts unchanged through completed decoder/extractor/probe resume with zero new work.

The corpus constructor loads broader label arrays for provenance, but the image/flow functions take no labels and the input poison tests leave features unchanged. Offline interpolated supplied annotations are not certified sensor-as-of observations.

No trajectory ADE/FDE, neural selector advantage or safe intervention is established by probability ranking alone. Four explored sites and rare motion remain limiting. Independent calibration/confirmation, main predictor comparisons and final paper evidence are still incomplete. No Stage5C or SMC.

Probes SHA256: `fa6e112a5890887760e8d940c314a2dca28fe1409c7e7c610be8210aed25da8f`.
Verification SHA256: `a5a5bb8d5b3c2279c6de3eee8727429ca62269febeca300f516c21561b740eed`.
