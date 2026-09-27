# Expected Cost and Harm-Mass Readout

## Material Passport
fresh_run:432 scalar moment readouts,loss decomposition and144 source-held views.
cached_verified:neural heads,forecasts,causal features and raw/L2 predictions.
not_run:new neural training,new policy,independent selection/calibration/confirmation.

Three seeds average within locality;3000 paired four-locality resamples. Overlapping exposed-source assignments;unadjusted intervals.
Primary remains expected easy-harm MSE,not trajectory accuracy or physical safety.

| Contrast/family | Positive/negative/overlap/missing CIs | Primary point range (%) |
|---|---|---|
| mass_cost_only_vs_raw_cost_only/full | [0, 3, 3, 0] | [-596.5590842867044, 0.5550650004142686] |
| mass_cost_only_vs_raw_cost_only/motion_only | [1, 4, 1, 0] | [-557.950857476144, 7.126695071190481] |
| mass_cost_only_vs_scaled_cost_only/full | [0, 4, 2, 0] | [-2938.764645735223, -5.988232223894374] |
| mass_cost_only_vs_scaled_cost_only/motion_only | [0, 4, 2, 0] | [-90969.13256985854, -15.494191445655671] |
| mass_cap_aux_vs_raw_cap_aux/full | [0, 5, 1, 0] | [-530.9406690462583, 0.9360812863276731] |
| mass_cap_aux_vs_raw_cap_aux/motion_only | [1, 4, 1, 0] | [-555.6474983001245, 7.655925971179614] |
| mass_cap_aux_vs_scaled_cap_aux/full | [0, 3, 3, 0] | [-3070.1321333875744, -7.853227711706299] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only | [0, 4, 2, 0] | [-150012.47555151978, -18.622317193065918] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full | [0, 3, 3, 0] | [-549.6983324612651, 0.2443703755780651] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only | [1, 4, 1, 0] | [-600.2920523628353, 7.196500668472464] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full | [0, 4, 2, 0] | [-2946.0724527270486, -4.496228936483449] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only | [0, 4, 2, 0] | [-89091.29683806901, -18.326246804817185] |
| mass_true_vs_mass_cost/full | [1, 1, 4, 0] | [-14.102855436733302, 1.390097189158866] |
| mass_true_vs_mass_cost/motion_only | [0, 1, 5, 0] | [-31.339516816389107, 0.20866352065661947] |
| mass_true_vs_mass_shuffled/full | [0, 1, 5, 0] | [-12.211877366639241, 0.7503601323397864] |
| mass_true_vs_mass_shuffled/motion_only | [0, 1, 5, 0] | [-31.980582418248314, 0.016428347514178432] |

## Every Primary Interval

| Contrast/family/assignment | Point (%) | 95% CI |
|---|---:|---|
| mass_cost_only_vs_raw_cost_only/full/producer0_controller1 | -17.12402061734369 | [-35.35610274903033, -0.6975285772584903] |
| mass_cost_only_vs_raw_cost_only/full/producer0_controller2 | -596.5590842867044 | [-1512.619049024243, -7.062754490376438] |
| mass_cost_only_vs_raw_cost_only/full/producer1_controller0 | 0.5550650004142686 | [-4.931273108104915, 7.035608337852728] |
| mass_cost_only_vs_raw_cost_only/full/producer1_controller2 | -84.5151102596484 | [-179.78844956581977, -0.7123111487312406] |
| mass_cost_only_vs_raw_cost_only/full/producer2_controller0 | -8.236824873788587 | [-21.512238232688212, 0.6714409772996968] |
| mass_cost_only_vs_raw_cost_only/full/producer2_controller1 | -4.772520191986869 | [-9.882995152156676, 0.33795476818293757] |
| mass_cost_only_vs_raw_cost_only/motion_only/producer0_controller1 | -328.0154064519762 | [-869.8536205226276, -2.1031125850993253] |
| mass_cost_only_vs_raw_cost_only/motion_only/producer0_controller2 | -557.950857476144 | [-1189.6765556768607, -9.859293200062192] |
| mass_cost_only_vs_raw_cost_only/motion_only/producer1_controller0 | 1.0007747683469248 | [-7.6520574438396025, 11.002970766706046] |
| mass_cost_only_vs_raw_cost_only/motion_only/producer1_controller2 | -315.508270551865 | [-914.1938127147732, -0.8407267676149035] |
| mass_cost_only_vs_raw_cost_only/motion_only/producer2_controller0 | 7.126695071190481 | [1.3904439045815913, 15.710153174281935] |
| mass_cost_only_vs_raw_cost_only/motion_only/producer2_controller1 | -105.80394238765633 | [-252.02541271897803, -3.0730269719429564] |
| mass_cost_only_vs_scaled_cost_only/full/producer0_controller1 | -18.79380145146524 | [-39.82648926528276, -0.6960822327595035] |
| mass_cost_only_vs_scaled_cost_only/full/producer0_controller2 | -2938.764645735223 | [-8313.815229529722, -7.135750737980732] |
| mass_cost_only_vs_scaled_cost_only/full/producer1_controller0 | -8.556783857916157 | [-20.67822187355755, -0.2741358818654818] |
| mass_cost_only_vs_scaled_cost_only/full/producer1_controller2 | -96.71899839817094 | [-210.43468687769706, -0.5760486525468742] |
| mass_cost_only_vs_scaled_cost_only/full/producer2_controller0 | -7.176749527251156 | [-22.92320820367103, 6.410877892574159] |
| mass_cost_only_vs_scaled_cost_only/full/producer2_controller1 | -5.988232223894374 | [-20.420309325715927, 8.44384487792718] |
| mass_cost_only_vs_scaled_cost_only/motion_only/producer0_controller1 | -90969.13256985854 | [-272765.05183555774, -2.2360521590899767] |
| mass_cost_only_vs_scaled_cost_only/motion_only/producer0_controller2 | -5118.752759817671 | [-12655.689778787157, -10.180084122784619] |
| mass_cost_only_vs_scaled_cost_only/motion_only/producer1_controller0 | -20.76791743899313 | [-62.0883789034724, 0.5074861792713014] |
| mass_cost_only_vs_scaled_cost_only/motion_only/producer1_controller2 | -445.6224964191016 | [-1303.2844914379614, -0.8351439388271795] |
| mass_cost_only_vs_scaled_cost_only/motion_only/producer2_controller0 | -15.494191445655671 | [-32.363737553514284, 0.19640257603748304] |
| mass_cost_only_vs_scaled_cost_only/motion_only/producer2_controller1 | -602.4596899548868 | [-1682.9214412443541, -3.163267483980719] |
| mass_cap_aux_vs_raw_cap_aux/full/producer0_controller1 | -19.325647915806115 | [-37.63373637946124, -1.017559452150991] |
| mass_cap_aux_vs_raw_cap_aux/full/producer0_controller2 | -530.9406690462583 | [-1355.0636697019975, -10.499139592534236] |
| mass_cap_aux_vs_raw_cap_aux/full/producer1_controller0 | 0.9360812863276731 | [-3.1450661438227936, 6.459695685866695] |
| mass_cap_aux_vs_raw_cap_aux/full/producer1_controller2 | -83.97728089344454 | [-196.33743010750982, -0.7495011677952746] |
| mass_cap_aux_vs_raw_cap_aux/full/producer2_controller0 | -5.717008314897865 | [-13.64739832996341, -1.2447830586598294] |
| mass_cap_aux_vs_raw_cap_aux/full/producer2_controller1 | -5.419702766994081 | [-9.198783176318871, -1.64062235766929] |
| mass_cap_aux_vs_raw_cap_aux/motion_only/producer0_controller1 | -338.7527052563081 | [-876.6625168587941, -1.7171670033151605] |
| mass_cap_aux_vs_raw_cap_aux/motion_only/producer0_controller2 | -555.6474983001245 | [-1298.7343505302897, -10.76863570743467] |
| mass_cap_aux_vs_raw_cap_aux/motion_only/producer1_controller0 | 0.6786369525508383 | [-9.972420909865177, 12.251010778300687] |
| mass_cap_aux_vs_raw_cap_aux/motion_only/producer1_controller2 | -397.72589710812105 | [-1161.6000548981583, -0.6184428864832087] |
| mass_cap_aux_vs_raw_cap_aux/motion_only/producer2_controller0 | 7.655925971179614 | [1.377047850191654, 17.128948746089243] |
| mass_cap_aux_vs_raw_cap_aux/motion_only/producer2_controller1 | -178.06266713151018 | [-413.37259919265205, -2.921774279117963] |
| mass_cap_aux_vs_scaled_cap_aux/full/producer0_controller1 | -21.55822067553561 | [-41.86659064907978, -1.2498507019914404] |
| mass_cap_aux_vs_scaled_cap_aux/full/producer0_controller2 | -3070.1321333875744 | [-8719.689175020045, -10.615673925503705] |
| mass_cap_aux_vs_scaled_cap_aux/full/producer1_controller0 | -7.853227711706299 | [-20.508495896302165, 0.15693435726000038] |
| mass_cap_aux_vs_scaled_cap_aux/full/producer1_controller2 | -96.13333428445114 | [-228.7956877613459, -0.5819641131774] |
| mass_cap_aux_vs_scaled_cap_aux/full/producer2_controller0 | -14.119826452399586 | [-29.064006359357002, 0.824353454557833] |
| mass_cap_aux_vs_scaled_cap_aux/full/producer2_controller1 | -14.383620378385636 | [-33.88008563182408, 2.0504228725991243] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only/producer0_controller1 | -150012.47555151978 | [-449861.103479384, -1.9089452072655362] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only/producer0_controller2 | -5994.445152581782 | [-15866.88616435961, -11.919830173003882] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only/producer1_controller0 | -23.25216341204252 | [-69.43873076428719, 0.6049318113455923] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only/producer1_controller2 | -581.8715873740941 | [-1712.7419417092433, -0.6129602872121208] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only/producer2_controller0 | -18.622317193065918 | [-39.16199452455398, 0.160434145401802] |
| mass_cap_aux_vs_scaled_cap_aux/motion_only/producer2_controller1 | -724.385360514866 | [-1910.324453997187, -2.9494804871996383] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full/producer0_controller1 | -24.30128112870222 | [-54.2899696060979, -0.71357536023434] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full/producer0_controller2 | -549.6983324612651 | [-1371.1083384562253, -12.681221042449199] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full/producer1_controller0 | 0.2443703755780651 | [-4.97565171635819, 7.246772570629361] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full/producer1_controller2 | -79.01584500413765 | [-160.19662122147497, -0.8385070921286262] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full/producer2_controller0 | -6.4100097757684305 | [-16.33359743353382, 0.6243134082185545] |
| mass_shuffled_aux_vs_raw_shuffled_aux/full/producer2_controller1 | -4.801369263637502 | [-9.697504024483319, 0.09476549720831542] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only/producer0_controller1 | -361.2288725155535 | [-945.8118550517504, -1.9383030860534531] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only/producer0_controller2 | -600.2920523628353 | [-1334.8929918638853, -6.29273996737401] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only/producer1_controller0 | 2.5704025429273605 | [-5.521110766633232, 13.586887588881552] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only/producer1_controller2 | -378.18649102143206 | [-1095.4605486890143, -1.09667126157346] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only/producer2_controller0 | 7.196500668472464 | [1.1617712613643527, 16.596398696267652] |
| mass_shuffled_aux_vs_raw_shuffled_aux/motion_only/producer2_controller1 | -134.01083387343382 | [-321.267490331082, -3.882096766925615] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full/producer0_controller1 | -26.613339689216527 | [-60.63853023373002, -0.7671433778432176] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full/producer0_controller2 | -2946.0724527270486 | [-8326.522509952643, -13.092201370156143] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full/producer1_controller0 | -7.5116565505471105 | [-17.778049304703448, -0.08769669183232764] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full/producer1_controller2 | -88.21097038402775 | [-182.54132045266567, -0.7211082937201945] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full/producer2_controller0 | -4.496228936483449 | [-17.38985062707626, 6.679244311422528] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/full/producer2_controller1 | -6.757161631420347 | [-21.385634794215314, 7.871311531374619] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only/producer0_controller1 | -89091.29683806901 | [-267105.2353782149, -2.009516437438936] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only/producer0_controller2 | -6041.039108903998 | [-15412.915013421472, -6.605163725026732] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only/producer1_controller0 | -18.326246804817185 | [-54.54421431319635, 0.46447254757945716] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only/producer1_controller2 | -674.8153889359925 | [-1983.3234518638328, -1.0964041313308557] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only/producer2_controller0 | -18.58748738818247 | [-42.891406952699526, 0.20825741203477627] |
| mass_shuffled_aux_vs_scaled_shuffled_aux/motion_only/producer2_controller1 | -893.2052648095377 | [-2538.4080640940842, -3.972812837376768] |
| mass_true_vs_mass_cost/full/producer0_controller1 | -2.212730136928987 | [-5.455861957170565, -0.10340738827132478] |
| mass_true_vs_mass_cost/full/producer0_controller2 | -4.313939887404816 | [-9.108716792707959, 0.3762570908770044] |
| mass_true_vs_mass_cost/full/producer1_controller0 | 1.390097189158866 | [0.48442144217875044, 2.295772936138982] |
| mass_true_vs_mass_cost/full/producer1_controller2 | -0.921322198679503 | [-8.053926294068365, 5.26443790449844] |
| mass_true_vs_mass_cost/full/producer2_controller0 | -11.460509812492042 | [-29.900977105703724, 1.6553647333112753] |
| mass_true_vs_mass_cost/full/producer2_controller1 | -14.102855436733302 | [-28.784190222782684, 0.5784793493160774] |
| mass_true_vs_mass_cost/motion_only/producer0_controller1 | -16.39825826037024 | [-33.0330669246955, 0.2365504039550097] |
| mass_true_vs_mass_cost/motion_only/producer0_controller2 | -8.42528137360215 | [-38.419461522611776, 14.32816538636468] |
| mass_true_vs_mass_cost/motion_only/producer1_controller0 | 0.20866352065661947 | [-0.5162314047968749, 0.8473850937105734] |
| mass_true_vs_mass_cost/motion_only/producer1_controller2 | -12.265760421944375 | [-37.553398040550896, 0.5680450890475398] |
| mass_true_vs_mass_cost/motion_only/producer2_controller0 | -2.1019882249971285 | [-4.347367768493869, -0.03652344500730565] |
| mass_true_vs_mass_cost/motion_only/producer2_controller1 | -31.339516816389107 | [-62.84024205425844, 0.16120842148021347] |
| mass_true_vs_mass_shuffled/full/producer0_controller1 | 0.7503601323397864 | [-2.9517449223116428, 5.598662386818227] |
| mass_true_vs_mass_shuffled/full/producer0_controller2 | -1.0445880728327013 | [-3.6353563111517166, 1.207620016032953] |
| mass_true_vs_mass_shuffled/full/producer1_controller0 | 0.6884594507641177 | [-0.7142603515190947, 2.09117925304733] |
| mass_true_vs_mass_shuffled/full/producer1_controller2 | -1.8214151936487455 | [-12.69882388392719, 7.064554059391136] |
| mass_true_vs_mass_shuffled/full/producer2_controller0 | -12.211877366639241 | [-30.262033054761122, -1.4415634576988885] |
| mass_true_vs_mass_shuffled/full/producer2_controller1 | -11.250815963744095 | [-22.889665053200183, 0.38803312571199294] |
| mass_true_vs_mass_shuffled/motion_only/producer0_controller1 | -10.651880693295531 | [-21.301973559195016, -0.0017878273960454105] |
| mass_true_vs_mass_shuffled/motion_only/producer0_controller2 | -1.0333893836495167 | [-12.484781244630033, 13.468084411052669] |
| mass_true_vs_mass_shuffled/motion_only/producer1_controller0 | -1.3625131228237612 | [-4.505790232275174, 0.3718681692537542] |
| mass_true_vs_mass_shuffled/motion_only/producer1_controller2 | -11.440598820428004 | [-39.66033621064856, 4.8763920822462055] |
| mass_true_vs_mass_shuffled/motion_only/producer2_controller0 | 0.016428347514178432 | [-2.8166291442164124, 2.916608017830559] |
| mass_true_vs_mass_shuffled/motion_only/producer2_controller1 | -31.980582418248314 | [-64.8616741396747, 0.9005093031780693] |

## Boundaries
Fitting moment equality is not conditional or held-scene calibration. Bound hits remain included.
Primary and all original guards remain unchanged;coverage gain cannot replace failed MSE.
Full/motion families have different forecast/event populations,not a matched feature ablation.
Obs8/pred12 native steps,detector pixels;no metric/seconds,human-gold,true3D or foundation claim.
```json
{
  "fitting_mass_constraints_supported": false,
  "primary_readout_gate": false,
  "guard_gate": false,
  "auxiliary_information_gate": false,
  "development_readout_screen": false,
  "independent_confirmation": false,
  "new_policy_evaluated": false,
  "deployment_changed": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```
