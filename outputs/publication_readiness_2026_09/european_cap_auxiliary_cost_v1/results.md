# Cap-Event Auxiliary Cost Results

## Material Passport
fresh_run:432 native-Torch heads,864000 fixed updates and144 source-held readouts.
cached_verified:source forecasters, nested risk producers and original cost estimators.
not_run:new trajectory training, policy evaluation, independent selection/calibration/confirmation.

| Contrast / inputs | Easy-harm MSE: positive / negative / overlap / missing | Point range (%) |
|---|---|---|
| aux_vs_control/full | [2, 0, 4, 0] | [-5.857642571071608, 14.351097127243193] |
| aux_vs_control/motion_only | [0, 1, 5, 0] | [-12.205489122480659, 2.5344104278319612] |
| aux_vs_original/full | [0, 3, 3, 0] | [-37.74034180743142, 0.6799118628409389] |
| aux_vs_original/motion_only | [1, 1, 4, 0] | [-273.85984543932193, 7.193688211786311] |
| aux_vs_shuffled/full | [1, 1, 4, 0] | [-13.584296866430213, 14.272128272802256] |
| aux_vs_shuffled/motion_only | [0, 0, 6, 0] | [-57.70619346071839, 1.9885288131013499] |
| control_vs_original/full | [0, 2, 4, 0] | [-120.87541031501965, 1.163802793806388] |
| control_vs_original/motion_only | [1, 0, 5, 0] | [-251.34341098887558, 10.999529320487738] |
| shuffled_vs_control/full | [1, 1, 4, 0] | [-4.618275167077655, 9.243176492303016] |
| shuffled_vs_control/motion_only | [0, 1, 5, 0] | [-1.2635850498888126, 9.777536671615268] |

## All Assignment Intervals

| Contrast / inputs / assignment | Point (%) | 95% locality CI |
|---|---:|---|
| aux_vs_control/full/producer0_controller1 | -0.16854594770546658 | [-1.5081225476879156, 0.9001666545763158] |
| aux_vs_control/full/producer0_controller2 | -4.220133719536934 | [-18.910009266017, 9.399678834560122] |
| aux_vs_control/full/producer1_controller0 | 2.1855228805254745 | [0.35141286730672433, 4.019632893744225] |
| aux_vs_control/full/producer1_controller2 | -0.5375882571646886 | [-1.7846293007143261, 0.16691969642640742] |
| aux_vs_control/full/producer2_controller0 | 14.351097127243193 | [3.6911777172614144, 31.62251474067976] |
| aux_vs_control/full/producer2_controller1 | -5.857642571071608 | [-19.414447563555008, 1.6604843399884992] |
| aux_vs_control/motion_only/producer0_controller1 | -12.205489122480659 | [-30.761345272225206, 0.01580014470481323] |
| aux_vs_control/motion_only/producer0_controller2 | -7.432493628301613 | [-22.353926059595278, 1.585023095915722] |
| aux_vs_control/motion_only/producer1_controller0 | 2.5344104278319612 | [-1.000366042615859, 8.617508832906665] |
| aux_vs_control/motion_only/producer1_controller2 | -6.285050574708732 | [-15.485854315884463, 0.015701753903968756] |
| aux_vs_control/motion_only/producer2_controller0 | -1.8712407961274624 | [-3.761612946440805, -0.42951856234704056] |
| aux_vs_control/motion_only/producer2_controller1 | -9.189302623842627 | [-36.99407856927197, 9.550674796820287] |
| aux_vs_original/full/producer0_controller1 | -0.231612682171851 | [-1.4863437408067133, 1.0692948774973616] |
| aux_vs_original/full/producer0_controller2 | -19.722391996248728 | [-33.555053039787325, -5.889730952710129] |
| aux_vs_original/full/producer1_controller0 | 0.5061899175196698 | [-3.2978275899728953, 5.58266860767632] |
| aux_vs_original/full/producer1_controller2 | 0.6799118628409389 | [-0.16330821225490466, 2.10216201676577] |
| aux_vs_original/full/producer2_controller0 | -4.774062054135726 | [-10.197670371659477, -0.8856192295502836] |
| aux_vs_original/full/producer2_controller1 | -37.74034180743142 | [-104.40165135607523, -3.541904131892455] |
| aux_vs_original/motion_only/producer0_controller1 | 3.372373394911939 | [-1.214230039942576, 11.264625528458623] |
| aux_vs_original/motion_only/producer0_controller2 | 7.193688211786311 | [0.055811015533393946, 14.331565408039229] |
| aux_vs_original/motion_only/producer1_controller0 | 4.129503933740978 | [-1.4322850997130048, 9.69129296719496] |
| aux_vs_original/motion_only/producer1_controller2 | -273.85984543932193 | [-818.7912352701123, -0.07606897971327571] |
| aux_vs_original/motion_only/producer2_controller0 | -6.871801315791138 | [-14.974580423796876, 0.1728961579354602] |
| aux_vs_original/motion_only/producer2_controller1 | -12.426403439804561 | [-76.12312713661385, 39.02702348188947] |
| aux_vs_shuffled/full/producer0_controller1 | -0.3022372514869685 | [-1.5122075307702678, 0.675457439339683] |
| aux_vs_shuffled/full/producer0_controller2 | -10.320103139936606 | [-25.37803217694514, -1.4047047787569158] |
| aux_vs_shuffled/full/producer1_controller0 | 0.46383465962092746 | [-1.1122443567500566, 2.1069147254441467] |
| aux_vs_shuffled/full/producer1_controller2 | 0.17821096650439777 | [-0.04596734467486562, 0.37083897898643864] |
| aux_vs_shuffled/full/producer2_controller0 | 14.272128272802256 | [5.854743970517355, 28.21212228794166] |
| aux_vs_shuffled/full/producer2_controller1 | -13.584296866430213 | [-41.32149369200773, 1.4136411369419761] |
| aux_vs_shuffled/motion_only/producer0_controller1 | -15.885460954169611 | [-42.546075905903564, 0.021100133612553397] |
| aux_vs_shuffled/motion_only/producer0_controller2 | -16.832912840607236 | [-33.937368300073096, 0.2715426188586241] |
| aux_vs_shuffled/motion_only/producer1_controller0 | 1.9885288131013499 | [-2.6148516542634206, 9.2329445372865] |
| aux_vs_shuffled/motion_only/producer1_controller2 | -57.70619346071839 | [-170.84250040558766, 0.07650311774931184] |
| aux_vs_shuffled/motion_only/producer2_controller0 | -0.5589521623686913 | [-2.466355804148212, 1.0759897860334942] |
| aux_vs_shuffled/motion_only/producer2_controller1 | -8.441678516673326 | [-35.104736452824206, 9.830276294749183] |
| control_vs_original/full/producer0_controller1 | -0.06753466239772801 | [-0.45015113000467877, 0.28115322610249605] |
| control_vs_original/full/producer0_controller2 | -17.130444889214168 | [-54.67201979010995, 6.10994702813769] |
| control_vs_original/full/producer1_controller0 | -1.725882679862366 | [-6.677146606291606, 2.629434731178251] |
| control_vs_original/full/producer1_controller2 | 1.163802793806388 | [-0.24013017330247155, 2.567735760915247] |
| control_vs_original/full/producer2_controller0 | -120.87541031501965 | [-345.77224316570357, -5.995725417590302] |
| control_vs_original/full/producer2_controller1 | -27.35779443433082 | [-71.2271457146477, -5.242761322975938] |
| control_vs_original/motion_only/producer0_controller1 | 10.999529320487738 | [0.0507582965194224, 28.486813881273214] |
| control_vs_original/motion_only/producer0_controller2 | 0.5493860473107195 | [-7.1222011296631145, 10.412672765333301] |
| control_vs_original/motion_only/producer1_controller0 | 1.73752233320661 | [-0.5587615983335825, 5.171751002965146] |
| control_vs_original/motion_only/producer1_controller2 | -251.34341098887558 | [-754.5226554271271, 0.5842787621612316] |
| control_vs_original/motion_only/producer2_controller0 | -4.815475951408885 | [-13.828425600041829, 0.5948734671975222] |
| control_vs_original/motion_only/producer2_controller1 | 0.8145569606312133 | [-21.4881057983195, 23.98961821680946] |
| shuffled_vs_control/full/producer0_controller1 | 0.13510622485849216 | [0.002488239300188705, 0.2677242104167956] |
| shuffled_vs_control/full/producer0_controller2 | 9.243176492303016 | [-0.1291551931433105, 18.61550817774934] |
| shuffled_vs_control/full/producer1_controller0 | 1.2926354466183094 | [-0.06831897578183328, 3.6206734439711514] |
| shuffled_vs_control/full/producer1_controller2 | -0.7638527480121657 | [-2.2984895660265807, 0.05044030213456753] |
| shuffled_vs_control/full/producer2_controller0 | -4.618275167077655 | [-10.698251670637543, -0.8553964840437096] |
| shuffled_vs_control/full/producer2_controller1 | 4.73785196142966 | [-0.06255863075237172, 13.234231178841629] |
| shuffled_vs_control/motion_only/producer0_controller1 | 1.905068293779956 | [-0.6438930722429429, 6.364400809616499] |
| shuffled_vs_control/motion_only/producer0_controller2 | 2.7228231105996787 | [-7.2163360092207265, 14.078032678950763] |
| shuffled_vs_control/motion_only/producer1_controller0 | 0.416671527659274 | [-1.068251471824092, 2.5319132031341147] |
| shuffled_vs_control/motion_only/producer1_controller2 | 9.777536671615268 | [-0.9983379042360507, 30.39206581635273] |
| shuffled_vs_control/motion_only/producer2_controller0 | -1.1735135473644944 | [-2.2762885503820476, -0.1736920476197237] |
| shuffled_vs_control/motion_only/producer2_controller1 | -1.2635850498888126 | [-4.042424675286485, 0.3255936405937415] |

Three seeds averaged within locality, then3000 paired resamples of four localities.
Six assignments overlap; source development has prior exposure. Intervals are exploratory, not multiplicity-adjusted.
Full/motion changes forecasts and event populations; not a matched feature ablation. Unknown support is retained.
Primary requires positive full-input cost intervals against control and original, with tail/coverage/all-harm guards.
Task-information additionally requires improvement against locality-shuffled auxiliary. Event classification alone cannot pass.
New matched controls have399 inputs and two learned harm costs; the frozen original is a stronger historical comparator, not an identical control.
Event probability never multiplies expected harm. The zero-reference guard is unchanged. No policy lift claimed.

```json
{
  "primary_cost_gate": false,
  "tail_coverage_all_harm_guards": false,
  "true_vs_shuffled_gate": false,
  "auxiliary_cost_contribution": false,
  "new_policy_evaluated": false,
  "independent_confirmation": false,
  "deployment_changed": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Obs8/pred12 native annotation steps, detector pixels; no metric/seconds, human-gold, physical-safety, true3D or foundation claim.
