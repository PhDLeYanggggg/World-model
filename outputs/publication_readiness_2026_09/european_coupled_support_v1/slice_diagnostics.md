# Coupled Risk and Donor Support: Fixed-Roster Readout

## Material Passport

72 exposed-TRAIN fitting views. Repeated queries, not independent observations.
Equal-view normalized signed-risk MSE, not trajectory ADE/FDE. Query unknownness retained.
No whole-pipeline OOF, held/independent role opening, policy change or safety certification.

| Arm | All MSE | Easy MSE | Predicted easy H | Observed easy H | EH>H | B+H>E |
|---|---:|---:|---:|---:|---:|---:|
| origin | 0.0794587451 | 0.000309373416 | 1.37560653 | 95.5606418 | 0 | 0 |
| independent_cost | 0.0896372541 | 0.000442399512 | 55.7726369 | 95.5606418 | 9019 | 0 |
| coupled_cost | 0.0896423016 | 0.00043565802 | 53.3493796 | 95.5606418 | 0 | 0 |
| record_mean_blend | 0.0787324997 | 0.000325424661 | 19.7742209 | 95.5606418 | 0 | 0 |
| shuffled_alpha_blend | 0.0834041079 | 0.000347115395 | 20.2005294 | 95.5606418 | 0 | 0 |
| nested_support_blend | 0.0805338585 | 0.000326961776 | 24.1909501 | 95.5606418 | 0 | 0 |

## Support and Blend

ESS measures weight concentration, not independent sample count.
Nested donor loss improvement is guaranteed by fitting alpha, not query generalization.

```json
{
  "local_supported": 62902,
  "local_unsupported": 0,
  "nested_supported": 59719,
  "alpha_zero": 27025,
  "alpha_one": 6942,
  "alpha_sum": 19055.521347210826,
  "nested_loss_before": 8693.82358361218,
  "nested_loss_after": 8147.906202513171
}
```

Old inconsistency versus low recording ESS:
```json
{
  "inconsistent_False_low_record_False": 36667,
  "inconsistent_False_low_record_True": 17216,
  "inconsistent_True_low_record_False": 4203,
  "inconsistent_True_low_record_True": 4816
}
```

Old inconsistency versus low row ESS:
```json
{
  "inconsistent_False_low_row_False": 49009,
  "inconsistent_False_low_row_True": 4874,
  "inconsistent_True_low_row_False": 6957,
  "inconsistent_True_low_row_True": 2062
}
```

## Prespecified Support Strata

| Stratum | Rows | Known | Original easy SSE/known | Coupled | Nested blend |
|---|---:|---:|---:|---:|---:|
| one_reference_record | 3183 | 3110 | 2.32709184e-05 | 2.43735487e-05 | 2.32709184e-05 |
| multiple_reference_records | 59719 | 58225 | 0.000719940747 | 0.000751617909 | 0.000726991524 |
| record_ESS_below2 | 22032 | 21477 | 0.000145137158 | 0.000184506894 | 0.000158937196 |
| record_ESS_atleast2 | 40870 | 39858 | 0.000975307638 | 0.00100045411 | 0.000978171506 |
| alpha_zero | 27025 | 26367 | 0.000318246072 | 0.000339562256 | 0.000318246072 |
| alpha_positive | 35877 | 34968 | 0.00096087075 | 0.000997641172 | 0.000972610955 |

Stratum MSE pools normalized SSE/known occurrences; it is NOT the equal-view primary metric.
Zero-alpha and one-record strata do not establish calibrated safety.

## All Source, Context and Head-Seed Slices

| Slice | Views | Original easy MSE | Independent | Coupled | Nested blend |
|---|---:|---:|---:|---:|---:|
| source:eu-locality-007 | 6 | 0.0014372615 | 0.00143906113 | 0.00143902435 | 0.00143785433 |
| source:eu-locality-008 | 6 | 0.000389529934 | 0.000403142928 | 0.000403433526 | 0.000393829201 |
| source:eu-locality-020 | 6 | 9.28695137e-06 | 0.00125858091 | 0.00117946789 | 0.000114915159 |
| source:eu-locality-048 | 6 | 0.00016394108 | 0.000180654301 | 0.000180001445 | 0.000169842473 |
| source:eu-locality-067 | 6 | 1.16620396e-05 | 2.57903591e-05 | 2.56778689e-05 | 1.48175258e-05 |
| source:eu-locality-074 | 6 | 0.00166629258 | 0.00168956904 | 0.00168894685 | 0.00167121987 |
| source:eu-locality-082 | 6 | 3.98978762e-08 | 8.14090484e-05 | 8.14053369e-05 | 5.98471141e-05 |
| source:eu-locality-110 | 6 | 7.63244503e-06 | 8.66379163e-06 | 8.66628351e-06 | 7.66756224e-06 |
| source:eu-locality-112 | 6 | 2.46679873e-07 | 0.000113394767 | 0.000113394633 | 4.63706964e-06 |
| source:eu-locality-119 | 6 | 9.75146133e-06 | 9.85852953e-06 | 9.85851703e-06 | 9.76898134e-06 |
| source:eu-locality-124 | 6 | 2.23731743e-07 | 1.50976037e-06 | 1.51040999e-06 | 7.38027796e-07 |
| source:eu-locality-126 | 6 | 1.66126888e-05 | 9.71595726e-05 | 9.65091258e-05 | 3.84040055e-05 |
| context:single0_seed43_controller1_dimensionless | 12 | 0.000175060151 | 0.000818034187 | 0.000778309438 | 0.000233500332 |
| context:single0_seed43_controller2_dimensionless | 12 | 0.000128487725 | 0.00012864698 | 0.000128633727 | 0.000128499721 |
| context:single1_seed43_controller0_dimensionless | 12 | 0.000112149852 | 0.000116050063 | 0.000115980927 | 0.000113201847 |
| context:single1_seed43_controller2_dimensionless | 12 | 0.000196039928 | 0.000207874486 | 0.000207803428 | 0.000198061969 |
| context:single2_seed43_controller0_dimensionless | 12 | 0.000598946844 | 0.000600899626 | 0.000600896055 | 0.000599514732 |
| context:single2_seed43_controller1_dimensionless | 12 | 0.000645555997 | 0.00078289173 | 0.000782324547 | 0.000688992058 |
| seed:17 | 24 | 0.000313074292 | 0.000352121175 | 0.000352030247 | 0.000334150578 |
| seed:29 | 24 | 0.000337231259 | 0.000417043302 | 0.000416676619 | 0.000348080813 |
| seed:43 | 24 | 0.000277814698 | 0.000558034058 | 0.000538267194 | 0.000298653938 |

No posthoc primary substitution or threshold search. No confirmatory CI/p-value from fitting repeats.
Head seeds are not new forecaster seeds. obs8/pred12 rawstride12, pixel/local detector-silver.
No metric/seconds/human-gold/physical-safety/true3D/foundation claim. Stage5C/SMC remain off.
