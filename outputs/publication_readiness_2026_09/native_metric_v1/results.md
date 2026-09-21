# Native-Coordinate Source Readout

## Material Passport

Fresh paired score computation; cached/hash-verified inputs and predictions.
Post-hoc evaluation amendment, four previously explored SDD training sites.
No training, new inference, independent test, threshold search or deployment.
SDD annotation pixels, 8 observed / 12 predicted steps, K=1; no meters/seconds.

| Cohort | Method | Equal-scene ADE gain vs CV (%) | Scene bootstrap 95% interval | Worst scene (%) |
| --- | --- | ---: | --- | ---: |
| supported_masked | constant_position | -213.0834988861967 | [-266.1404600815545, -146.44601871369858] | -281.9171114750332 |
| supported_masked | constant_velocity_causal_fd | 0.0 | [0.0, 0.0] | 0.0 |
| supported_masked | damped_velocity_005 | -6.621048011964839 | [-13.186851990527243, 0.14150295326141582] | -15.703277519140668 |
| supported_masked | damped_velocity_010 | -28.415998309604454 | [-42.82468041743959, -12.590687224396035] | -48.07073414489662 |
| supported_masked | damped_velocity_020 | -67.83098802031174 | [-92.61547587270324, -38.86044503757113] | -101.14409430898847 |
| supported_masked | constant_acceleration_causal | -300.25000459275805 | [-316.8168528508478, -283.94206509923447] | -323.23948094768343 |
| supported_masked | constant_turn_rate | -74.57593488953091 | [-83.97356292313117, -59.57696001238557] | -84.60263132017035 |
| supported_masked | oracle_diagnostic_not_deployable | 28.633381261346617 | [25.942826970366177, 32.55751995570691] | 25.414645025769754 |
| supported_masked | complement_selected_baseline | 0.0 | [0.0, 0.0] | 0.0 |
| complete_future | constant_position | -213.1337272595207 | [-267.02224406495776, -143.80504358839474] | -281.3325300303507 |
| complete_future | constant_velocity_causal_fd | 0.0 | [0.0, 0.0] | 0.0 |
| complete_future | damped_velocity_005 | -7.63649859598369 | [-14.966532749786065, -0.1425735803947087] | -17.637183160227575 |
| complete_future | damped_velocity_010 | -30.926841497731512 | [-46.56750362080751, -13.692023514500557] | -51.8184576436171 |
| complete_future | damped_velocity_020 | -72.09068915360207 | [-98.40864352264265, -40.95410962813942] | -106.53361248246145 |
| complete_future | constant_acceleration_causal | -319.26258846337595 | [-332.26819121899877, -305.2310121578712] | -336.8730947115841 |
| complete_future | constant_turn_rate | -78.35913181646694 | [-89.01463707305767, -62.17173651057679] | -90.7453624064941 |
| complete_future | oracle_diagnostic_not_deployable | 28.354496751111547 | [25.677321914129468, 32.644133119845286] | 25.17800556399482 |
| complete_future | complement_selected_baseline | 0.0 | [0.0, 0.0] | 0.0 |

All scene-native ADE/FDE, tail errors and unchanged old metrics are in analysis.json.
Oracle uses future ADE labels and is not an executable policy. Its FDE uses the same ADE-selected candidate.
Complement baseline is selected only on other source sites, not this site.

## Existing Neural Predictions: Different, Explicit Subset

All twelve old source-crossfit fits / three seeds are retained. Their cached
predictions cover 15,430 static-history windows, not the full source population.
Future labels are used for scoring only; no excluded labels were opened.

| Seed | Native equal-scene ADE gain (%) | Old normalized gain (%) |
| --- | ---: | ---: |
| 17 | -5.4579106814754255 | -5.083089784708927 |
| 29 | -6.137417327029998 | -5.709360467433799 |
| 43 | -4.549613095502781 | -4.2554896924606345 |

Zero-CV easy rows report absolute pixel harm; percentage safety is undefined, not passed.
No best seed or baseline is promoted from this readout. The four-scene bootstrap
describes reused source sites and cannot establish independent generalization.
The new score does not fix old negative probes, lineage concerns or calibration.
A matched full-population neural experiment and native-scale training/risk registration
are still required. Original validation/test, main/external and bookstore readouts stay closed.
Stage5C/SMC disabled. The research goal and submission readiness remain unmet.

Reproduce: `.venv-pytorch/bin/python scripts/rescore_m3w_native_metric.py --verify`.
