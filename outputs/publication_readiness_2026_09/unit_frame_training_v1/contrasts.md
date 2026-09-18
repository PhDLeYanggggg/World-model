# Conditioning Contrasts and Failure Slices

Descriptive results from all fixed exposed-site folds; no selection or deployment.

| Arm | Gain vs CV | Descriptive site CI | Per-seed gains | Easy degradation range |
| --- | ---: | --- | --- | --- |
| legacy_sdd_aux | -0.805231% | [-7.225310648290861, -0.2256037499129393] | [-0.8926496261201011, -0.6816656889689687, -0.8413781478963545] | [325.30451344776657, 7255.272904919146] |
| unit_inputs_only | -0.687296% | [-9.451591753581855, 0.23118018322064726] | [-0.6453150679881947, -0.705973235208246, -0.7105995341827764] | [297.37509922311756, 9151.29451092046] |
| unit_primary_log | -2.489283% | [-5.519958199582575, -0.07793225261976122] | [-2.1619123336481083, -3.035331158348731, -2.270606461393565] | [445.05861876433556, 80272.53375791654] |
| unit_internal_log | -185.777151% | [-408.6385225638169, -18.870105994759466] | [-242.67445064355044, -149.79440662141803, -164.86259661130077] | [60661.03259321776, 5389260.104878157] |

Three repeatedly exposed site clusters are not independent confirmation. Easy percentages can be very large when CV error is near zero; absolute harm is retained below.

| Arm | Absolute easy harm range | Train gain range (%) | Static-start gain (%) | Static-stay harm | Hard gain (%) |
| --- | --- | --- | ---: | ---: | ---: |
| legacy_sdd_aux | [0.03504575875958233, 0.18984748733608844] | [0.5687008999296195, 1.5567874088501243] | -0.004044 | 0.031639 | -0.063211 |
| unit_inputs_only | [0.017985426419215265, 0.23946036097316847] | [0.8358913523255573, 1.8159338679010761] | -0.005090 | 0.012604 | 0.060100 |
| unit_primary_log | [0.05914610343660627, 1.4424429964745362] | [-1.6476760615654618, 0.4345237122768908] | -0.266013 | 1.011297 | -0.435314 |
| unit_internal_log | [8.06155314617022, 96.84134946960579] | [-89.47839795545833, 24.356983404783506] | -31.702941 | 77.712232 | -45.005825 |

Legacy gain -0.80523115%; adding only the no-anchor CV guard gives -0.80027354%.
This guard counterfactual is post hoc attribution, not a selected policy.
Native-coordinate aggregate metrics mix local units and are diagnostic only; the primary normalized metric is unchanged.
