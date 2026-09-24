# Fixed Query Allocation Results

Development-only, native annotation pixels, obs8/pred12. Seed means are means of errors, not an ensemble.
CI95: 3,000 paired physical-site resamples, only four exposed sites. Not independent confirmation.
Zero-CV/unknown/incomplete counts sum query/seed instances. Every policy is retained.

| Policy | ADE gain % [CI95] | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed |
|---|---:|---:|---:|---:|---:|
| damped_velocity_005__floor | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| damped_velocity_005__net_stop | 5.2812 [3.9559, 6.6064] | 8.2417 | 13.5232 | 26.893 | 3 |
| damped_velocity_005__strict_stop | 3.6347 [2.6423, 4.6860] | 4.7232 | 2.4944 | 12.252 | 0 |
| damped_velocity_005__pointwise | 0.1241 [0.0232, 0.2850] | 0.0004 | 0.0000 | 1.459 | 0 |
| damped_velocity_005__aggregate_selected | 0.1913 [0.0276, 0.4711] | 0.0022 | 0.0000 | 1.806 | 0 |
| damped_velocity_005__aggregate_population | 0.9204 [0.4873, 1.5329] | 0.6221 | 0.0000 | 6.021 | 0 |
| damped_velocity_005__scene_uniform | 0.0006 [0.0000, 0.0012] | 0.0000 | 0.0000 | 0.014 | 0 |
| damped_velocity_005__aggregate_unary | 0.9199 [0.4871, 1.5318] | 0.6217 | 0.0000 | 6.019 | 0 |
| damped_velocity_005__aggregate_joint | 0.9198 [0.4871, 1.5317] | 0.6217 | 0.0000 | 6.019 | 0 |
| transformer__floor | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| transformer__net_stop | 8.3791 [6.7300, 9.9360] | 10.2513 | 21.7326 | 54.365 | 21 |
| transformer__strict_stop | 2.4368 [1.6792, 3.1114] | 2.2918 | 1.0669 | 6.676 | 0 |
| transformer__pointwise | 0.0134 [0.0023, 0.0314] | 0.0000 | 0.0000 | 0.232 | 0 |
| transformer__aggregate_selected | 0.0170 [0.0027, 0.0411] | 0.0001 | 0.0000 | 0.260 | 0 |
| transformer__aggregate_population | 1.2756 [0.5632, 2.1118] | 0.6388 | 0.1154 | 10.721 | 2 |
| transformer__scene_uniform | 0.0001 [0.0000, 0.0002] | 0.0000 | 0.0000 | 0.007 | 0 |
| transformer__aggregate_unary | 1.2727 [0.5611, 2.1087] | 0.6371 | 0.0941 | 10.714 | 2 |
| transformer__aggregate_joint | 1.2727 [0.5611, 2.1087] | 0.6371 | 0.0941 | 10.713 | 2 |
| eqmotion__floor | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| eqmotion__net_stop | 11.0065 [8.8537, 13.1594] | 13.3300 | 40.9449 | 69.162 | 21 |
| eqmotion__strict_stop | 1.6093 [0.7163, 3.0393] | 0.5271 | 0.4522 | 3.926 | 0 |
| eqmotion__pointwise | 0.0013 [0.0005, 0.0024] | 0.0000 | 0.0000 | 0.010 | 0 |
| eqmotion__aggregate_selected | 0.0013 [0.0005, 0.0024] | 0.0000 | 0.0000 | 0.010 | 0 |
| eqmotion__aggregate_population | 0.9551 [0.3536, 1.6287] | 0.2504 | 0.0381 | 4.421 | 0 |
| eqmotion__scene_uniform | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| eqmotion__aggregate_unary | 0.9540 [0.3525, 1.6268] | 0.2503 | 0.0381 | 4.418 | 0 |
| eqmotion__aggregate_joint | 0.9539 [0.3524, 1.6268] | 0.2503 | 0.0381 | 4.418 | 0 |

## Paired Contrasts

Nominal development contrasts, no multiple-comparison or population claim.

| Predictor / contrast | Difference pp | CI95 pp |
|---|---:|---|
| damped_velocity_005 / aggregate_selected_minus_pointwise | 0.067206 | [0.004424742767727974, 0.18660101193498368] |
| damped_velocity_005 / aggregate_population_minus_aggregate_selected | 0.729172 | [0.4414639496952838, 1.0638993510877355] |
| damped_velocity_005 / aggregate_population_minus_strict_stop | -2.714253 | [-3.8516894673073976, -1.7454694316377695] |
| damped_velocity_005 / aggregate_joint_minus_aggregate_unary | -0.000058 | [-0.0001679889026301451, 0.0] |
| damped_velocity_005 / aggregate_joint_minus_aggregate_population | -0.000617 | [-0.0011984118742230931, -0.00014453847407558396] |
| transformer / aggregate_selected_minus_pointwise | 0.003562 | [0.0002889554536533723, 0.00949517119901433] |
| transformer / aggregate_population_minus_aggregate_selected | 1.258619 | [0.5605213862356601, 2.070676076259822] |
| transformer / aggregate_population_minus_strict_stop | -1.161222 | [-2.0160864309800077, -0.5238179910478369] |
| transformer / aggregate_joint_minus_aggregate_unary | 0.000009 | [-4.922737218693296e-05, 7.4899711782761e-05] |
| transformer / aggregate_joint_minus_aggregate_population | -0.002868 | [-0.004344696847707885, -0.0013913023057643414] |
| eqmotion / aggregate_selected_minus_pointwise | 0.000000 | [0.0, 0.0] |
| eqmotion / aggregate_population_minus_aggregate_selected | 0.953788 | [0.3514390042888005, 1.6280580176961212] |
| eqmotion / aggregate_population_minus_strict_stop | -0.654177 | [-1.3864865231533101, -0.14039332383455339] |
| eqmotion / aggregate_joint_minus_aggregate_unary | -0.000061 | [-0.00012273164433085348, 0.0] |
| eqmotion / aggregate_joint_minus_aggregate_population | -0.001204 | [-0.0020271295492702013, -0.00038013518236512667] |

## Query Geometry and Numerical Support

| Predictor | Query/seed instances | Unmatched | Selected solver failures | Population / unary / joint failures | Nonadditive queries | Changed joint/unary queries |
|---|---:|---:|---:|---|---:|---:|
| damped_velocity_005 | 62796 | 17 | 0 | 15 / 17 / 17 | 149 | 2 |
| transformer | 62796 | 93 | 0 | 77 / 91 / 93 | 1245 | 21 |
| eqmotion | 62796 | 17 | 0 | 10 / 17 / 17 | 387 | 3 |
