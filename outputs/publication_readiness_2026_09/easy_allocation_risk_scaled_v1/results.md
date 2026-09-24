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
| damped_velocity_005__aggregate_population | 0.9213 [0.4882, 1.5333] | 0.6229 | 0.0000 | 6.028 | 0 |
| damped_velocity_005__scene_uniform | 0.0006 [0.0000, 0.0012] | 0.0000 | 0.0000 | 0.014 | 0 |
| damped_velocity_005__aggregate_unary | 0.9212 [0.4879, 1.5334] | 0.6227 | 0.0000 | 6.028 | 0 |
| damped_velocity_005__aggregate_joint | 0.9212 [0.4879, 1.5332] | 0.6227 | 0.0000 | 6.028 | 0 |
| transformer__floor | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| transformer__net_stop | 8.3791 [6.7300, 9.9360] | 10.2513 | 21.7326 | 54.365 | 21 |
| transformer__strict_stop | 2.4368 [1.6792, 3.1114] | 2.2918 | 1.0669 | 6.676 | 0 |
| transformer__pointwise | 0.0134 [0.0023, 0.0314] | 0.0000 | 0.0000 | 0.232 | 0 |
| transformer__aggregate_selected | 0.0170 [0.0027, 0.0411] | 0.0001 | 0.0000 | 0.260 | 0 |
| transformer__aggregate_population | 1.2802 [0.5647, 2.1160] | 0.6425 | 0.1120 | 10.759 | 2 |
| transformer__scene_uniform | 0.0001 [0.0000, 0.0002] | 0.0000 | 0.0000 | 0.007 | 0 |
| transformer__aggregate_unary | 1.2781 [0.5628, 2.1142] | 0.6412 | 0.0954 | 10.759 | 2 |
| transformer__aggregate_joint | 1.2781 [0.5628, 2.1142] | 0.6412 | 0.0954 | 10.759 | 2 |
| eqmotion__floor | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| eqmotion__net_stop | 11.0065 [8.8537, 13.1594] | 13.3300 | 40.9449 | 69.162 | 21 |
| eqmotion__strict_stop | 1.6093 [0.7163, 3.0393] | 0.5271 | 0.4522 | 3.926 | 0 |
| eqmotion__pointwise | 0.0013 [0.0005, 0.0024] | 0.0000 | 0.0000 | 0.010 | 0 |
| eqmotion__aggregate_selected | 0.0013 [0.0005, 0.0024] | 0.0000 | 0.0000 | 0.010 | 0 |
| eqmotion__aggregate_population | 0.9556 [0.3536, 1.6291] | 0.2504 | 0.0381 | 4.423 | 0 |
| eqmotion__scene_uniform | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 | 0 |
| eqmotion__aggregate_unary | 0.9551 [0.3526, 1.6285] | 0.2503 | 0.0381 | 4.423 | 0 |
| eqmotion__aggregate_joint | 0.9550 [0.3525, 1.6285] | 0.2503 | 0.0381 | 4.423 | 0 |

## Paired Contrasts

Nominal development contrasts, no multiple-comparison or population claim.

| Predictor / contrast | Difference pp | CI95 pp |
|---|---:|---|
| damped_velocity_005 / aggregate_selected_minus_pointwise | 0.067206 | [0.004424742767727974, 0.18660101193498368] |
| damped_velocity_005 / aggregate_population_minus_aggregate_selected | 0.730037 | [0.4418799904213455, 1.0643488267890466] |
| damped_velocity_005 / aggregate_population_minus_strict_stop | -2.713387 | [-3.850341040203464, -1.7454694316377695] |
| damped_velocity_005 / aggregate_joint_minus_aggregate_unary | -0.000058 | [-0.00016798890264679844, 0.0] |
| damped_velocity_005 / aggregate_joint_minus_aggregate_population | -0.000122 | [-0.00036305328953989413, 0.0] |
| transformer / aggregate_selected_minus_pointwise | 0.003562 | [0.0002889554536533723, 0.00949517119901433] |
| transformer / aggregate_population_minus_aggregate_selected | 1.263170 | [0.5620403583042693, 2.0749322467147584] |
| transformer / aggregate_population_minus_strict_stop | -1.156671 | [-2.0064407704125626, -0.5222707167591795] |
| transformer / aggregate_joint_minus_aggregate_unary | 0.000048 | [0.0, 9.622009615184446e-05] |
| transformer / aggregate_joint_minus_aggregate_population | -0.002036 | [-0.0034265751821721535, -0.0008405985197135202] |
| eqmotion / aggregate_selected_minus_pointwise | 0.000000 | [0.0, 0.0] |
| eqmotion / aggregate_population_minus_aggregate_selected | 0.954220 | [0.3514270613385606, 1.6284507069080634] |
| eqmotion / aggregate_population_minus_strict_stop | -0.653745 | [-1.386093833941368, -0.13979503880879873] |
| eqmotion / aggregate_joint_minus_aggregate_unary | -0.000061 | [-0.00012273164432530237, 0.0] |
| eqmotion / aggregate_joint_minus_aggregate_population | -0.000530 | [-0.0016860285403236253, 9.516858635316705e-05] |

## Query Geometry and Numerical Support

| Predictor | Query/seed instances | Unmatched | Selected solver failures | Population / unary / joint failures | Nonadditive queries | Changed joint/unary queries |
|---|---:|---:|---:|---|---:|---:|
| damped_velocity_005 | 62796 | 0 | 0 | 0 / 0 / 0 | 150 | 2 |
| transformer | 62796 | 0 | 0 | 0 / 0 / 0 | 1253 | 20 |
| eqmotion | 62796 | 0 | 0 | 0 / 0 / 0 | 387 | 3 |
