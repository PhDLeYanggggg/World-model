# Conditional Easy-Moment Results

Development only; equal-site relative ADE gain over CV. Three seeds, four exposed sites.
No metric/seconds, independent safety, confirmation or deployment claim. All fixed policies retained.

| Policy | ADE gain % [CI95] | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed, summed seeds |
|---|---:|---:|---:|---:|---:|
| damped_velocity_005__net_stop | 5.2812 [3.9559, 6.6064] | 8.2417 | 13.5232 | 26.893 | 3 |
| damped_velocity_005__strict_stop | 3.6347 [2.6423, 4.6860] | 4.7232 | 2.4944 | 12.252 | 0 |
| damped_velocity_005__joint_easy_moment | 0.1241 [0.0232, 0.2850] | 0.0004 | 0.0000 | 1.459 | 0 |
| damped_velocity_005__product_easy_marginals | 0.2118 [0.0417, 0.4933] | 0.0007 | 0.0000 | 2.045 | 0 |
| damped_velocity_005__matched_joint | 0.1241 [0.0232, 0.2850] | 0.0004 | 0.0000 | 1.459 | 0 |
| damped_velocity_005__matched_product | 0.1624 [0.0336, 0.3686] | 0.0007 | 0.0000 | 1.459 | 0 |
| transformer__net_stop | 8.3791 [6.7300, 9.9360] | 10.2513 | 21.7326 | 54.365 | 21 |
| transformer__strict_stop | 2.4368 [1.6792, 3.1114] | 2.2918 | 1.0669 | 6.676 | 0 |
| transformer__joint_easy_moment | 0.0134 [0.0023, 0.0314] | 0.0000 | 0.0000 | 0.232 | 0 |
| transformer__product_easy_marginals | 0.0266 [0.0118, 0.0414] | 0.0004 | 0.0000 | 0.499 | 0 |
| transformer__matched_joint | 0.0134 [0.0023, 0.0314] | 0.0000 | 0.0000 | 0.232 | 0 |
| transformer__matched_product | 0.0209 [0.0057, 0.0391] | 0.0002 | 0.0377 | 0.232 | 0 |
| eqmotion__net_stop | 11.0065 [8.8537, 13.1594] | 13.3300 | 40.9449 | 69.162 | 21 |
| eqmotion__strict_stop | 1.6093 [0.7163, 3.0393] | 0.5271 | 0.4522 | 3.926 | 0 |
| eqmotion__joint_easy_moment | 0.0013 [0.0005, 0.0024] | 0.0000 | 0.0000 | 0.010 | 0 |
| eqmotion__product_easy_marginals | 0.0390 [0.0076, 0.0835] | -0.0000 | 0.0000 | 0.311 | 0 |
| eqmotion__matched_joint | 0.0013 [0.0005, 0.0024] | 0.0000 | 0.0000 | 0.010 | 0 |
| eqmotion__matched_product | 0.0025 [0.0009, 0.0042] | 0.0000 | 0.0000 | 0.010 | 0 |

## Paired Contrasts

Nominal exploratory physical-site bootstrap intervals; no multiple-comparison claim.

| Contrast | ADE difference pp | CI95 |
|---|---:|---|
| damped_velocity_005__joint_easy_moment_minus_product_easy_marginals | -0.0878 | [-0.2095374062517974, -0.01848730795587783] |
| damped_velocity_005__joint_easy_moment_minus_strict_stop | -3.5106 | [-4.579963312435614, -2.601046287423639] |
| damped_velocity_005__matched_joint_minus_matched_product | -0.0383 | [-0.08375662155400976, -0.010462051769888348] |
| transformer__joint_easy_moment_minus_product_easy_marginals | -0.0132 | [-0.02101705921302155, -0.007510764453227958] |
| transformer__joint_easy_moment_minus_strict_stop | -2.4234 | [-3.089052461471015, -1.6686563308494744] |
| transformer__matched_joint_minus_matched_product | -0.0074 | [-0.011531825866006873, -0.0027913381224348166] |
| eqmotion__joint_easy_moment_minus_product_easy_marginals | -0.0377 | [-0.08268155190665205, -0.006600983250790171] |
| eqmotion__joint_easy_moment_minus_strict_stop | -1.6080 | [-3.0382664262285948, -0.7140890412631884] |
| eqmotion__matched_joint_minus_matched_product | -0.0011 | [-0.0016833598891180523, -0.000370997349860569] |
