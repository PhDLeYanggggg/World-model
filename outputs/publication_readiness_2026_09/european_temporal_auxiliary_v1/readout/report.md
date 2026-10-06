# Temporal Auxiliary: Complete Development Readout

Result source: `fresh_run` fixed-final development predictions, scalar checks
and 3000 paired locality bootstrap draws. Models and original controls are
`cached_verified`. All 216 heads completed 2000 updates before this readout.
There are 72 source/head-seed views over 12 exposed localities, not 72
independent scenes. Head seeds are not independently retrained forecasters.

## Registered Decision

Advance to transfer design: **false**.
Deployment remains unchanged. Independent calibration and confirmation remain closed.
No validation threshold, checkpoint, architecture or comparator selection was performed.

## Primary Cost and Paired Utility

Each entry is temporal minus comparator, mean [nominal 95% CI]. Negative
signed-score MSE is better; positive paired-completion utility is better.
Utility is percent of full known reference cost, not raw FDE improvement.
These are exposed-development, non-multiplicity-adjusted intervals. Missing
support stays undefined; no head is dropped to manufacture a CI.

| Comparator | All MSE | Original-selected MSE | Full paired lower utility | Matched paired lower utility |
|---|---|---|---|---|
| original | +0.261571 [+0.104999, +0.451020] | undefined (undefined_support_no_dropping) | +1.390725 [+0.566962, +2.193647] | +0.261338 [+0.034097, +0.566836] |
| additive | +0.293717 [+0.126013, +0.504089] | undefined (undefined_support_no_dropping) | +1.053742 [+0.239029, +1.861736] | +0.220231 [-0.024922, +0.521718] |
| poisson | +0.125410 [-0.070909, +0.319224] | undefined (undefined_support_no_dropping) | +1.345108 [+0.527166, +2.147304] | +0.270922 [+0.023377, +0.589251] |
| cost | +0.157417 [-0.056953, +0.367987] | undefined (undefined_support_no_dropping) | +1.384831 [+0.561310, +2.187567] | +0.257274 [+0.017003, +0.570425] |
| none | -0.062288 [-0.151167, +0.005691] | undefined (undefined_support_no_dropping) | -1.026746 [-1.557310, -0.574154] | -0.567678 [-0.939022, -0.286494] |
| rowmean | -0.024422 [-0.054556, -0.001405] | undefined (undefined_support_no_dropping) | -0.353836 [-0.545292, -0.186078] | -0.231955 [-0.372209, -0.112760] |

## Lower-Proxy Differences

These are differences of lower bounds, not bounds on paired differences.
They remain separate from the same-outcome paired completion intervals above.

| Comparator | Full lower-proxy difference | Matched lower-proxy difference |
|---|---|---|
| original | +1.395548 [+0.571491, +2.195878] | +0.265465 [+0.037867, +0.572923] |
| additive | +1.109573 [+0.311687, +1.904905] | +0.245381 [+0.009968, +0.540311] |
| poisson | +1.351727 [+0.534454, +2.152466] | +0.276739 [+0.028260, +0.596523] |
| cost | +1.389808 [+0.566720, +2.190690] | +0.261510 [+0.019947, +0.576775] |
| none | -0.418593 [-0.850034, -0.083136] | -0.340183 [-0.701233, -0.067811] |
| rowmean | -0.047442 [-0.164936, +0.068123] | -0.075093 [-0.168284, +0.007354] |

## Safety and Support

Counts below are repeated source/seed occurrences, not distinct agents or
independent windows. The registered budget is 2% selected positive easy
harm/reference. Undefined reference support is not a passing safety result.

| Policy | Selected occurrences | Unknown selected | Undefined easy risk /72 | Easy-upper violations /72 | Worst easy upper | Complete finite support /72 |
|---|---:|---:|---:|---:|---:|---:|
| original | 95455 | 918 | 29 | 7 | 5.405761% | 33 |
| additive | 112456 | 1143 | 11 | 42 | 1200.168389% | 19 |
| poisson | 111031 | 1050 | 21 | 11 | 18.027742% | 37 |
| cost | 96720 | 926 | 24 | 7 | 5.719494% | 36 |
| none | 107596 | 1487 | 0 | 72 | 1159.622176% | 0 |
| rowmean | 84251 | 1201 | 0 | 72 | 848.686895% | 0 |
| temporal | 83168 | 1191 | 0 | 72 | 1153.549199% | 0 |
| original_matched_temporal | 45406 | 459 | 33 | 12 | 7.675617% | 24 |
| temporal_matched_original | 45406 | 563 | 36 | 31 | 214.402851% | 4 |
| additive_matched_temporal | 51651 | 605 | 13 | 42 | 1242.153698% | 17 |
| temporal_matched_additive | 51651 | 665 | 13 | 51 | 3805.406277% | 6 |
| poisson_matched_temporal | 50441 | 524 | 21 | 19 | 34.962107% | 24 |
| temporal_matched_poisson | 50441 | 622 | 27 | 43 | 245.607643% | 1 |
| cost_matched_temporal | 46042 | 464 | 26 | 12 | 7.675617% | 26 |
| temporal_matched_cost | 46042 | 569 | 31 | 34 | 238.087126% | 7 |
| none_matched_temporal | 74069 | 1068 | 0 | 72 | 1507.603854% | 0 |
| temporal_matched_none | 74069 | 1053 | 0 | 71 | 998.041745% | 1 |
| rowmean_matched_temporal | 73245 | 1033 | 0 | 72 | 962.920810% | 0 |
| temporal_matched_rowmean | 73245 | 1042 | 0 | 72 | 1078.751936% | 0 |

## Unmet Conditions

- unsupported_original_all_MSE
- unsupported_original_original_selected_MSE
- unsupported_additive_all_MSE
- unsupported_additive_original_selected_MSE
- unsupported_additive_matched_paired_lower_percent
- unsupported_poisson_all_MSE
- unsupported_poisson_original_selected_MSE
- unsupported_cost_all_MSE
- unsupported_cost_original_selected_MSE
- unsupported_none_all_MSE
- unsupported_none_original_selected_MSE
- unsupported_none_full_lower_proxy_delta_percent
- unsupported_none_full_paired_lower_percent
- unsupported_none_matched_lower_proxy_delta_percent
- unsupported_none_matched_paired_lower_percent
- unsupported_rowmean_original_selected_MSE
- unsupported_rowmean_full_lower_proxy_delta_percent
- unsupported_rowmean_full_paired_lower_percent
- unsupported_rowmean_matched_lower_proxy_delta_percent
- unsupported_rowmean_matched_paired_lower_percent
- absolute_original_risk_or_utility_not_supported_temporal
- absolute_original_risk_or_utility_not_supported_temporal_matched_original
- absolute_original_risk_or_utility_not_supported_temporal_matched_additive
- absolute_original_risk_or_utility_not_supported_temporal_matched_poisson
- absolute_original_risk_or_utility_not_supported_temporal_matched_cost
- absolute_original_risk_or_utility_not_supported_temporal_matched_none
- absolute_original_risk_or_utility_not_supported_temporal_matched_rowmean

## Verification and Limits

- 11391961 scalar cross-checks; original-control replay and repeated neural inference match exactly.
- Readout process elapsed 222.26 seconds; peak RSS 7.330 GiB.
- No checkpoint disk cache, new training, transfer evaluation or deployment change during readout.
- Image-local detector-silver obs8/pred12 at raw stride12; no meter/seconds/human-gold/physical-safety claims.
- Cost-head learning around frozen predictors is not new world-dynamics pretraining, true 3D, or foundation-model evidence.
- Stage5C and SMC remain disabled.
