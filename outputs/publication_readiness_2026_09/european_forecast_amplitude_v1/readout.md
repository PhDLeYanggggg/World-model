# Fixed Forecast Amplitude: Fitting-Only Results

## Provenance

Fresh: all 144 output-control groups / 720 amplitude views and a separate exact
inference replay. Cached_verified: source packets, frozen trajectories and cost
estimators. Not_run: new learning, held/independent evaluation or deployment.

PID 20388, exit 0, takes 575.886368s; peak RSS 7,060,619,264 B.
Replay PID 21457, exit 0, takes 574.746712s.
All original endpoint choices and full metrics match the preceding experiment.
Every fixed-mask convexity check passes. This numerical property does not
certify the original learned conditional costs or real-world safety.

## Complete Reselected Policies

| Family | Alpha | Bank | Known ADE gain vs floor (%) | Required recording risk support | Undefined easy | Known all/easy violations | Selected (unknown) |
|---|---:|---|---:|---:|---:|---:|---:|
| forest | 1 | physical_only | +0.000243 | 333/444 | 106 | 0/0 | 5764 (70) |
| forest | 1 | physical_neural | +0.000111 | 330/444 | 104 | 2/2 | 7026 (80) |
| forest | 1 | neural_only | +0.071258 | 165/444 | 209 | 16/24 | 85544 (836) |
| forest | 1 | factorized | +0.071514 | 285/444 | 104 | 1/7 | 88680 (883) |
| forest | 0.5 | physical_only | +0.000916 | 368/444 | 69 | 4/4 | 17026 (174) |
| forest | 0.5 | physical_neural | +0.001096 | 369/444 | 69 | 3/3 | 20172 (214) |
| forest | 0.5 | neural_only | +0.107531 | 226/444 | 138 | 17/17 | 201771 (1883) |
| forest | 0.5 | factorized | +0.108613 | 313/444 | 57 | 9/13 | 205538 (1932) |
| forest | 0.25 | physical_only | +0.012026 | 399/444 | 41 | 0/3 | 53918 (513) |
| forest | 0.25 | physical_neural | +0.024538 | 395/444 | 44 | 0/2 | 67574 (622) |
| forest | 0.25 | neural_only | +0.104640 | 262/444 | 116 | 0/3 | 256945 (2417) |
| forest | 0.25 | factorized | +0.112795 | 350/444 | 35 | 0/3 | 267145 (2506) |
| forest | 0.125 | physical_only | +0.164232 | 351/444 | 18 | 0/4 | 351693 (3225) |
| forest | 0.125 | physical_neural | +0.164019 | 334/444 | 19 | 0/3 | 362047 (3399) |
| forest | 0.125 | neural_only | +0.100090 | 288/444 | 85 | 1/4 | 299328 (2897) |
| forest | 0.125 | factorized | +0.211957 | 306/444 | 18 | 0/4 | 378057 (3541) |
| forest | 0.0625 | physical_only | +0.191768 | 264/444 | 10 | 0/0 | 472721 (4868) |
| forest | 0.0625 | physical_neural | +0.197107 | 251/444 | 9 | 0/0 | 470255 (4838) |
| forest | 0.0625 | neural_only | +0.126925 | 297/444 | 18 | 0/3 | 368176 (3728) |
| forest | 0.0625 | factorized | +0.257150 | 225/444 | 10 | 0/2 | 481670 (5080) |
| neural | 1 | physical_only | +0.671048 | 73/444 | 50 | 85/229 | 77771 (901) |
| neural | 1 | physical_neural | +0.678064 | 77/444 | 49 | 92/230 | 70561 (853) |
| neural | 1 | neural_only | +0.579900 | 126/444 | 95 | 117/113 | 23993 (314) |
| neural | 1 | factorized | +1.122806 | 57/444 | 42 | 116/239 | 89911 (1076) |
| neural | 0.5 | physical_only | +0.790283 | 82/444 | 47 | 34/163 | 133306 (1790) |
| neural | 0.5 | physical_neural | +0.825022 | 70/444 | 43 | 40/186 | 122792 (1679) |
| neural | 0.5 | neural_only | +0.644216 | 133/444 | 67 | 89/92 | 44355 (600) |
| neural | 0.5 | factorized | +1.276370 | 65/444 | 34 | 59/168 | 154889 (2113) |
| neural | 0.25 | physical_only | +0.905931 | 67/444 | 39 | 13/123 | 197731 (3029) |
| neural | 0.25 | physical_neural | +0.989775 | 64/444 | 31 | 8/125 | 184619 (2935) |
| neural | 0.25 | neural_only | +0.695332 | 117/444 | 41 | 41/71 | 78086 (1177) |
| neural | 0.25 | factorized | +1.398761 | 51/444 | 27 | 15/116 | 233857 (3644) |
| neural | 0.125 | physical_only | +0.913529 | 50/444 | 30 | 5/73 | 253324 (4273) |
| neural | 0.125 | physical_neural | +1.027710 | 43/444 | 22 | 3/78 | 238194 (4197) |
| neural | 0.125 | neural_only | +0.683283 | 92/444 | 27 | 1/48 | 122685 (2187) |
| neural | 0.125 | factorized | +1.327806 | 34/444 | 18 | 5/62 | 308377 (5292) |
| neural | 0.0625 | physical_only | +0.728568 | 48/444 | 25 | 3/34 | 285360 (5031) |
| neural | 0.0625 | physical_neural | +0.818671 | 40/444 | 18 | 1/36 | 269038 (4968) |
| neural | 0.0625 | neural_only | +0.532545 | 82/444 | 22 | 0/15 | 165314 (3307) |
| neural | 0.0625 | factorized | +0.989977 | 29/444 | 14 | 0/23 | 359204 (6551) |

All coefficients and banks are reported, not an outcome-selected winning model.
Every risk row retains all required recordings. Undefined risk is not safe zero.

## Reselection Versus The Same Frozen Endpoint Policy

Positive pp means the fractional policy improves its own alpha=1 endpoint.
Count matching concerns utility only, never the safety contract.

| Family | Alpha | Bank | Full known change (pp), fitting 95% CI | Full completion-lower (pp), fitting 95% CI | Matched known change (pp), fitting 95% CI |
|---|---:|---|---|---|---|
| forest | 1 | physical_only | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| forest | 1 | physical_neural | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| forest | 1 | neural_only | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| forest | 1 | factorized | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| forest | 0.5 | physical_only | +0.000673 [-0.000187, +0.002147] | +0.000452 [-0.000268, +0.001671] | -0.000105 [-0.000415, +0.000067] |
| forest | 0.5 | physical_neural | +0.000985 [-0.000229, +0.002931] | +0.000807 [-0.000303, +0.002574] | +0.000023 [-0.000073, +0.000106] |
| forest | 0.5 | neural_only | +0.036272 [+0.018044, +0.057032] | +0.033240 [+0.016144, +0.052458] | -0.016500 [-0.035319, -0.003168] |
| forest | 0.5 | factorized | +0.037100 [+0.019388, +0.057545] | +0.033901 [+0.017415, +0.053005] | -0.015342 [-0.033521, -0.002277] |
| forest | 0.25 | physical_only | +0.011783 [+0.002863, +0.025788] | +0.010400 [+0.002663, +0.021838] | +0.000470 [+0.000152, +0.000883] |
| forest | 0.25 | physical_neural | +0.024427 [+0.007633, +0.050635] | +0.022204 [+0.007334, +0.044659] | +0.001209 [+0.000527, +0.001955] |
| forest | 0.25 | neural_only | +0.033382 [+0.007283, +0.063005] | +0.028052 [+0.002792, +0.056191] | -0.034100 [-0.069243, -0.007790] |
| forest | 0.25 | factorized | +0.041281 [+0.011409, +0.077306] | +0.035325 [+0.007149, +0.067811] | -0.031747 [-0.066022, -0.006014] |
| forest | 0.125 | physical_only | +0.163989 [+0.101426, +0.233732] | +0.153553 [+0.095777, +0.216861] | +0.003958 [+0.001486, +0.006970] |
| forest | 0.125 | physical_neural | +0.163908 [+0.098618, +0.238826] | +0.152830 [+0.092709, +0.220182] | +0.004695 [+0.001886, +0.007950] |
| forest | 0.125 | neural_only | +0.028832 [-0.015855, +0.080123] | +0.012363 [-0.028919, +0.057357] | -0.044283 [-0.088432, -0.010460] |
| forest | 0.125 | factorized | +0.140443 [+0.081465, +0.214089] | +0.119433 [+0.065514, +0.182121] | -0.017695 [-0.039357, -0.001464] |
| forest | 0.0625 | physical_only | +0.191526 [+0.123404, +0.263769] | +0.179953 [+0.115966, +0.247137] | +0.004288 [+0.001211, +0.007787] |
| forest | 0.0625 | physical_neural | +0.196996 [+0.128218, +0.272761] | +0.182485 [+0.116574, +0.251304] | +0.005272 [+0.002149, +0.008837] |
| forest | 0.0625 | neural_only | +0.055666 [-0.007916, +0.117382] | +0.028482 [-0.027081, +0.082324] | -0.046528 [-0.093975, -0.008826] |
| forest | 0.0625 | factorized | +0.185636 [+0.122068, +0.255705] | +0.151669 [+0.093380, +0.210227] | -0.022208 [-0.048698, -0.001227] |
| neural | 1 | physical_only | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| neural | 1 | physical_neural | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| neural | 1 | neural_only | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| neural | 1 | factorized | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| neural | 0.5 | physical_only | +0.119235 [+0.003075, +0.220956] | +0.016147 [-0.101957, +0.118270] | -0.032513 [-0.121656, +0.038302] |
| neural | 0.5 | physical_neural | +0.146958 [+0.008525, +0.272204] | +0.025643 [-0.109377, +0.151351] | -0.008952 [-0.108352, +0.077114] |
| neural | 0.5 | neural_only | +0.064316 [-0.069197, +0.194653] | -0.039637 [-0.162957, +0.066755] | -0.158429 [-0.311898, -0.027636] |
| neural | 0.5 | factorized | +0.153564 [-0.016725, +0.325590] | -0.031673 [-0.187022, +0.114359] | -0.130653 [-0.301817, +0.026395] |
| neural | 0.25 | physical_only | +0.234883 [+0.030516, +0.439706] | +0.074312 [-0.126580, +0.259631] | -0.110251 [-0.254564, +0.014608] |
| neural | 0.25 | physical_neural | +0.311711 [+0.081518, +0.530111] | +0.119799 [-0.071670, +0.287443] | -0.072476 [-0.236985, +0.077448] |
| neural | 0.25 | neural_only | +0.115432 [-0.164435, +0.371052] | -0.047240 [-0.310617, +0.164630] | -0.304606 [-0.577768, -0.081993] |
| neural | 0.25 | factorized | +0.275955 [-0.103259, +0.626119] | -0.011498 [-0.367419, +0.297757] | -0.327492 [-0.662188, -0.042311] |
| neural | 0.125 | physical_only | +0.242481 [+0.005848, +0.465666] | +0.041891 [-0.167353, +0.243907] | -0.223279 [-0.419661, -0.045090] |
| neural | 0.125 | physical_neural | +0.349646 [+0.094835, +0.593688] | +0.122205 [-0.094238, +0.335446] | -0.196080 [-0.399356, -0.016193] |
| neural | 0.125 | neural_only | +0.103383 [-0.255559, +0.429675] | -0.093223 [-0.426956, +0.187003] | -0.394792 [-0.725751, -0.123616] |
| neural | 0.125 | factorized | +0.205000 [-0.310205, +0.627985] | -0.126418 [-0.593613, +0.242886] | -0.521453 [-0.963257, -0.162228] |
| neural | 0.0625 | physical_only | +0.057520 [-0.252213, +0.349913] | -0.115038 [-0.404625, +0.162563] | -0.370020 [-0.637059, -0.132310] |
| neural | 0.0625 | physical_neural | +0.140607 [-0.160606, +0.435496] | -0.070710 [-0.352942, +0.219520] | -0.357352 [-0.617311, -0.126051] |
| neural | 0.0625 | neural_only | -0.047355 [-0.427442, +0.270991] | -0.216339 [-0.580801, +0.077818] | -0.467863 [-0.827361, -0.172213] |
| neural | 0.0625 | factorized | -0.132829 [-0.732341, +0.311942] | -0.408560 [-0.971958, +0.010687] | -0.731037 [-1.267269, -0.329109] |

## Output Change Without Reselection

These controls retain the endpoint policy selection set, except analytic no-ops.
They distinguish a smaller output from the newly admitted actions in the table above.

| Family | Alpha | Bank | Known ADE gain vs floor (%) | Recording support | Undefined easy | Selected (unknown) |
|---|---:|---|---:|---:|---:|---:|
| forest | 1 | physical_only | +0.000243 | 333/444 | 106 | 5764 (70) |
| forest | 1 | physical_neural | +0.000111 | 330/444 | 104 | 7026 (80) |
| forest | 1 | neural_only | +0.071258 | 165/444 | 209 | 85544 (836) |
| forest | 1 | factorized | +0.071514 | 285/444 | 104 | 88680 (883) |
| forest | 0.5 | physical_only | +0.000126 | 335/444 | 106 | 5764 (70) |
| forest | 0.5 | physical_neural | +0.000061 | 332/444 | 104 | 7026 (80) |
| forest | 0.5 | neural_only | +0.036715 | 214/444 | 209 | 85544 (836) |
| forest | 0.5 | factorized | +0.036846 | 325/444 | 104 | 88680 (883) |
| forest | 0.25 | physical_only | +0.000064 | 337/444 | 106 | 5764 (70) |
| forest | 0.25 | physical_neural | +0.000032 | 335/444 | 104 | 7026 (80) |
| forest | 0.25 | neural_only | +0.018594 | 227/444 | 209 | 85544 (836) |
| forest | 0.25 | factorized | +0.018661 | 333/444 | 104 | 88680 (883) |
| forest | 0.125 | physical_only | +0.000032 | 337/444 | 106 | 5764 (70) |
| forest | 0.125 | physical_neural | +0.000017 | 336/444 | 104 | 7026 (80) |
| forest | 0.125 | neural_only | +0.009352 | 231/444 | 209 | 85544 (836) |
| forest | 0.125 | factorized | +0.009385 | 336/444 | 104 | 88680 (883) |
| forest | 0.0625 | physical_only | +0.000016 | 337/444 | 106 | 5764 (70) |
| forest | 0.0625 | physical_neural | +0.000008 | 336/444 | 104 | 7026 (80) |
| forest | 0.0625 | neural_only | +0.004689 | 231/444 | 209 | 85544 (836) |
| forest | 0.0625 | factorized | +0.004706 | 337/444 | 104 | 88680 (883) |
| neural | 1 | physical_only | +0.671048 | 73/444 | 50 | 77771 (901) |
| neural | 1 | physical_neural | +0.678064 | 77/444 | 49 | 70561 (853) |
| neural | 1 | neural_only | +0.579900 | 126/444 | 95 | 23993 (314) |
| neural | 1 | factorized | +1.122806 | 57/444 | 42 | 89911 (1076) |
| neural | 0.5 | physical_only | +0.361324 | 164/444 | 50 | 77771 (901) |
| neural | 0.5 | physical_neural | +0.365750 | 165/444 | 49 | 70561 (853) |
| neural | 0.5 | neural_only | +0.325611 | 185/444 | 95 | 23993 (314) |
| neural | 0.5 | factorized | +0.616696 | 149/444 | 42 | 89911 (1076) |
| neural | 0.25 | physical_only | +0.184447 | 245/444 | 50 | 77771 (901) |
| neural | 0.25 | physical_neural | +0.186670 | 235/444 | 49 | 70561 (853) |
| neural | 0.25 | neural_only | +0.168001 | 232/444 | 95 | 23993 (314) |
| neural | 0.25 | factorized | +0.316595 | 233/444 | 42 | 89911 (1076) |
| neural | 0.125 | physical_only | +0.093071 | 280/444 | 50 | 77771 (901) |
| neural | 0.125 | physical_neural | +0.094143 | 272/444 | 49 | 70561 (853) |
| neural | 0.125 | neural_only | +0.085110 | 257/444 | 95 | 23993 (314) |
| neural | 0.125 | factorized | +0.160124 | 271/444 | 42 | 89911 (1076) |
| neural | 0.0625 | physical_only | +0.046739 | 304/444 | 50 | 77771 (901) |
| neural | 0.0625 | physical_neural | +0.047258 | 291/444 | 49 | 70561 (853) |
| neural | 0.0625 | neural_only | +0.042815 | 275/444 | 95 | 23993 (314) |
| neural | 0.0625 | factorized | +0.080497 | 292/444 | 42 | 89911 (1076) |

## Neural Bank Versus The Physical Bank At The Same Amplitude

Descriptive contrasts from the same verified groups, not an alpha-selection rule.

| Family | Alpha | Mixed minus physical known gain (pp), fitting 95% CI | Completion-lower (pp), fitting 95% CI |
|---|---:|---|---|
| forest | 1 | -0.000132 [-0.000335, -0.000019] | -0.000134 [-0.000337, -0.000022] |
| forest | 0.5 | +0.000180 [-0.000108, +0.000685] | +0.000119 [-0.000153, +0.000610] |
| forest | 0.25 | +0.012512 [+0.004486, +0.024870] | +0.011437 [+0.004260, +0.021918] |
| forest | 0.125 | -0.000213 [-0.008333, +0.007979] | -0.004256 [-0.011752, +0.002529] |
| forest | 0.0625 | +0.005339 [-0.000627, +0.013001] | -0.001557 [-0.005951, +0.002725] |
| neural | 1 | +0.007016 [-0.040237, +0.047475] | -0.065668 [-0.127277, -0.007714] |
| neural | 0.5 | +0.034739 [-0.012376, +0.084861] | -0.075255 [-0.146292, -0.021026] |
| neural | 0.25 | +0.083844 [+0.016006, +0.150347] | -0.030095 [-0.124577, +0.037521] |
| neural | 0.125 | +0.114181 [+0.069000, +0.166084] | +0.018536 [-0.036756, +0.066644] |
| neural | 0.0625 | +0.090103 [+0.054860, +0.128650] | +0.015549 [-0.032177, +0.056583] |

## Risk Failure Accounting

| Family | Alpha | Policy | Known selected-risk violation views | Unsupported with selected unknown but no known violation | Undefined easy |
|---|---:|---|---:|---:|---:|
| forest | 1 | physical_neural | 2 | 8 | 104 |
| forest | 1 | physical_neural_same_set | 2 | 8 | 104 |
| forest | 0.5 | physical_neural | 3 | 3 | 69 |
| forest | 0.5 | physical_neural_same_set | 2 | 6 | 104 |
| forest | 0.25 | physical_neural | 2 | 5 | 44 |
| forest | 0.25 | physical_neural_same_set | 0 | 5 | 104 |
| forest | 0.125 | physical_neural | 3 | 95 | 19 |
| forest | 0.125 | physical_neural_same_set | 0 | 4 | 104 |
| forest | 0.0625 | physical_neural | 0 | 188 | 9 |
| forest | 0.0625 | physical_neural_same_set | 0 | 4 | 104 |
| neural | 1 | physical_neural | 270 | 62 | 49 |
| neural | 1 | physical_neural_same_set | 270 | 62 | 49 |
| neural | 0.5 | physical_neural | 204 | 141 | 43 |
| neural | 0.5 | physical_neural_same_set | 104 | 134 | 49 |
| neural | 0.25 | physical_neural | 129 | 231 | 31 |
| neural | 0.25 | physical_neural_same_set | 27 | 140 | 49 |
| neural | 0.125 | physical_neural | 79 | 309 | 22 |
| neural | 0.125 | physical_neural_same_set | 6 | 123 | 49 |
| neural | 0.0625 | physical_neural | 37 | 356 | 18 |
| neural | 0.0625 | physical_neural_same_set | 1 | 109 | 49 |

These counts overlap and do not uniquely identify why a prediction is wrong.
An unknown-outcome bound failing is not evidence that the unknown realized error
was large. Missing support is not imputed away or treated as observed harm.

## Evidence Boundaries

Cost transport uses convex-loss surrogate benefit/harm bounds, not freshly fitted
expected fractional errors. Actual ADE/FDE and canonical completion envelopes
are recomputed from the new paths; endpoint errors are not multiplied by alpha.
Labels and future masks enter scoring only. The zero-amplitude engineering
control is an analytic no-op and does not pass empty selected-risk support.

Intervals use 3,000 locality resamples after repeated context/head-seed averaging.
They are fitting-resubstitution, not generalization, whole-pipeline OOF or
independent confirmation. Endpoint percentages that lack a required denominator
remain not_estimable in summary.json. Completion utility divides by fixed known
floor-error mass, not an identified complete-population error denominator.

The 444 recording views per family repeat 74 unique fitting recordings. Neither
query/amplitude repetitions nor three head seeds create new forecasting pipelines.
The factorized policy still uses twice the cost-estimator capacity. Even a positive
fitting amplitude control cannot establish learned neural dynamics or deployment.

EuropeanSquares SOURCE detector-track midpoint pixels, obs8/pred12/stride12 raw
frames; not SDD human gold, metric/seconds, true 3D or foundation evidence.
Independent roles and DroneCrowd stay closed. No Stage5C or SMC.

## Gates

- all144groups720amplitude_views_complete: true
- endpoint_exact: true
- numerical_convexity_checks: 720
- positive_fitting_full_risk_controls: []
- at_least_one_fitting_full_risk_and_completion_gain: false
- any_coefficient_selected_for_deployment: false
- learned_costs_are_calibrated: false
- held_scored: false
- new_optimizer_updates: 0
- new_trees: 0
- independent_confirmation: false
- deployment_promoted: false
- submission_ready: false
- stage5c_executed: false
- smc_enabled: false
