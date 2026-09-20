# Frozen Neural Forecasts: Coupling Comparison

Post-hoc development evidence on two already explored UCY recordings of one physical site.
No new training, threshold selection, primary-metric change or deployment. All 24 fixed combinations retained.
Each combination: 970 scene queries, 37,775 agent queries; 28,324 ADE labels and 28,335 endpoint labels.
Agent queries can overlap in time. Repetition across 24 fits does not multiply the independent population.

## Same-Predictor Contrasts

Negative ADE/FDE difference favors full joint over unary geometry. Errors retain the original past normalization.

| Family | Candidate | Joint minus unary ADE | Joint minus unary FDE | Nonzero matched queries | Switch disagreements |
| --- | --- | ---: | ---: | ---: | ---: |
| transformer | seed17_ridge_conservative | 0 | 0 | 905 | 0 |
| transformer | seed17_ridge_moderate | 0 | 0 | 926 | 0 |
| transformer | seed17_neural_cost_conservative | 0 | 0 | 583 | 0 |
| transformer | seed17_neural_cost_moderate | 0 | 0 | 751 | 0 |
| transformer | seed29_ridge_conservative | -8.7911347606e-06 | -1.35308243472e-05 | 933 | 2 |
| transformer | seed29_ridge_moderate | 0 | 0 | 963 | 0 |
| transformer | seed29_neural_cost_conservative | 0 | 0 | 464 | 0 |
| transformer | seed29_neural_cost_moderate | 0 | 0 | 647 | 0 |
| transformer | seed43_ridge_conservative | 0 | 0 | 538 | 0 |
| transformer | seed43_ridge_moderate | 0 | 0 | 650 | 0 |
| transformer | seed43_neural_cost_conservative | 0 | 0 | 632 | 0 |
| transformer | seed43_neural_cost_moderate | 0 | 0 | 773 | 0 |
| eqmotion | seed17_ridge_conservative | 0 | 0 | 883 | 0 |
| eqmotion | seed17_ridge_moderate | 0 | 0 | 942 | 0 |
| eqmotion | seed17_neural_cost_conservative | 0 | 0 | 121 | 0 |
| eqmotion | seed17_neural_cost_moderate | 0 | 0 | 301 | 0 |
| eqmotion | seed29_ridge_conservative | -1.52241684228e-05 | -2.37677296564e-05 | 968 | 42 |
| eqmotion | seed29_ridge_moderate | 6.01726903948e-06 | 1.39058460427e-05 | 968 | 20 |
| eqmotion | seed29_neural_cost_conservative | 0 | 0 | 508 | 0 |
| eqmotion | seed29_neural_cost_moderate | 0 | 0 | 679 | 0 |
| eqmotion | seed43_ridge_conservative | 0 | 0 | 755 | 0 |
| eqmotion | seed43_ridge_moderate | 0 | 0 | 774 | 0 |
| eqmotion | seed43_neural_cost_conservative | 0 | 0 | 591 | 0 |
| eqmotion | seed43_neural_cost_moderate | 0 | 0 | 728 | 0 |

## Coupling Opportunity And Replay

Each row has 970 scene queries. Possible products need not change an optimum; changed identities need not improve forecast accuracy.
These counts repeat the same observed population across fixed fits and are not independent samples.

| Family | Candidate | Possible product queries | Proxy-advantage queries | Unmatched queries | New/legacy risk-only disagreements | Legacy row changes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| transformer | seed17_ridge_conservative | 60 | 0 | 0 | 0 | 0 |
| transformer | seed17_ridge_moderate | 69 | 0 | 0 | 0 | 0 |
| transformer | seed17_neural_cost_conservative | 7 | 0 | 0 | 0 | 0 |
| transformer | seed17_neural_cost_moderate | 24 | 0 | 0 | 0 | 0 |
| transformer | seed29_ridge_conservative | 78 | 1 | 0 | 0 | 0 |
| transformer | seed29_ridge_moderate | 130 | 0 | 0 | 0 | 0 |
| transformer | seed29_neural_cost_conservative | 2 | 0 | 0 | 0 | 0 |
| transformer | seed29_neural_cost_moderate | 10 | 0 | 0 | 0 | 0 |
| transformer | seed43_ridge_conservative | 3 | 0 | 0 | 0 | 0 |
| transformer | seed43_ridge_moderate | 4 | 0 | 0 | 0 | 0 |
| transformer | seed43_neural_cost_conservative | 3 | 0 | 0 | 0 | 0 |
| transformer | seed43_neural_cost_moderate | 13 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_ridge_conservative | 53 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_ridge_moderate | 89 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_neural_cost_conservative | 1 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_neural_cost_moderate | 7 | 0 | 0 | 0 | 0 |
| eqmotion | seed29_ridge_conservative | 354 | 20 | 0 | 0 | 0 |
| eqmotion | seed29_ridge_moderate | 428 | 9 | 0 | 2 | 0 |
| eqmotion | seed29_neural_cost_conservative | 5 | 0 | 0 | 0 | 0 |
| eqmotion | seed29_neural_cost_moderate | 13 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_ridge_conservative | 8 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_ridge_moderate | 13 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_neural_cost_conservative | 15 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_neural_cost_moderate | 24 | 0 | 0 | 0 | 0 |

## Changed Decisions, Not New Independent Cases

Post-hoc descriptive attribution only; favorable rows are not used for selection or a new success criterion.

| Family | Candidate | Changed scene queries | Changed agent queries | ADE-labeled changed agents | Better | Worse | Equal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer | seed17_ridge_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed17_ridge_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed17_neural_cost_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed17_neural_cost_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed29_ridge_conservative | 1 | 2 | 1 | 1 | 0 | 0 |
| transformer | seed29_ridge_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed29_neural_cost_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed29_neural_cost_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed43_ridge_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed43_ridge_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed43_neural_cost_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| transformer | seed43_neural_cost_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_ridge_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_ridge_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_neural_cost_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed17_neural_cost_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed29_ridge_conservative | 20 | 42 | 30 | 19 | 11 | 0 |
| eqmotion | seed29_ridge_moderate | 9 | 20 | 14 | 6 | 8 | 0 |
| eqmotion | seed29_neural_cost_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed29_neural_cost_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_ridge_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_ridge_moderate | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_neural_cost_conservative | 0 | 0 | 0 | 0 | 0 | 0 |
| eqmotion | seed43_neural_cost_moderate | 0 | 0 | 0 | 0 | 0 | 0 |

## All New Controls

I = risk-only independent; U = geometry-aware independent at exact count; J = full joint at exact count.
The CSV and analysis JSON retain FDE, positive harm, tail errors, coverage, raw-recording errors and all legacy controls.

| Family | Candidate | Arm | ADE gain vs CV (%) | Easy degradation (%) | Hard gain (%) | Switch rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| transformer | seed17_ridge_conservative | I | -0.2337143 | 2474.46 | -0.02409742 | 0.09601588 |
| transformer | seed17_ridge_conservative | U | -0.2337143 | 2474.46 | -0.02409742 | 0.09601588 |
| transformer | seed17_ridge_conservative | J | -0.2337143 | 2474.46 | -0.02409742 | 0.09601588 |
| transformer | seed17_ridge_moderate | I | -0.2424415 | 2620.503 | -0.01927438 | 0.1162674 |
| transformer | seed17_ridge_moderate | U | -0.2424415 | 2620.503 | -0.01927438 | 0.1162674 |
| transformer | seed17_ridge_moderate | J | -0.2424415 | 2620.503 | -0.01927438 | 0.1162674 |
| transformer | seed17_neural_cost_conservative | I | -0.004936137 | 107.262 | 0.00627327 | 0.03052283 |
| transformer | seed17_neural_cost_conservative | U | -0.004936137 | 107.262 | 0.00627327 | 0.03052283 |
| transformer | seed17_neural_cost_conservative | J | -0.004936137 | 107.262 | 0.00627327 | 0.03052283 |
| transformer | seed17_neural_cost_moderate | I | -0.01348302 | 178.8628 | 0.006353833 | 0.05641297 |
| transformer | seed17_neural_cost_moderate | U | -0.01348302 | 178.8628 | 0.006353833 | 0.05641297 |
| transformer | seed17_neural_cost_moderate | J | -0.01348302 | 178.8628 | 0.006353833 | 0.05641297 |
| transformer | seed29_ridge_conservative | I | -0.0117984 | 548.4971 | 0.04061198 | 0.1199206 |
| transformer | seed29_ridge_conservative | U | -0.01194309 | 548.4971 | 0.04046658 | 0.1199206 |
| transformer | seed29_ridge_conservative | J | -0.0117984 | 548.4971 | 0.04061198 | 0.1199206 |
| transformer | seed29_ridge_moderate | I | -0.008720354 | 669.3812 | 0.05874384 | 0.1807015 |
| transformer | seed29_ridge_moderate | U | -0.008720354 | 669.3812 | 0.05874384 | 0.1807015 |
| transformer | seed29_ridge_moderate | J | -0.008720354 | 669.3812 | 0.05874384 | 0.1807015 |
| transformer | seed29_neural_cost_conservative | I | 0.002742608 | 31.30719 | 0.00633339 | 0.01784249 |
| transformer | seed29_neural_cost_conservative | U | 0.002742608 | 31.30719 | 0.00633339 | 0.01784249 |
| transformer | seed29_neural_cost_conservative | J | 0.002742608 | 31.30719 | 0.00633339 | 0.01784249 |
| transformer | seed29_neural_cost_moderate | I | 0.003913255 | 69.88806 | 0.01216154 | 0.03454666 |
| transformer | seed29_neural_cost_moderate | U | 0.003913255 | 69.88806 | 0.01216154 | 0.03454666 |
| transformer | seed29_neural_cost_moderate | J | 0.003913255 | 69.88806 | 0.01216154 | 0.03454666 |
| transformer | seed43_ridge_conservative | I | -0.1943655 | 1874.225 | -0.03964406 | 0.02546658 |
| transformer | seed43_ridge_conservative | U | -0.1943655 | 1874.225 | -0.03964406 | 0.02546658 |
| transformer | seed43_ridge_conservative | J | -0.1943655 | 1874.225 | -0.03964406 | 0.02546658 |
| transformer | seed43_ridge_moderate | I | -0.2020778 | 1953.7 | -0.04038874 | 0.0345996 |
| transformer | seed43_ridge_moderate | U | -0.2020778 | 1953.7 | -0.04038874 | 0.0345996 |
| transformer | seed43_ridge_moderate | J | -0.2020778 | 1953.7 | -0.04038874 | 0.0345996 |
| transformer | seed43_neural_cost_conservative | I | 0.00546235 | 146.9581 | 0.0206271 | 0.0341231 |
| transformer | seed43_neural_cost_conservative | U | 0.00546235 | 146.9581 | 0.0206271 | 0.0341231 |
| transformer | seed43_neural_cost_conservative | J | 0.00546235 | 146.9581 | 0.0206271 | 0.0341231 |
| transformer | seed43_neural_cost_moderate | I | -0.002375572 | 339.1927 | 0.03014083 | 0.05514229 |
| transformer | seed43_neural_cost_moderate | U | -0.002375572 | 339.1927 | 0.03014083 | 0.05514229 |
| transformer | seed43_neural_cost_moderate | J | -0.002375572 | 339.1927 | 0.03014083 | 0.05514229 |
| eqmotion | seed17_ridge_conservative | I | -1.442159 | 16349.81 | -0.1005199 | 0.07912641 |
| eqmotion | seed17_ridge_conservative | U | -1.442159 | 16349.81 | -0.1005199 | 0.07912641 |
| eqmotion | seed17_ridge_conservative | J | -1.442159 | 16349.81 | -0.1005199 | 0.07912641 |
| eqmotion | seed17_ridge_moderate | I | -1.407831 | 16400.89 | -0.05821796 | 0.1211648 |
| eqmotion | seed17_ridge_moderate | U | -1.407831 | 16400.89 | -0.05821796 | 0.1211648 |
| eqmotion | seed17_ridge_moderate | J | -1.407831 | 16400.89 | -0.05821796 | 0.1211648 |
| eqmotion | seed17_neural_cost_conservative | I | -0.006805119 | 124.3657 | 0.003851615 | 0.004023825 |
| eqmotion | seed17_neural_cost_conservative | U | -0.006805119 | 124.3657 | 0.003851615 | 0.004023825 |
| eqmotion | seed17_neural_cost_conservative | J | -0.006805119 | 124.3657 | 0.003851615 | 0.004023825 |
| eqmotion | seed17_neural_cost_moderate | I | -0.01325153 | 183.9938 | 0.002863472 | 0.01180675 |
| eqmotion | seed17_neural_cost_moderate | U | -0.01325153 | 183.9938 | 0.002863472 | 0.01180675 |
| eqmotion | seed17_neural_cost_moderate | J | -0.01325153 | 183.9938 | 0.002863472 | 0.01180675 |
| eqmotion | seed29_ridge_conservative | I | -1.825493 | 21597.52 | -0.02491763 | 0.2344937 |
| eqmotion | seed29_ridge_conservative | U | -1.825835 | 21598.08 | -0.02518085 | 0.2344937 |
| eqmotion | seed29_ridge_conservative | J | -1.825584 | 21598.08 | -0.02506944 | 0.2344937 |
| eqmotion | seed29_ridge_moderate | I | -1.846694 | 21821.61 | -0.008064264 | 0.3960026 |
| eqmotion | seed29_ridge_moderate | U | -1.84639 | 21822.35 | -0.007723362 | 0.3960026 |
| eqmotion | seed29_ridge_moderate | J | -1.846489 | 21821.61 | -0.007948803 | 0.3960026 |
| eqmotion | seed29_neural_cost_conservative | I | -0.04353485 | 517.1105 | 0.001016387 | 0.02131039 |
| eqmotion | seed29_neural_cost_conservative | U | -0.04353485 | 517.1105 | 0.001016387 | 0.02131039 |
| eqmotion | seed29_neural_cost_conservative | J | -0.04353485 | 517.1105 | 0.001016387 | 0.02131039 |
| eqmotion | seed29_neural_cost_moderate | I | -0.08769207 | 1140.065 | 0.009527497 | 0.03629385 |
| eqmotion | seed29_neural_cost_moderate | U | -0.08769207 | 1140.065 | 0.009527497 | 0.03629385 |
| eqmotion | seed29_neural_cost_moderate | J | -0.08769207 | 1140.065 | 0.009527497 | 0.03629385 |
| eqmotion | seed43_ridge_conservative | I | -0.3782308 | 3639.09 | -0.07916951 | 0.04825943 |
| eqmotion | seed43_ridge_conservative | U | -0.3782308 | 3639.09 | -0.07916951 | 0.04825943 |
| eqmotion | seed43_ridge_conservative | J | -0.3782308 | 3639.09 | -0.07916951 | 0.04825943 |
| eqmotion | seed43_ridge_moderate | I | -0.394119 | 3824.568 | -0.07971539 | 0.05265387 |
| eqmotion | seed43_ridge_moderate | U | -0.394119 | 3824.568 | -0.07971539 | 0.05265387 |
| eqmotion | seed43_ridge_moderate | J | -0.394119 | 3824.568 | -0.07971539 | 0.05265387 |
| eqmotion | seed43_neural_cost_conservative | I | -0.03316248 | 452.6919 | 0.007829262 | 0.02917273 |
| eqmotion | seed43_neural_cost_conservative | U | -0.03316248 | 452.6919 | 0.007829262 | 0.02917273 |
| eqmotion | seed43_neural_cost_conservative | J | -0.03316248 | 452.6919 | 0.007829262 | 0.02917273 |
| eqmotion | seed43_neural_cost_moderate | I | -0.05096856 | 680.019 | 0.009559557 | 0.04465917 |
| eqmotion | seed43_neural_cost_moderate | U | -0.05096856 | 680.019 | 0.009559557 | 0.04465917 |
| eqmotion | seed43_neural_cost_moderate | J | -0.05096856 | 680.019 | 0.009559557 | 0.04465917 |

## Evidence Limits

Exactly one physical site: scene-bootstrap confidence intervals are unavailable, not replaced by window bootstraps.
Three seed values describe fit variability only. Repeated head/policy exports are not independent agents or scenes.
Matched past-agent counts do not imply equal scored-label coverage, equal realized risk or physical safety.
Original future validity and fixed forecast errors are verified before admitting each query. No future label influences decisions.
The historical model bytes are executed from a verified private mirror; the newer checkout remains unchanged.
No Stage5C, SMC, seconds, metric, foundation or submission-readiness claim.
