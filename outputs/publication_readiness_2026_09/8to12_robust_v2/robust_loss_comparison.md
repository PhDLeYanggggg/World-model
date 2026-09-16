# MSE vs Smooth-L1: Paired Development Ablation

Fresh analysis of completed real experiments; same rows, model, updates, seeds and evaluation.
Adaptive development, not a new untouched test. Training loss values use different units and are not compared.

| Seed | MSE gain vs CV % | Smooth-L1 gain vs CV % | Candidate ADE reduction vs MSE % | MSE oracle % | Robust oracle % | Robust selection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 17 | -7.093 | -0.231 | 6.407 | 0.722 | 0.320 | floor |
| 29 | -7.900 | -0.329 | 7.017 | 0.704 | 0.351 | floor |
| 43 | -7.249 | -0.181 | 6.590 | 0.441 | 0.231 | floor |

Primary ADE is past-normalized, not meters. Positive improvement versus the failed MSE model
does not mean improvement versus CV, preservation of easy cases, or a useful joint-selection mechanism.
The JSON preserves absolute/relative easy errors, per-recording native-coordinate results and all seed decisions.
Oracle rows read future labels for evaluation only. They are not inference scores or model results.
One University development site cannot support a physical-scene CI; seed SD is training variability only.

## Native-Coordinate Strong Baseline Context

| Loss | Seed | Recording | Candidate ADE | CV gain % | Lowest development causal ADE | Gain vs that baseline % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| mse | 17 | ucy_students01 | 0.588812 | -18.871 | 0.488945 | -20.425 |
| mse | 17 | ucy_students03 | 0.677736 | 0.596 | 0.628628 | -7.812 |
| mse | 29 | ucy_students01 | 0.577596 | -16.607 | 0.488945 | -18.131 |
| mse | 29 | ucy_students03 | 0.727748 | -6.739 | 0.628628 | -15.768 |
| mse | 43 | ucy_students01 | 0.942960 | -90.368 | 0.488945 | -92.856 |
| mse | 43 | ucy_students03 | 0.968199 | -42.007 | 0.628628 | -54.018 |
| robust | 17 | ucy_students01 | 0.496083 | -0.151 | 0.488945 | -1.460 |
| robust | 17 | ucy_students03 | 0.639077 | 6.266 | 0.628628 | -1.662 |
| robust | 29 | ucy_students01 | 0.503047 | -1.557 | 0.488945 | -2.884 |
| robust | 29 | ucy_students03 | 0.626024 | 8.181 | 0.628628 | 0.414 |
| robust | 43 | ucy_students01 | 0.494911 | 0.086 | 0.488945 | -1.220 |
| robust | 43 | ucy_students03 | 0.653535 | 4.145 | 0.628628 | -3.962 |

Units remain recording-local and unverified. The lowest development causal score is
descriptive context, not a test-selected floor or a retrospective protocol change.
Do not interpret a CV-only gain as an advantage over the stronger causal comparator.
