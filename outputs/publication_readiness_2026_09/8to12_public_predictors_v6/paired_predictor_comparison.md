# Matched Public-Core vs Local Predictor

Fresh real training and evaluation on historically exposed development data.
Same obs8/pred12 task, rows, complete-aligned past neighbors, seeds, updates and loss.
Equal update/sample budgets do not mean equal model capacity or compute. Per-fit device and elapsed time are retained in the input metrics; explicit CPU recovery is not hidden as MPS-only execution.
EqMotion is a fixed K=1 author-core adaptation, not published best-of-20 performance.

| Seed | Transformer gain vs CV % | EqMotion-K1 gain vs CV % | EqMotion gain vs Transformer % | Local oracle % | Public oracle % | Local selection | Public selection |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 17 | -5.736 | -14.119 | -7.929 | 0.556 | 0.455 | floor | floor |
| 29 | -7.686 | -8.473 | -0.730 | 0.579 | 0.604 | floor | floor |
| 43 | -6.700 | -14.081 | -6.918 | 0.506 | 0.576 | floor | floor |

Primary errors are past-normalized ADE, not meters. Oracle reads labels only for diagnosis.
One physical development scene prevents a meaningful scene confidence interval. Seed SD is not scene uncertainty.
Full reports retain easy absolute/relative error, hard error, switch rate, harm, all policies and label denominators.

## Native-Coordinate Strong Baseline Context

| Model | Seed | Recording | Candidate ADE | CV gain % | Best development causal ADE | Gain over that baseline % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| transformer | 17 | ucy_students01 | 0.477761 | -4.280 | 0.454433 | -5.133 |
| transformer | 17 | ucy_students03 | 0.657150 | 3.615 | 0.628628 | -4.537 |
| transformer | 29 | ucy_students01 | 0.475305 | -3.744 | 0.454433 | -4.593 |
| transformer | 29 | ucy_students03 | 0.648099 | 4.943 | 0.628628 | -3.097 |
| transformer | 43 | ucy_students01 | 0.477596 | -4.244 | 0.454433 | -5.097 |
| transformer | 43 | ucy_students03 | 0.648122 | 4.939 | 0.628628 | -3.101 |
| eqmotion_K1 | 17 | ucy_students01 | 0.476999 | -4.113 | 0.454433 | -4.966 |
| eqmotion_K1 | 17 | ucy_students03 | 0.615410 | 9.737 | 0.628628 | 2.103 |
| eqmotion_K1 | 29 | ucy_students01 | 0.480294 | -4.832 | 0.454433 | -5.691 |
| eqmotion_K1 | 29 | ucy_students03 | 0.614055 | 9.936 | 0.628628 | 2.318 |
| eqmotion_K1 | 43 | ucy_students01 | 0.464096 | -1.297 | 0.454433 | -2.126 |
| eqmotion_K1 | 43 | ucy_students03 | 0.622719 | 8.665 | 0.628628 | 0.940 |

The development-best causal baseline is descriptive context, not a changed floor or a test-selected deployment rule.
This table does not evaluate raw-frame t+50 or actual-count-matched controls; consult their separate supplementary reports. Real deferral, independent calibration/confirmation and visual-scene contribution remain unestablished.
