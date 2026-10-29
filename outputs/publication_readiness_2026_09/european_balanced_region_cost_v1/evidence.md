# Complete-Region Cost Learning: Fixed Evidence

Verified completed training and exact SOURCE-development readout. This export
does not train,select a threshold,create a new bootstrap or open held roles.

| Bank | Known ADE gain % [conditional95%CI] | Supported record/head views | Undefined easy | Selected (unknown) |
|---|---:|---:|---:|---:|
| physical_only | 0.932450 [0.238201,2.097328] | 33/186 | 18 | 36977 (549) |
| physical_neural | 0.943728 [0.284133,2.015080] | 24/186 | 19 | 36182 (584) |
| neural_only | 0.722430 [0.341358,1.121427] | 29/186 | 21 | 19392 (350) |

## Paired Mixed-Bank Contrasts

Percentage points;unknown outcomes remain in completion bounds.
| Contrast | Known | Completion lower | Completion upper |
|---|---:|---:|---:|
| complete minus stochastic full | -0.000080 [-0.108174,0.097264] | -0.644138 [-1.790891,0.020977] | 0.643977 [0.070206,1.603176] |
| complete minus stochastic matched | 0.005261 [-0.048063,0.063554] | -0.410482 [-1.093460,-0.000935] | 0.421004 [0.042481,1.026623] |
| mixed minus physical full | 0.011277 [-0.121830,0.102166] | -0.904799 [-2.609433,0.018689] | 0.927353 [0.093799,2.410792] |
| mixed minus physical matched | -0.007701 [-0.131179,0.072763] | -0.583211 [-1.707389,0.008589] | 0.567809 [0.070891,1.481000] |

## Actual Compute, Not Equal FLOPs

54fresh Torch cost heads,108,000primary optimizer updates,
454.858144s,peak RSS 10,212,655,104bytes.
Additional known teacher-row visits:12,275,011.
Additional two-pass query-row forwards:24,550,022.
Candidate multiplication depends on bank size. Identical base draws and steps
do not mean identical training FLOPs. The trajectory producer stays frozen.
Three seeds are controller seeds,not three new full forecasting pipelines.

Full required-recording all/easy2% gate: False
Positive completion-lower neural-increment CI: False

Cyclic training is exact within one fixed teacher region,not an exact
full-all-region update or independent calibration. Empty selected risk is
not_estimable,not safe zero. Matched counts measure utility only. No harm
multiplier,seed or subgroup is chosen from readout. SOURCE inner-held is
exposed development;3,000-locality CIs are conditional,not confirmation.
Obs8/pred12/rawstride12 webcam detector-track pixels are not metric/seconds,
human gold,true3D or foundation evidence. No deployment promotion,
Stage5C execution,SMC or submission readiness.
