# Residual Range and Objective Comparison

54 fresh Torch fits; 18 exactly replayed controls. Fixed-end checkpoints, no selection on held outcomes.
All 11,966 previously exposed fit windows retained. No independent confirmation or deployment.

| Feature / arm | Gain vs CV (%) | Gain vs linear/log (%) | Safe positive held fits |
| --- | ---: | ---: | ---: |
| quality_control_linear_log | -0.99582 | 0.00000 | 0/9 |
| quality_control_sinh_log | -0.95875 | 0.03671 | 0/9 |
| quality_control_linear_asinh | -14.17415 | -13.04839 | 0/9 |
| quality_control_sinh_asinh | -72.66424 | -70.96177 | 0/9 |
| directed_linear_log | -0.97244 | 0.00000 | 0/9 |
| directed_sinh_log | -0.90994 | 0.06190 | 0/9 |
| directed_linear_asinh | -15.79533 | -14.68013 | 0/9 |
| directed_sinh_asinh | -171.03448 | -168.42420 | 0/9 |

Sinh cap is numerical only, not a physical bound. Asinh is a training target transform, not a changed evaluation metric.
All scene intervals are exploratory resampling over three exposed scenes.
