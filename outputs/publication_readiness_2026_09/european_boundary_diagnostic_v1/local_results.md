# Local Frozen-Boundary Readout

fresh_run local diagnosis and exact numerical replay. All 288 reconstructed packet bytes match the previously committed CREATE packet hashes. cached_verified frozen inputs, models and actions. No training or policy changes. This is not completed CREATE analysis or independent confirmation.

| Phase | Arm | Event | Actual risk (%) | Predicted excess (pp) | Harm error (pp) | Reference error (pp) | Violations / defined |
|---|---|---|---:|---:|---:|---:|---:|
| fitting | affine | all | 2.7794 | -3.0267 | 1.1371 | 2.6690 | 45/72 |
| fitting | affine | easy | 5.7280 | -1.0171 | 5.2771 | -0.5320 | 45/72 |
| fitting | nonlinear | all | 2.8062 | -2.2980 | 1.8138 | 1.2905 | 47/72 |
| fitting | nonlinear | easy | 7.9003 | -1.2294 | 7.7106 | -0.5809 | 61/72 |
| internal_transfer | affine | all | 4.3135 | -4.3395 | 2.6650 | 3.9881 | 158/216 |
| internal_transfer | affine | easy | 8.4537 | -1.1386 | 7.9603 | -0.3680 | 166/216 |
| internal_transfer | nonlinear | all | 5.6317 | -3.6565 | 4.4607 | 2.8275 | 168/216 |
| internal_transfer | nonlinear | easy | 14.0113 | -1.1642 | 13.8401 | -0.6646 | 178/216 |

Components share the actual selected-reference denominator. Actual risk = 2% + predicted excess + harm error + reference error. Signed components may offset. Equal-locality weighting. Unknown outcomes and zero denominators remain unknown, not safety passes.

There are 72 fitting and 216 transfer views of 12 previously opened localities; they are dependent. No fresh bootstrap or independent calibration/confirmation. Image-local detector-silver, obs8/pred12 stride12 raw frames only. No metric/seconds, true3D, foundation or human-gold claim. Stage5C and SMC remain off.

Verified: 64,512 native checks; 672 summary checks; 1,728 parent agreements. Peak RSS 7.492 GiB; no row cache. Wall time 187.98s.
