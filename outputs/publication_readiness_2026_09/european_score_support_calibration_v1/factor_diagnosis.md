# Frozen-Policy Factor Diagnosis

Secondary paired contrasts from registered controls. No refit, winner selection or threshold change.

| Contrast | Paired ADE gain % | 95% locality interval | Intervention change points |
|---|---:|---|---:|
| dimensionless_moments_calibrated_vs_guarded | -0.119181 | [-0.1902636563078432, -0.058200429989240694] | -5.2981 |
| dimensionless_moments_supported_vs_guarded | -0.077252 | [-0.16230259328919155, -0.017476407727372842] | -0.1281 |
| dimensionless_moments_calibrated_supported_vs_supported | -0.097822 | [-0.13960132225994643, -0.05866390575074382] | -5.8397 |
| dimensionless_moments_calibrated_supported_vs_calibrated | -0.055949 | [-0.1332412920775503, -0.01268711602030392] | -0.6697 |
| dimensionless_excess_calibrated_vs_guarded | -0.103357 | [-0.15107602502876696, -0.056154177062926684] | -4.2274 |
| dimensionless_excess_supported_vs_guarded | -0.050027 | [-0.09197466397052169, -0.014661305538439163] | -0.1247 |
| dimensionless_excess_calibrated_supported_vs_supported | -0.066305 | [-0.10643407184551328, -0.030613479168104064] | -4.1729 |
| dimensionless_excess_calibrated_supported_vs_calibrated | -0.013024 | [-0.02091382722594352, -0.005641114666615703] | -0.0702 |
| damped_moments_calibrated_vs_guarded | 0.000000 | [0.0, 0.0] | 0.0000 |
| damped_moments_supported_vs_guarded | -0.007387 | [-0.014618487739026511, -0.002253790439645018] | -0.0911 |
| damped_moments_calibrated_supported_vs_supported | 0.000000 | [0.0, 0.0] | 0.0000 |
| damped_moments_calibrated_supported_vs_calibrated | -0.007387 | [-0.014618487739026511, -0.002253790439645018] | -0.0911 |
| damped_excess_calibrated_vs_guarded | 0.000000 | [0.0, 0.0] | 0.0000 |
| damped_excess_supported_vs_guarded | -0.007622 | [-0.015178637234326909, -0.0023323581389759563] | -0.0979 |
| damped_excess_calibrated_supported_vs_supported | 0.000000 | [0.0, 0.0] | 0.0000 |
| damped_excess_calibrated_supported_vs_calibrated | -0.007622 | [-0.015178637234326909, -0.0023323581389759563] | -0.0979 |

## Remaining Signed-Calibrated-Supported Risk Violations

| Site | Seed | Candidate | Harm/reference % | Net ADE gain % | Easy gain % | Switches |
|---|---:|---|---:|---:|---:|---:|
| eu-locality-082 | 17 | dimensionless | 2.0271 | 0.1292 | 1.9575 | 160 |
| eu-locality-082 | 17 | dimensionless | 2.0271 | 0.1292 | 1.9575 | 160 |
| eu-locality-082 | 17 | dimensionless | 2.0271 | 0.1292 | 1.9575 | 160 |
| eu-locality-020 | 29 | dimensionless | 3.4295 | 0.0751 | 0.7913 | 154 |
| eu-locality-020 | 29 | dimensionless | 3.4295 | 0.0751 | 0.7913 | 154 |
| eu-locality-020 | 29 | dimensionless | 3.4295 | 0.0751 | 0.7913 | 154 |
| eu-locality-020 | 43 | dimensionless | 2.6437 | 0.1376 | 1.1899 | 167 |
| eu-locality-020 | 43 | dimensionless | 2.6437 | 0.1376 | 1.1899 | 167 |
| eu-locality-020 | 43 | dimensionless | 2.6437 | 0.1376 | 1.1899 | 167 |

Repeated source/seed rows can belong to different calibration pairs. Positive harm is not net
degradation. Overlapping views are averaged at the locality level, not treated as independent samples.
