# Stage43-H Source-Level Latent Robustness Audit

- source: `fresh_stage43_h_source_level_latent_robustness`
- verdict: `stage43_h_unit_consistent_audit_failed_keep_floor`
- gate: `9 / 10`
- deploy Stage43-G: `False`
- keep frozen floor: `True`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt`
- checkpoint committed: `False`

## Unit Consistency Finding

Stage43-G reported normalized-delta error against a dataset-local floor FDE. Stage43-H recomputes neural endpoint error as `normalized_error * row_scale` before comparing with the floor.

| metric | normalized Stage43-G | unit-consistent Stage43-H |
| --- | ---: | ---: |
| all | 0.858018 | 0.351410 |
| t50 | 0.821362 | 0.158059 |
| t100 raw diagnostic | 0.783976 | 0.004466 |
| hard/failure | 0.866818 | 0.377402 |
| easy degradation | 0.000000 | 1.597489 |

## Bootstrap CI on Unit-Consistent Metrics

| metric | rows | mean | ci low | ci high |
| --- | ---: | ---: | ---: | ---: |
| unit_all | 89736 | 0.351410 | 0.347108 | 0.356375 |
| unit_t50 | 21754 | 0.158059 | 0.150491 | 0.163156 |
| unit_t100_raw_frame_diagnostic | 18070 | 0.004466 | -0.004712 | 0.012482 |
| unit_hard_failure | 70119 | 0.377402 | 0.372777 | 0.380607 |
| unit_easy_degradation | 26927 | 1.597489 | 1.572842 | 1.639598 |

## Endpoint Proximity Proxy

- selected near@0.05: `0.033504`
- floor near@0.05: `0.037551`
- near@0.05 delta: `-0.004048`
- selected near@0.10: `0.168449`
- floor near@0.10: `0.160795`
- near@0.10 delta: `0.007653`

## Gate

| gate | passed |
| --- | --- |
| stage43_g_checkpoint_exists | True |
| source_level_full_test_eval_completed | True |
| unit_mismatch_detected_and_reported | True |
| unit_all_ci_low_positive | True |
| unit_t50_ci_low_positive | True |
| unit_hard_ci_low_positive | True |
| easy_preservation_gate | False |
| proximity_not_materially_worse | True |
| full_switch_caveat_recorded | True |
| no_metric_seconds_stage5c_smc_claim | True |

Conclusion: Stage43-G remains an interesting source-level neural dynamics signal, but the deployment claim is not valid under unit-consistent safety auditing because easy degradation is unsafe. Keep the frozen Stage37/Stage42 floor and repair with a calibrated safe-switch policy.
