# Stage43-H Source-Level Latent Robustness Gate

- verdict: `stage43_h_unit_consistent_audit_failed_keep_floor`
- gate: `9 / 10`
- deploy Stage43-G: `False`
- keep frozen floor: `True`

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
