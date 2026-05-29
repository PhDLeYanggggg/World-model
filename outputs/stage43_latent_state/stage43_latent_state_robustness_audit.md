# Stage43-D Latent-State Robustness Audit

- source: `fresh_stage43_d_latent_state_robustness_audit`
- verdict: `stage43_d_latent_state_robustness_ucy_pass`
- gate: `9 / 9`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt`
- checkpoint committed: `False`
- latent variance: `0.400007`
- domain scope: `{'test_domains': ['UCY'], 'multi_domain_test': False, 'limitation': 'Current Stage43-D robustness audit evaluates the full held-out UCY test split only; it does not yet prove multi-external-domain robustness.'}`

## Full Held-Out Test Metrics vs Stage37/Stage42 Floor

- rows: `66303`
- all improvement: `0.163151`
- t50 improvement: `0.136820`
- t100 raw-frame diagnostic: `0.009722`
- hard/failure improvement: `0.164765`
- easy degradation: `0.000000`
- switch rate: `0.170113`

## Bootstrap CI

| metric | rows | mean | ci low | ci high |
| --- | ---: | ---: | ---: | ---: |
| all | 66303 | 0.163151 | 0.159866 | 0.166509 |
| t50 | 16263 | 0.136820 | 0.130597 | 0.142779 |
| t100_raw_frame_diagnostic | 10008 | 0.009722 | 0.007416 | 0.012160 |
| hard_failure | 45917 | 0.164765 | 0.160986 | 0.168408 |
| easy_degradation | 20798 | 0.000000 | 0.000000 | 0.000000 |

## Gate

| gate | passed |
| --- | --- |
| stage43_c_checkpoint_exists | True |
| full_or_larger_test_eval_completed | True |
| latent_noncollapse | True |
| all_ci_low_positive | True |
| t50_ci_low_positive | True |
| hard_failure_ci_low_positive | True |
| easy_ci_high_safe | True |
| domain_scope_recorded | True |
| no_metric_seconds_stage5c_smc_claim | True |

This audit supports a UCY held-out dataset-local/raw-frame protected latent-state result only. It does not authorize a multi-domain, metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.
