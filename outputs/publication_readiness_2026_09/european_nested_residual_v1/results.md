# Nested Residual Results

## Material Passport
Fresh432 native Torch cost heads /864000 updates and864 closed-form probes. Frozen trajectory producers and outer estimators are cached_verified.
Source-development readout; not independent calibration or confirmation. No new trajectory training or deployment.

| Comparison / inputs | Positive / negative / overlap / missing MSE intervals | Point range (%) |
|---|---|---|
| oof_context_vs_original / full | 0 / 0 / 6 / 0 | [-5.001898409283898, 0.19762833620398657] |
| oof_context_vs_original / motion_only | 0 / 0 / 6 / 0 | [-22.04296183447165, -3.202835907342694] |
| oof_context_vs_outer_context / full | 0 / 1 / 5 / 0 | [-2.936660777554414, 1.144119594377774] |
| oof_context_vs_outer_context / motion_only | 1 / 0 / 5 / 0 | [-17.041854038474625, 7.761851973435011] |
| oof_context_vs_oof_global / full | 1 / 0 / 5 / 0 | [-3.77285013016316, 0.34161817945880796] |
| oof_context_vs_oof_global / motion_only | 1 / 1 / 4 / 0 | [-7.7523626213999, 6.698247414589511] |
| oof_context_vs_next / full | 0 / 1 / 5 / 0 | [-4.982872215270126, 0.39607232647904667] |
| oof_context_vs_next / motion_only | 1 / 0 / 5 / 0 | [-20.60929227649851, 5.909968110570022] |
| oof_context_vs_prev / full | 0 / 0 / 6 / 0 | [-3.62236551165035, 0.291818964673133] |
| oof_context_vs_prev / motion_only | 1 / 0 / 5 / 0 | [-17.136058609573496, 5.80254034282936] |
| oof_global_bias_vs_original / full | 1 / 0 / 5 / 0 | [-1.1205237957544483, 0.7041774560347197] |
| oof_global_bias_vs_original / motion_only | 0 / 0 / 6 / 0 | [-37.51081248757062, 1.3839917988768773] |
| in_sample_next_global_bias_vs_original / full | 0 / 0 / 6 / 0 | [-0.15973069847347507, 0.030843807645913816] |
| in_sample_next_global_bias_vs_original / motion_only | 1 / 2 / 3 / 0 | [-6.302844413545904, 0.32483321793512004] |
| in_sample_next_context_bias_vs_original / full | 2 / 0 / 4 / 0 | [-0.3732314076468068, 0.541774500781527] |
| in_sample_next_context_bias_vs_original / motion_only | 1 / 1 / 4 / 0 | [-12.39822930115209, 0.5581416775618995] |
| in_sample_prev_global_bias_vs_original / full | 0 / 0 / 6 / 0 | [-0.5737500373469084, 0.11759196295747247] |
| in_sample_prev_global_bias_vs_original / motion_only | 0 / 0 / 6 / 0 | [-10.728050126475484, -0.03673196290647368] |
| in_sample_prev_context_bias_vs_original / full | 3 / 0 / 3 / 0 | [-1.2787423250333503, 0.4694951869749305] |
| in_sample_prev_context_bias_vs_original / motion_only | 0 / 0 / 6 / 0 | [-12.862535329727669, -0.023458475870741985] |

Three seeds averaged within locality,3000 resamples of four localities per assignment. Six dependent assignments retained; no multiplicity adjustment.
All inner cuts/scales are fitting-only. Different inner/outer cuts and two-vs-three locality training remain transport limitations, not hidden endpoint changes.
No negative tail interval is not a safety or noninferiority proof.

```json
{
  "primary_six_positive_all_controls": false,
  "tail_coverage_guards": false,
  "nested_residual_diagnostic_signal": false,
  "new_forecaster_training": false,
  "independent_calibration": false,
  "independent_confirmation": false,
  "policy_evaluated": false,
  "deployment_changed": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Eight observed/twelve predicted annotation steps; detector pixels only. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.
