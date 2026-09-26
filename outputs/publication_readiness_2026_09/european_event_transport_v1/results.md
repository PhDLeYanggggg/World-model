# Common-Event Residual Results

## Material Passport
fresh_run:864 fixed ridge probes; cached_verified:432 inner risk heads and original outer estimators.
No new neural or trajectory training. Source-development only, not independent confirmation.

| Comparison / inputs | Positive / negative / overlapping / missing MSE intervals | Point range (%) |
|---|---|---|
| common_oof_vs_old_oof / full | 0 / 1 / 5 / 0 | [-3.9816315719757123, 1.8992307107868436] |
| common_oof_vs_old_oof / motion_only | 0 / 1 / 5 / 0 | [-10.578033472116465, 3.078720845523349] |
| common_oof_vs_original / full | 0 / 1 / 5 / 0 | [-3.96293730363889, 0.05044690600619274] |
| common_oof_vs_original / motion_only | 0 / 0 / 6 / 0 | [-19.945204673549632, -1.6390562308566206] |
| common_oof_vs_outer_context / full | 0 / 4 / 2 / 0 | [-2.73618505979776, -0.22779880277870102] |
| common_oof_vs_outer_context / motion_only | 0 / 2 / 4 / 0 | [-15.144735558333478, -0.23024137664610253] |
| common_oof_vs_common_global / full | 1 / 1 / 4 / 0 | [-2.2670688874758045, 1.0183153063787167] |
| common_oof_vs_common_global / motion_only | 1 / 0 / 5 / 0 | [-11.719056740197829, 5.382997092672028] |
| common_oof_vs_common_next / full | 0 / 2 / 4 / 0 | [-2.76378764817126, -0.47330536027933967] |
| common_oof_vs_common_next / motion_only | 0 / 1 / 5 / 0 | [-15.739008584440878, -0.06215700174975249] |
| common_oof_vs_common_prev / full | 0 / 1 / 5 / 0 | [-3.92358711368232, -0.3494978325321325] |
| common_oof_vs_common_prev / motion_only | 0 / 2 / 4 / 0 | [-19.349338171038738, -2.328126640861108] |
| oof_global_bias_common_vs_inner / full | 0 / 1 / 5 / 0 | [-2.4070404115888366, 0.48690669227746564] |
| oof_global_bias_common_vs_inner / motion_only | 0 / 2 / 4 / 0 | [-6.571125096951972, 2.9477285121048804] |
| oof_context_bias_common_vs_inner / full | 0 / 1 / 5 / 0 | [-3.9816315719757123, 1.8992307107868436] |
| oof_context_bias_common_vs_inner / motion_only | 0 / 1 / 5 / 0 | [-10.578033472116465, 3.078720845523349] |
| in_sample_next_global_bias_common_vs_inner / full | 0 / 1 / 5 / 0 | [-1.2079594756197412, 0.014601542785141319] |
| in_sample_next_global_bias_common_vs_inner / motion_only | 0 / 2 / 4 / 0 | [-2.5291601181942145, -0.014087281829249412] |
| in_sample_next_context_bias_common_vs_inner / full | 0 / 2 / 4 / 0 | [-1.7856653587471059, 0.07207602158699379] |
| in_sample_next_context_bias_common_vs_inner / motion_only | 0 / 0 / 6 / 0 | [-2.836896709160631, 0.15060205266341603] |
| in_sample_prev_global_bias_common_vs_inner / full | 0 / 1 / 5 / 0 | [-0.19424521542676265, 0.5471042140129787] |
| in_sample_prev_global_bias_common_vs_inner / motion_only | 1 / 0 / 5 / 0 | [0.22931001277217417, 3.540514368404356] |
| in_sample_prev_context_bias_common_vs_inner / full | 1 / 1 / 4 / 0 | [-0.1018337340229346, 1.2216076003742504] |
| in_sample_prev_context_bias_common_vs_inner / motion_only | 1 / 0 / 5 / 0 | [0.025194567306370808, 2.210700173090193] |

Range denotes six point estimates, not one CI. Three seeds averaged per locality,3000 paired locality resamples.
Six assignments overlap; four localities per assignment, no multiplicity adjustment. No negative interval is not a safety proof.
Common event is meta-fitting-only; inner teachers still predict their original events. No teacher retraining or independent calibration.
The864 exact preclip event projections isolate this fixed label change algebraically, not a causal explanation of all errors.

```json
{
  "event_alignment_signal": false,
  "primary_six_positive_all_controls": false,
  "tail_coverage_guards": false,
  "common_event_repair_signal": false,
  "policy_evaluated": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Eight observed/twelve predicted annotation steps, detector pixels. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.
