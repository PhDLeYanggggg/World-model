# Risk-Conditioned Residual Results

fresh_run:864 fixed ridge probes and144 fitting-only frozen-Torch inferences.
cached_verified:432 inner risk heads, original outer estimators and prior controls. No new neural or trajectory training.
Source development only; independent selection, calibration and confirmation remain unopened.

| Comparison / inputs | Positive / negative / overlapping / missing MSE intervals | Point range (%) |
|---|---|---|
| risk_oof_vs_original / full | 0 / 0 / 6 / 0 | [-2.2159652107778, -0.19362630316696472] |
| risk_oof_vs_original / motion_only | 1 / 0 / 5 / 0 | [-14.268631861074121, 1.6110713895159887] |
| risk_oof_vs_common_context / full | 0 / 0 / 6 / 0 | [-0.24668881767259832, 2.098148789698873] |
| risk_oof_vs_common_context / motion_only | 3 / 0 / 3 / 0 | [-0.23981064324483958, 8.979871177070981] |
| risk_oof_vs_inner_event / full | 0 / 2 / 4 / 0 | [-1.6877039242089842, 2.406615773272002] |
| risk_oof_vs_inner_event / motion_only | 0 / 1 / 5 / 0 | [-6.062417388802211, 5.949233771061478] |
| risk_oof_vs_outer_context / full | 0 / 3 / 3 / 0 | [-1.1436528841775795, -0.03663556101815438] |
| risk_oof_vs_outer_context / motion_only | 3 / 1 / 2 / 0 | [-10.23063142880716, 7.698828272242306] |
| risk_oof_vs_risk_only / full | 2 / 1 / 3 / 0 | [-1.0028161447059634, 1.4003287049419288] |
| risk_oof_vs_risk_only / motion_only | 0 / 0 / 6 / 0 | [-10.973571121134547, -0.02923754747982177] |
| risk_oof_vs_risk_next / full | 0 / 2 / 4 / 0 | [-3.4108144568272634, 0.36319898796633] |
| risk_oof_vs_risk_next / motion_only | 2 / 1 / 3 / 0 | [-11.746159391873308, 8.075154358997633] |
| risk_oof_vs_risk_prev / full | 0 / 3 / 3 / 0 | [-3.654992293736945, -0.4303883621892424] |
| risk_oof_vs_risk_prev / motion_only | 1 / 1 / 4 / 0 | [-26.18830214713177, 5.437889688394604] |
| oof_risk_only_vs_original / full | 0 / 1 / 5 / 0 | [-1.7424611259385334, -0.4021879031364187] |
| oof_risk_only_vs_original / motion_only | 2 / 0 / 4 / 0 | [-14.471717119303959, 4.584432924309934] |
| oof_risk_context_vs_original / full | 0 / 0 / 6 / 0 | [-2.2159652107778, -0.19362630316696472] |
| oof_risk_context_vs_original / motion_only | 1 / 0 / 5 / 0 | [-14.268631861074121, 1.6110713895159887] |
| in_sample_next_risk_only_vs_original / full | 1 / 0 / 5 / 0 | [-1.0031020278063931, 0.5869908510679352] |
| in_sample_next_risk_only_vs_original / motion_only | 2 / 2 / 2 / 0 | [-7.209955606756292, 2.564048521351242] |
| in_sample_next_risk_context_vs_original / full | 2 / 0 / 4 / 0 | [-2.063798519643604, 1.117626378644712] |
| in_sample_next_risk_context_vs_original / motion_only | 2 / 1 / 3 / 0 | [-16.232178080983047, 2.3807328467190327] |
| in_sample_prev_risk_only_vs_original / full | 2 / 0 / 4 / 0 | [-0.6065890343286421, 1.1587502485509713] |
| in_sample_prev_risk_only_vs_original / motion_only | 2 / 0 / 4 / 0 | [-6.549019795344153, 2.5578929855476917] |
| in_sample_prev_risk_context_vs_original / full | 2 / 0 / 4 / 0 | [-0.5874870041286442, 1.3353019304966358] |
| in_sample_prev_risk_context_vs_original / motion_only | 2 / 0 / 4 / 0 | [-12.460285252897767, 2.5712133044489667] |

Range is six point estimates, not one CI. Three seeds averaged per locality;3000 paired resamples of four localities per assignment.
Six assignments overlap. No multiplicity adjustment or window-independent claims. No negative interval is not a safety proof.
Only predicted H_E changes. This is cost estimation, not a trajectory gain, independent calibration or deployment.

```json
{
  "risk_conditioning_signal": false,
  "primary_six_positive_all_controls": false,
  "tail_coverage_guards": false,
  "risk_conditioned_repair_signal": false,
  "policy_evaluated": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Obs8/pred12 annotation steps, detector pixels. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.
