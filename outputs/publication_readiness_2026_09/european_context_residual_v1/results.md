# Causal Context Residual Results

## Material Passport
864 fresh closed-form probes on432 frozen estimators /144 aligned views. Zero new Torch updates.
Cached_verified producers, predictors and exposed source development. Not independent calibration.

| Comparison / inputs | Positive / negative / overlap / missing MSE intervals | Point range (%) |
|---|---|---|
| original_global_vs_self / full | 1 / 0 / 5 / 0 | [-1.1111587449096931, 0.2698592811858658] |
| original_global_vs_self / motion_only | 0 / 1 / 5 / 0 | [-10.472765622605309, -0.8354321583166189] |
| original_context_vs_self / full | 2 / 0 / 4 / 0 | [-1.8796009859698606, 0.9351186614399043] |
| original_context_vs_self / motion_only | 0 / 0 / 6 / 0 | [-16.532726500747206, -0.5337792038511876] |
| original_context_vs_global / full | 2 / 0 / 4 / 0 | [-1.1154598893328187, 0.667892153136461] |
| original_context_vs_global / motion_only | 1 / 0 / 5 / 0 | [-8.274575782852091, 5.2873896792485615] |
| ordinary_aux_global_vs_self / full | 1 / 0 / 5 / 0 | [-0.8375330018459125, 0.4265585917723851] |
| ordinary_aux_global_vs_self / motion_only | 0 / 1 / 5 / 0 | [-11.311710102174505, -0.28279598158043084] |
| ordinary_aux_context_vs_self / full | 1 / 0 / 5 / 0 | [-1.6658671715860993, 0.9235021196232317] |
| ordinary_aux_context_vs_self / motion_only | 1 / 0 / 5 / 0 | [-20.648400893352314, 1.442765730793143] |
| ordinary_aux_context_vs_global / full | 2 / 0 / 4 / 0 | [-0.9876492603855069, 0.502129215327468] |
| ordinary_aux_context_vs_global / motion_only | 1 / 0 / 5 / 0 | [-9.123524928765363, 6.412788549998667] |
| ordinary_aux_context_vs_original / full | 0 / 0 / 6 / 0 | [-9.970299966035222, 0.5249110346091362] |
| ordinary_aux_context_vs_original / motion_only | 0 / 0 / 6 / 0 | [-22.44760507298775, -0.5257057694265008] |
| severity_aux_global_vs_self / full | 1 / 0 / 5 / 0 | [-0.8762574624648819, 0.2934008496434684] |
| severity_aux_global_vs_self / motion_only | 0 / 1 / 5 / 0 | [-10.262648125814616, -0.41194913172218106] |
| severity_aux_context_vs_self / full | 1 / 0 / 5 / 0 | [-1.4626145995790705, 0.8492305203992181] |
| severity_aux_context_vs_self / motion_only | 0 / 0 / 6 / 0 | [-16.172467339328726, -0.22609380541420832] |
| severity_aux_context_vs_global / full | 2 / 0 / 4 / 0 | [-0.6158571028627144, 0.5584552301215399] |
| severity_aux_context_vs_global / motion_only | 1 / 0 / 5 / 0 | [-7.961222776186118, 4.4734418975514965] |
| severity_aux_context_vs_original / full | 1 / 0 / 5 / 0 | [-21.052129793733346, 1.191965280166977] |
| severity_aux_context_vs_original / motion_only | 0 / 1 / 5 / 0 | [-51.36432631605501, 4.209201645783529] |

Three seeds averaged within locality;3000 resamples of four localities per assignment.
All six assignments retained; dependent exploratory comparisons, no multiplicity adjustment.
Training residuals are in-sample for the frozen base. Any positive probe needs separate nested-OOF validation.

```json
{
  "primary_six_positive_vs_original_and_global": false,
  "tail_coverage_guards": true,
  "diagnostic_context_signal": false,
  "independent_calibration": false,
  "new_neural_training": false,
  "policy_evaluated": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Detector pixels and annotation steps only. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.
