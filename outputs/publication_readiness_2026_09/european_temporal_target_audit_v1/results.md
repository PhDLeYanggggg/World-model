# Temporal Target Probe Results

fresh_run: TRAIN-only analytic probe fitting, exact refitting and validation readout.
cached_verified: original forests, upstream forecasts, source labels and action hashes.
not_run: neural auxiliary training, changed policy, independent calibration/confirmation.

No future masks enter probe inference. No primary risk definition or deployment changes.

| Cohort / temporal-leaf contrast | Mean normalized MSE change | Nominal 95% locality CI | Localities |
|---|---:|---|---:|
| validation_temporal_minus_rowmean_leaf_signed_error | -0.094342 | [-0.158824, -0.038614] | 12 |
| validation_temporal_minus_rowmean_leaf_reference_error | -1.000240 | [-1.259768, -0.745426] | 12 |
| validation_temporal_minus_rowmean_leaf_mean_channels | -0.547291 | [-0.694845, -0.399388] | 12 |
| validation_temporal_minus_global_temporal_signed_error | -0.137388 | [-0.232556, -0.047422] | 12 |
| validation_temporal_minus_global_temporal_reference_error | -1.228049 | [-1.716269, -0.836964] | 12 |
| validation_temporal_minus_global_temporal_mean_channels | -0.682719 | [-0.949627, -0.458721] | 12 |
| complete_validation_temporal_minus_rowmean_leaf_signed_error | -0.098473 | [-0.159674, -0.047870] | 12 |
| complete_validation_temporal_minus_rowmean_leaf_reference_error | -0.954204 | [-1.234845, -0.680724] | 12 |
| complete_validation_temporal_minus_rowmean_leaf_mean_channels | -0.526338 | [-0.683768, -0.373489] | 12 |
| complete_validation_temporal_minus_global_temporal_signed_error | -0.171893 | [-0.270002, -0.085365] | 12 |
| complete_validation_temporal_minus_global_temporal_reference_error | -1.388404 | [-1.865951, -1.026257] | 12 |
| complete_validation_temporal_minus_global_temporal_mean_channels | -0.780149 | [-1.039575, -0.573808] | 12 |
| selected_validation_temporal_minus_rowmean_leaf_signed_error | +0.000043 | [-0.000140, +0.000256] | 11 |
| selected_validation_temporal_minus_rowmean_leaf_reference_error | +0.013270 | [-0.008763, +0.044258] | 11 |
| selected_validation_temporal_minus_rowmean_leaf_mean_channels | +0.006656 | [-0.004446, +0.022262] | 11 |
| selected_validation_temporal_minus_global_temporal_signed_error | -0.031441 | [-0.040050, -0.022099] | 11 |
| selected_validation_temporal_minus_global_temporal_reference_error | -1.162782 | [-1.344088, -0.992655] | 11 |
| selected_validation_temporal_minus_global_temporal_mean_channels | -0.597112 | [-0.688938, -0.510094] | 11 |

## Frozen Selected Cohort

Counts below are repeated head-view occurrences, not independent samples.

| Quantity | Value |
|---|---:|
| rows | 596988 |
| unknown_rows | 14076 |
| full_label_rows | 356904 |
| selected | 95455 |
| unknown_selected | 918 |
| selected_opposite_sign | 63622 |
| selected_netharm_rows | 12733 |
| selected_netharm_with_cancellation | 10749 |
| selected_nonharmful_but_step_harmful | 52873 |
| selected_harm | 1785.1213467305474 |
| selected_step_harm | 2967.1883943696203 |
| selected_cancellation | 1182.067047639071 |

Cancellation share of pooled gross step harm: 39.8380%.

Gross step harm minus cancellation reconstructs the original whole-trajectory harm.
These image-local masses are descriptive. They are neither risk percentages nor ADE/FDE gains.

## Per-Locality Signed-Error Contrast

| Locality | Temporal minus row-mean leaf | Temporal minus global time mean |
|---|---:|---:|
| eu-locality-007 | -0.032614 | -0.045907 |
| eu-locality-008 | -0.103706 | -0.210761 |
| eu-locality-020 | -0.273104 | -0.366215 |
| eu-locality-048 | -0.119972 | -0.218858 |
| eu-locality-067 | -0.351748 | -0.524608 |
| eu-locality-074 | -0.075520 | -0.122714 |
| eu-locality-082 | -0.039458 | +0.002013 |
| eu-locality-110 | +0.050553 | +0.120299 |
| eu-locality-112 | -0.032966 | -0.050296 |
| eu-locality-119 | -0.031057 | -0.066729 |
| eu-locality-124 | -0.033734 | -0.031156 |
| eu-locality-126 | -0.088772 | -0.133728 |

## Registered Diagnostic Screen

Advance to separately specified auxiliary-training design: True.
This screen cannot certify a deployment policy or a world-model improvement.


The 3,000-resample intervals cluster by locality and are nominal exposed-development
evidence, not search-adjusted independent confirmation. Scores retain the observed-label
task; a complete-label stratum is diagnostic, never a future-based inference filter.

Image-local detector-silver, rawstride12 obs8/pred12. No seconds, metric, physical-safety,
true-3D or foundation claim. Stage5C and SMC remain off.
