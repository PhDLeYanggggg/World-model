# Training-Only Microfit Results

## Material Passport

Fresh Torch optimization diagnostic on selected training rows. No held-site or main benchmark result.
Twelve models,24000 updates; full-batch2000 passes per row. This differs from the full-source exposure budget.
Three-seed ranges are descriptive optimizer variation, not confidence intervals for generalization.

| Cohort | Decoder | Rows | Final ADE | CV ADE | Mean gain (%) [seed range] | Easy absolute harm | Clipped updates |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| nonzero_only | context_radius | 16 | 46.475581 | 2552.587891 | +98.1793 [+97.8537, +98.4326] | n/a | 100.00% |
| nonzero_only | training_cost_scale | 16 | 39.826771 | 2552.587891 | +98.4397 [+98.3140, +98.5763] | n/a | 87.18% |
| mixed_zero | context_radius | 32 | 43.337593 | 1276.293945 | +96.6044 [+96.0840, +97.0074] | 33.888121 | 100.00% |
| mixed_zero | training_cost_scale | 32 | 38.459909 | 1276.293945 | +96.9866 [+96.2703, +97.6579] | 21.511272 | 72.03% |

All12 forecasts replay exactly; six decoder pairs have matched full-batch exposure/RNG state.
Completed resume preserves37 artifacts and the report with zero new updates. Targets are only loss labels.
Feasible training-label cohort selection is not a deployment filter. Easy relative degradation is undefined.
No independent confirmation, held-source score, main protocol change, Stage5C orSMC.
