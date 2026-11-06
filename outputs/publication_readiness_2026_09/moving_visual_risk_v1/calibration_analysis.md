# Frozen Source-Held Calibration Audit

Posthoc analysis after all36choices and the complete original readout.
This does not change the registered decision, select a model or tune a threshold.
Costs use a common past-history scale, not the different fitting RMS.
Conditional MSE uses known labels only. Unknowns stay in observation Brier;
they are not assigned zero-error ground truth. We average eight source groups
and three risk seeds with within-source query-balanced candidate weights.

| Context | Observation Brier | Gain MSE | Harm MSE | Reference MSE | Easy reference MSE | Easy harm MSE |
|---|---:|---:|---:|---:|---:|---:|
| no_image | 0.063197 | 0.006898 | 0.001664 | 0.123981 | 0.000579 | 0.000420 |
| image_bounds | 0.039600 | 0.006850 | 0.002875 | 0.115165 | 0.000520 | 0.000350 |
| quality_geometry | 0.049945 | 0.006566 | 0.001455 | 0.120155 | 0.000546 | 0.000328 |
| motion_geometry | 0.038765 | 0.007317 | 0.003445 | 0.133059 | 0.000577 | 0.000608 |

These diagnostics can explain a failed source-held policy but cannot
replace per-recording positive harm, signed easy-net and support checks.
They are conditional empirical errors, not calibration guarantees under
selection or on unseen sources. All per-source/seed values remain in JSON.
Same120+56input schema, no future inputs, no new C/H or reserved outcomes.
Native8/12image-coordinate development only; not metric/seconds or physical safety.
