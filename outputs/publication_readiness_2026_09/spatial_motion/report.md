# Spatial Image Motion Forecasting

Fit-only exploratory comparison; no independent confirmation.

| Variant/objective | Gain vs CV (%) | Gain vs quality control (%) | Positive held fits | Easy passes | Binary oracle (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality_control_row_log | -1.0260 | 0.0000 | 0/9 | 0/9 | 0.3255 |
| lowpass_pool_row_log | -1.0365 | -0.0103 | 0/9 | 0/9 | 0.3453 |
| lowpass_grid_row_log | -1.0810 | -0.0544 | 0/9 | 0/9 | 0.3393 |
| native_grid_row_log | -1.1385 | -0.1114 | 0/9 | 0/9 | 0.3281 |

Three historically used fit scenes;2,000 paired scene bootstrap draws are exploratory only.
No physical calibration, time-unit, strict online-sensor or deployment claim.
