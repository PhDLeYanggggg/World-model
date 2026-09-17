# Past Image Motion Forecasting

Fit-only exploratory comparison; no independent confirmation.

| Variant/objective | Gain vs CV (%) | Gain vs quality control (%) | Positive held fits | Easy passes | Binary oracle (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality_control_row_log | -0.9958 | 0.0000 | 0/9 | 0/9 | 0.3161 |
| magnitude_row_log | -0.8431 | 0.1513 | 0/9 | 0/9 | 0.3119 |
| directed_row_log | -0.9724 | 0.0231 | 0/9 | 0/9 | 0.3177 |
| quality_control_scene_ade_harm | -226.7387 | 0.0000 | 0/9 | 0/9 | 0.9935 |
| magnitude_scene_ade_harm | -201.3599 | 7.7673 | 0/9 | 0/9 | 1.3737 |
| directed_scene_ade_harm | -188.9957 | 11.5514 | 0/9 | 0/9 | 1.1655 |

Three historically used fit scenes;2,000 paired scene bootstrap draws are exploratory only.
No physical calibration, time-unit, strict online-sensor or deployment claim.
