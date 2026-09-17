# Matched Track/Event Sampling

54 fresh Torch fits and 18 exact-replayed cached controls. Fixed-end checkpoints; all 11,966 fit windows retained.
Three historically used physical scenes, not independent confirmation. Targets only drive supervised training sampling and evaluation.

| Feature/sampling | Gain vs CV (%) | Gain vs row control (%) | Positive folds | Easy passes |
| --- | ---: | ---: | ---: | ---: |
| quality_control_row_uniform | -0.9958 | 0.0000 | 0/9 | 0/9 |
| quality_control_scene_uniform | -1.3725 | -0.3730 | 0/9 | 0/9 |
| quality_control_scene_track | -1.1993 | -0.2015 | 0/9 | 0/9 |
| quality_control_scene_event_track | -16.0531 | -14.9088 | 0/9 | 0/9 |
| directed_row_uniform | -0.9724 | 0.0000 | 0/9 | 0/9 |
| directed_scene_uniform | -1.2393 | -0.2643 | 0/9 | 0/9 |
| directed_scene_track | -1.1877 | -0.2132 | 0/9 | 0/9 |
| directed_scene_event_track | -10.4303 | -9.3668 | 0/9 | 0/9 |

2,000 scene bootstrap draws are exploratory; overlapping windows are not independent observations.
No new deployment, metric/physical time, Stage5C, SMC or foundation claim.
