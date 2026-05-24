# Stage38 Correction Eval

- source: `fresh_run`
- correction deployable: `False`
- deployment decision: `keep_stage37_selector`

| model | all | t50 | hard | easy |
| --- | ---: | ---: | ---: | ---: |
| external_strongest_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Stage35_policy | 0.121319 | 0.000000 | 0.139849 | 0.000411 |
| Stage37_frozen_policy | 0.134825 | 0.084573 | 0.155434 | 0.000411 |
| Stage38_correction_with_fallback | 0.134825 | 0.084573 | 0.155434 | 0.000411 |
| Stage38_correction_without_fallback | 0.026500 | -0.593725 | 0.030441 | 0.420372 |
| Stage38_hard_only_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 |
| Stage38_t50_only_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 |
