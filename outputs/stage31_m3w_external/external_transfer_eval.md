# Stage31 External Transfer Eval

- source: `fresh_run`
- Units are dataset-local coordinates; no metric/seconds claim.

| model | all improvement | t50 | hard | easy degradation | switch | regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage26_style_sdd_zero_shot_base_selector | -0.437562 | -1.038702 | -0.160809 | 2.804433 | 0.099835 | 0.077707 |
| m3w_las_v2_zero_shot_all_latent | -0.926675 | -2.785728 | -0.435809 | 3.164720 | 0.099835 | 0.145227 |
| external_strongest_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.017304 |
| external_oracle_diagnostic | 0.125352 | 0.066041 | 0.151735 | 0.000000 | 0.401815 | 0.000000 |
