# Stage37 Cross-Domain Eval

- source: `fresh_run`
- final policy: `neighbor_history_t50_selector`
- bootstrap CI t50: `{'source': 'fresh_run', 'method': 'bootstrap_rows_500', 'low': 0.07692573799568757, 'mid': 0.08445480442424597, 'high': 0.09151819823299866, 'rows': 16263}`
- t100 status: `diagnostic_raw_frame_dataset_local`

| slice | all/improvement | t50 | t100 | hard | easy | switch | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| external_all | 0.134825 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| external_t10 | 0.306205 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| external_t25 | 0.000000 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| external_t50 | 0.084573 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| external_t100_diagnostic | 0.000000 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| external_hard_failure | 0.155434 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| external_easy | -0.000411 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| held_out_external_scenes | 0.134825 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| per_dataset_ETH | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | not_run |
| per_dataset_UCY | 0.134825 | 0.084573 | 0.000000 | 0.155434 | 0.000411 | 0.099045 | fresh_run |
| per_dataset_TrajNet | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | not_run |
| SDD_safety_check | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | cached_verified |
| SDD_easy_preservation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | cached_verified |
