# Stage36 Cross-Domain Eval v4

- source: `fresh_run`
- final policy: `stage35_selective_transfer_with_t50_fallback`
- t+100 status: `diagnostic_raw_frame_dataset_local`

| slice | all/improvement | t50 | t100 | hard | easy | switch | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| external_all | 0.121319 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| external_t10 | 0.306205 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| external_t25 | 0.000000 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| external_t50 | 0.000000 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| external_t100_diagnostic | 0.000000 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| external_hard_failure | 0.139849 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| external_easy | -0.000411 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| held_out_external_scenes | 0.121319 | 0.000000 | 0.000000 | 0.139849 | 0.000411 | 0.049998 | fresh_run |
| SDD_safety_check | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | cached_verified |
| SDD_easy_preservation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | cached_verified |
