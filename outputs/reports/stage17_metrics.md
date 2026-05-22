# Stage 17 Metrics

- official t+50 oracle improvement: `0.271291`
- diagnostic t+100 oracle improvement: `0.106599`
- official t+50 selector improvement: `0.081954`
- hard/failure selector improvement: `0.040700`
- correction incremental gain: `0.000000`
- easy degradation: `0.000000`

| model | subset | FDE | improvement | official |
| --- | --- | ---: | ---: | --- |
| global_strongest_causal_baseline | official_t50 | 3.318468 | 0.000000 | True |
| per_sample_oracle_baseline_diagnostic | official_t50 | 2.418196 | 0.271291 | False |
| trained_baseline_selector | official_t50 | 3.046507 | 0.081954 | True |
| BPSG-MA_v1 | official_t50 | 3.318468 | 0.000000 | True |
| selector_only_final_model | official_t50 | 3.046507 | 0.081954 | True |
| selector_plus_correction_specialist | official_t50 | 3.046507 | 0.081954 | True |
| no_scene_selector | official_t50 | 3.046507 | 0.081954 | True |
| no_goal_selector | official_t50 | 3.046507 | 0.081954 | True |
| no_interaction_selector | official_t50 | 3.318468 | 0.000000 | True |
