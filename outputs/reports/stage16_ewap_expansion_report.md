# Stage 16 EWAP Expansion Report

- t+100 rows: `81`
- t+50 rows: `433`
- official_policy: `t50_official_t100_diagnostic`
- t+100 target reached: `False`
- t+50 target reached: `False`

| policy | t50 rows | t100 rows | official allowed | diagnostic only | leakage risk |
| --- | ---: | ---: | --- | --- | --- |
| primary_agent_complete | 81 | 81 | True | False | low |
| partial_future_allowed | 81 | 81 | False | True | low |
| relaxed_neighbor_future_mask | 81 | 81 | True | False | low |
| scene_level_target_with_per_agent_evaluable_mask | 81 | 81 | False | True | low |
| t50_official_t100_diagnostic | 433 | 0 | True | False | low |

t+100 remains diagnostic because official per-agent rows are below 200; do not package t+50 as t+100.
