# Stage 5B.5 Failure Analysis

| dataset | scene_id | episode_id | agent_id | event_type | hardness | baseline_FDE | learned_FDE | ratio | likely_cause | recommended_fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trajnet | trajnet_bookstore_0 | 21 | 668 | turning,stop_and_go | easy | 0.086684 | 0.210209 | 2.425004 | baseline already strong; residual adds noise | Improve deterministic temporal model and validate residual gates per horizon/subset. |
| trajnet | trajnet_bookstore_0 | 19 | 667 | turning,stop_and_go | easy | 0.287335 | 0.531148 | 1.848531 | baseline already strong; residual adds noise | Improve deterministic temporal model and validate residual gates per horizon/subset. |
| trajnet | trajnet_bookstore_0 | 29 | 389 | turning,stop_and_go,close_interaction,high_density | medium | 0.069338 | 0.109914 | 1.585193 | baseline already strong; residual adds noise | Improve deterministic temporal model and validate residual gates per horizon/subset. |
