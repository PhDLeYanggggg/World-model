# Stage 5B.5 Hard Subset Summary

Hard subset mining is used for evaluation stratification and training weights. It is not used as a future-derived input feature to the model.

| dataset | total | easy | medium | hard | hard_ratio | t100_hard | t50_hard | t10_hard | events | train_ok | eval_ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| eth_ucy | 23 | 10 | 11 | 2 | 0.087 | 0 | 0 | 2 | stop_and_go, turning, high_density, close_interaction, long_horizon_non_linear_motion | False | False |
| tgsim | 32 | 14 | 15 | 3 | 0.0938 | 3 | 3 | 3 | stop_and_go, turning, long_horizon_non_linear_motion, high_density, acceleration_or_deceleration | False | True |
| tgsim_i90 | 31 | 14 | 9 | 8 | 0.2581 | 8 | 8 | 8 | turning, acceleration_or_deceleration, stop_and_go, near_collision | True | True |
| trajnet | 32 | 14 | 12 | 6 | 0.1875 | 0 | 0 | 6 | stop_and_go, turning, close_interaction, long_horizon_non_linear_motion, high_density | True | True |
