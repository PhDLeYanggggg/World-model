# Matched-Count Safety Supplement

Completes the registered subset analysis of the already frozen count-matched choices.
No thresholds or selections change. T = Transformer; C = the indicated causal action.
The same smaller strict count is used per outer fold/seed, not a deployable online budget.

| Action | Head | T gain % | C gain % | T worst easy degradation % | C worst easy degradation % |
|---|---|---:|---:|---:|---:|
| constant_position | neural | 2.0004 | 1.8167 | 1.9542 | 2.0551 |
| constant_position | forest | 0.8585 | 0.7484 | 0.3066 | 0.3484 |
| damped_velocity_005 | neural | 2.4368 | 3.1952 | 1.0669 | 3.1656 |
| damped_velocity_005 | forest | 1.2889 | 1.3766 | 0.4444 | 0.2129 |
| damped_velocity_010 | neural | 2.4329 | 3.4922 | 1.0669 | 3.0824 |
| damped_velocity_010 | forest | 1.2565 | 1.2550 | 0.3955 | 0.0080 |
| damped_velocity_020 | neural | 2.3755 | 3.6091 | 1.6739 | 3.7374 |
| damped_velocity_020 | forest | 1.2249 | 1.1025 | 0.3274 | 0.0000 |
| constant_acceleration_causal | neural | 0.0070 | -0.0014 | 0.1035 | 0.0005 |
| constant_acceleration_causal | forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| constant_turn_rate | neural | 1.0032 | 0.5635 | 1.2313 | 1.4846 |
| constant_turn_rate | forest | 0.0316 | 0.0039 | 0.3784 | 0.0041 |

Reducing selection count does not monotonically reduce subgroup harm. In particular,
damping005 with the neural head has worse worst-site/seed easy degradation after this
count matching (3.166%) than under its original strict choices (2.494%). Selecting fewer
rows is not a safety certificate. All comparisons remain four-site development evidence.
