# Stage 25 Oracle Margin Filter Report

| epsilon | retained | retained fraction | oracle headroom | top classes |
| --- | ---: | ---: | ---: | --- |
| 0px | 100000 | 1.0000 | 0.263732 | `[('constant_position', 73478), ('damped_velocity', 14040), ('constant_velocity_causal_fd', 10050), ('goal_directed_baseline', 1735), ('scene_clamped_baseline', 688), ('constant_acceleration_causal', 9)]` |
| 1px | 20746 | 0.2075 | 0.368285 | `[('damped_velocity', 13251), ('constant_position', 5381), ('goal_directed_baseline', 1448), ('scene_clamped_baseline', 658), ('constant_acceleration_causal', 8)]` |
| 2px | 19207 | 0.1921 | 0.374138 | `[('damped_velocity', 12230), ('constant_position', 5055), ('goal_directed_baseline', 1285), ('scene_clamped_baseline', 629), ('constant_acceleration_causal', 8)]` |
| 5px | 15589 | 0.1559 | 0.389575 | `[('damped_velocity', 9812), ('constant_position', 4204), ('goal_directed_baseline', 1003), ('scene_clamped_baseline', 564), ('constant_acceleration_causal', 6)]` |
| 10px | 12100 | 0.1210 | 0.411663 | `[('damped_velocity', 7845), ('constant_position', 2978), ('goal_directed_baseline', 789), ('scene_clamped_baseline', 485), ('constant_acceleration_causal', 3)]` |
| 1pct | 21699 | 0.2170 | 0.367342 | `[('damped_velocity', 13829), ('constant_position', 5533), ('goal_directed_baseline', 1650), ('scene_clamped_baseline', 678), ('constant_acceleration_causal', 9)]` |
| 2pct | 21451 | 0.2145 | 0.372025 | `[('damped_velocity', 13685), ('constant_position', 5482), ('goal_directed_baseline', 1605), ('scene_clamped_baseline', 670), ('constant_acceleration_causal', 9)]` |
| 5pct | 20722 | 0.2072 | 0.382397 | `[('damped_velocity', 13237), ('constant_position', 5352), ('goal_directed_baseline', 1487), ('scene_clamped_baseline', 638), ('constant_acceleration_causal', 8)]` |
| 10pct | 19736 | 0.1974 | 0.399391 | `[('damped_velocity', 12718), ('constant_position', 5098), ('goal_directed_baseline', 1319), ('scene_clamped_baseline', 594), ('constant_acceleration_causal', 7)]` |

- recommended epsilon: `0px`
- Best-baseline labels with tiny margins are unstable; Stage25 selectors must prefer soft labels or fallback.
