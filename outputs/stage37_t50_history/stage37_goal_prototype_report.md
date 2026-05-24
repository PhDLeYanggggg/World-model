# Stage37 Scene-Agnostic Goal Prototype Report

- source: `fresh_run`
- prototypes are generated from past motion patterns, not test endpoints.
- prototype names: `['straight_continue', 'slow_stop', 'left_turn', 'right_turn', 'reverse_or_u_turn', 'group_follow', 'density_avoid', 'exit_like_direction_from_past_motion']`
- reports: `{'train': {'rows': 158942, 'prototype_count': 8, 'mean_entropy': 1.5178409814834595, 'mean_ambiguity': 0.6515766382217407}, 'val': {'rows': 112746, 'prototype_count': 8, 'mean_entropy': 1.5990002155303955, 'mean_ambiguity': 0.676776111125946}, 'test': {'rows': 66303, 'prototype_count': 8, 'mean_entropy': 1.5864468812942505, 'mean_ambiguity': 0.6736359000205994}}`
