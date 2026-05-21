# Stage 5B No-Leakage Audit

Official benchmark inputs use causal finite-difference velocity. Central-difference velocity is not used as an official input.

| dataset | passed | episodes | split_counts | official_velocity | cross_split_agents | flags |
| --- | --- | ---: | --- | --- | ---: | --- |
| eth_ucy | True | 23 | {'train': 15, 'val': 2, 'test': 6} | ['causal_fd'] | 0 | none |
| tgsim | True | 32 | {'train': 23, 'val': 5, 'test': 4} | ['causal_fd'] | 0 | none |
| tgsim_i90 | True | 31 | {'train': 18, 'test': 6, 'val': 7} | ['causal_fd'] | 0 | none |
| trajnet | True | 32 | {'train': 19, 'val': 6, 'test': 7} | ['causal_fd'] | 0 | none |
