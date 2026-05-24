# Stage41 External Split Report

- source: `fresh_run`
- Rebuilt external held-out protocol; no UCY-only test.

| domain | split | rows | scenes | files | t50 | t100 | hard | easy | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | train | 112589 | 3 | 3 | 27939 | 24154 | 89648 | 28990 | 44336 |
| ETH_UCY | val | 16611 | 1 | 1 | 3994 | 2560 | 14776 | 5755 | 6307 |
| ETH_UCY | test | 21598 | 1 | 1 | 5074 | 2614 | 18865 | 1812 | 10862 |
| TrajNet | train | 95517 | 3 | 6 | 22751 | 13896 | 68173 | 29716 | 33099 |
| TrajNet | val | 21734 | 1 | 2 | 5110 | 3032 | 15138 | 8999 | 7138 |
| TrajNet | test | 3639 | 2 | 2 | 831 | 480 | 3116 | 431 | 1854 |
| UCY | train | 9540 | 1 | 1 | 2340 | 1440 | 7389 | 1979 | 4302 |
| UCY | val | 47223 | 1 | 1 | 11583 | 7128 | 31139 | 16840 | 14287 |
| UCY | test | 9540 | 1 | 1 | 2340 | 1440 | 7389 | 1979 | 4302 |

- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'test_endpoint_goals': False, 'central_velocity': False, 'candidate_goals': 'Stage37 scene-agnostic prototypes/past motion only; no new test endpoint goals'}`
