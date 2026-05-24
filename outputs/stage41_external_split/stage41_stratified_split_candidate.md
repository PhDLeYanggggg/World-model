# Stage41 Stratified Split Candidate

- source: `fresh_run`
- status: `candidate_protocol_not_used_for_stage41_claims`
- this does not overwrite `stage41_split_index.npz`; it is an input for the next retraining loop.

| domain | split | rows | files | t50 rows | oracle t50 | candidate6 t50 | hard | easy | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | train | 20057 | 3 | 4885 | 0.0611 | -1.2590 | 16951 | 4445 | 8387 |
| ETH_UCY | val | 21598 | 1 | 5074 | 0.4881 | 0.3271 | 18865 | 1812 | 10862 |
| ETH_UCY | test | 26092 | 1 | 6458 | 0.4474 | -0.9254 | 20199 | 7833 | 9797 |
| TrajNet | train | 13855 | 8 | 3019 | 0.0761 | -1.2393 | 10943 | 4350 | 5227 |
| TrajNet | val | 13698 | 1 | 3452 | 0.3283 | -0.9532 | 9966 | 3324 | 5337 |
| TrajNet | test | 17193 | 1 | 4258 | 0.5230 | -0.9062 | 11491 | 6169 | 5390 |
| UCY | train | 3490 | 1 | 869 | 0.0792 | -1.2378 | 2710 | 746 | 1558 |
| UCY | val | 9540 | 1 | 2340 | 0.0764 | -1.2484 | 7389 | 1979 | 4302 |
| UCY | test | 13254 | 1 | 3282 | 0.5226 | -0.8793 | 8805 | 4689 | 4087 |
