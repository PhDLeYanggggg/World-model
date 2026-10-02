# Registered Advance Screen: Fail

| Requirement | Result | Evidence |
|---|---|---|
| Real72 fixed fits and exact replay |pass|72 fits plus exact refits; serialized inference/readout match|
| Original controls and input provenance unchanged |pass|Original selected95,455, unknown918, support33, upper7 reproduced|
| Training only, no independent-role access |pass|Registered source recording partition and train-only normalizers|
| MSE CI upper<0 vs original |fail|Upper+0.257608|
| MSE CI upper<0 vs additive |fail|Upper+0.301117|
| Full and matched utility CI lower>0 vs original |pass|+0.019681%,+0.006997%|
| Full and matched utility CI lower>0 vs additive |fail|-0.453519%,-0.078341%|
| Complete support not below original |pass|37 vs33|
| Known-label violations not above original |pass|2 vs4|
| Upper violations not above original |fail|11 vs7|
| Worst upper not above original |fail|18.028% vs5.406%|
| Advance to transfer |false|Six failed registered conditions|
| Deployment / independent confirmation / submission readiness |false|Not established|
| Stage5C / SMC |false|Not executed|

Passing engineering checks is not passing the research hypothesis. No combined
gate count is used to hide failed scientific criteria. Missing/empty support
remains failure. Utility percentages are not ADE/FDE gains. Locality intervals
are nominal exposed-development evidence only.
