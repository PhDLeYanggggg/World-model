# Stage41 Source-Level Validation Repair

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- source-level validation repair pass: `True`
- pure UCY source-level gate: `False`
- UCY-family surrogate gate: `True`
- UCY source-level blocker: `Pure UCY still has no independent validation source in the source-rotation split. Internal/temporal validation is useful but is not source-level evidence.`

## Frozen Test Source Metrics

| source | rows | t50 | t100 | all | t50 imp | t100 imp | hard | easy degr | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet/Train/crowds/crowds_zara02.txt` | 20087 | 4927 | 3032 | 0.2260 | 0.1645 | 0.1322 | 0.2157 | 0.0000 | 0.2650 |
| `UCY/zara02/obsmat.txt` | 25901 | 6422 | 5433 | 0.1808 | 0.1264 | 0.1168 | 0.1765 | 0.0000 | 0.3101 |
| `UCY/zara03/crowds_zara03.txt` | 9540 | 2340 | 1440 | 0.2296 | 0.0912 | 0.1846 | 0.2225 | 0.0000 | 0.3195 |

## Family Metrics

| family | rows | all | t50 | t100 | hard/failure | easy degradation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY_test` | 25901 | 0.1808 | 0.1264 | 0.1168 | 0.1765 | 0.0000 |
| `TrajNet_test` | 20087 | 0.2260 | 0.1645 | 0.1322 | 0.2157 | 0.0000 |
| `UCY_family_surrogate_test` | 55528 | 0.2036 | 0.1312 | 0.1337 | 0.1966 | 0.0000 |
| `pure_UCY_test` | 9540 | 0.2296 | 0.0912 | 0.1846 | 0.2225 | 0.0000 |

## Interpretation

The frozen teacher-guided candidate has positive source-heldout evidence and a positive UCY-family surrogate, but pure UCY source-level validation remains blocked. This supports continued candidate status, not final external source-level deployment evidence.

Pure UCY source-level validation is still not solved because the available split has no independent UCY validation source after excluding duplicate-like zara03. Internal folds and temporal UCY checks are cached-verified support, not a substitute for source-level evidence.

- no leakage: `{'future_endpoint_input': False, 'future_labels_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'source_file_overlap_pass': True, 'stage5c_executed': False, 'smc_enabled': False}`
