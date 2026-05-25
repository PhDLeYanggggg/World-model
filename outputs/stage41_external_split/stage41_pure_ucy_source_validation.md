# Stage41 Pure UCY Source-Heldout Validation

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- policy selected on: `non_ucy_validation_rows_only`
- pure UCY source-heldout gate: `True`
- pure UCY three-way train/val/test gate: `False`
- remaining blocker: `This validates frozen-policy UCY source-heldout behavior, but it is not a pure UCY-only retrain/select/test protocol because the frozen model and safety floor were trained on mixed external train data.`

| source | split | rows | all | t50 | t100 | hard/failure | easy | switch | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `UCY/zara01/obsmat.txt` | `val` | 16103 | 0.2183 | 0.1305 | 0.1520 | 0.2151 | 0.0000 | 0.4796 | `True` |
| `UCY/zara02/obsmat.txt` | `test` | 25901 | 0.1906 | 0.1358 | 0.1346 | 0.1871 | 0.0000 | 0.3938 | `True` |
| `UCY/zara03/crowds_zara03.txt` | `test` | 9540 | 0.2327 | 0.0914 | 0.1926 | 0.2260 | 0.0000 | 0.3334 | `True` |

- non-UCY validation rows used for policy selection: `53256`
- non-UCY validation domains: `['ETH_UCY', 'TrajNet']`
- duplicate blockers: `['TrajNet/Train/crowds/crowds_zara03.txt and UCY/zara03/crowds_zara03.txt are duplicate-like zara03 sources; they are not counted as independent UCY validation sources.']`
- no leakage: `{'future_endpoint_input': False, 'future_labels_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'target_source_excluded_from_policy_selection': True, 'policy_selected_on_non_ucy_validation_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## UCY Source Inventory

| split | source | rows | t50 | t100 |
| --- | --- | ---: | ---: | ---: |
| `train` | `UCY/students01/students001-trajnet.txt` | 47223 | 11583 | 7128 |
| `train` | `UCY/students03/obsmat.txt` | 70585 | 17529 | 15470 |
| `val` | `UCY/zara01/obsmat.txt` | 16103 | 3988 | 3251 |
| `test` | `UCY/zara02/obsmat.txt` | 25901 | 6422 | 5433 |
| `test` | `UCY/zara03/crowds_zara03.txt` | 9540 | 2340 | 1440 |

This is a pure-UCY source-heldout frozen-policy check, not a strict pure-UCY-only train/val/test retraining protocol. Coordinates remain dataset-local raw-frame 2.5D.
