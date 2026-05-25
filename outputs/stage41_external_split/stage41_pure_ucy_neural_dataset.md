# Stage41 Strict Pure-UCY Neural Dataset

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- splits: `{'train': {'rows': 117808, 'sources': 2, 't10': 34363, 't25': 31735, 't50': 29112, 't100': 22598, 'hard': 85739, 'easy': 38264, 'failure': 40518, 'history_len_mean': 23.249082565307617, 'history_ge_32': 31247, 'history_ge_64': 14985}, 'val': {'rows': 16103, 'sources': 1, 't10': 4580, 't25': 4284, 't50': 3988, 't100': 3251, 'hard': 14187, 'easy': 1352, 'failure': 7631, 'history_len_mean': 19.324100494384766, 'history_ge_32': 2251, 'history_ge_64': 517}, 'test': {'rows': 35441, 'sources': 2, 't10': 10283, 't25': 9523, 't50': 8762, 't100': 6873, 'hard': 28250, 'easy': 8193, 'failure': 14776, 'history_len_mean': 21.78970718383789, 'history_ge_32': 7203, 'history_ge_64': 4273}}`
- train-only strongest floor: `{10: 0, 25: 2, 50: 2, 100: 1}`
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'candidate_fde_input': False, 'candidate_fde_label_only': True, 'train_only_floor_selection': True, 'train_only_normalization_statistics': True, 'central_velocity': False, 'test_endpoint_goals': False, 'stage5c_executed': False, 'smc_enabled': False}`
