# Stage41 Source-Rotation Fresh Confirmation Split

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`

| domain | split | rows | source files | scenes | t50 | t100 | hard | easy | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | train | 108794 | 3 | 3 | 26597 | 20644 | 88241 | 28991 | 43400 |
| ETH_UCY | val | 16103 | 1 | 1 | 3988 | 3251 | 14187 | 1352 | 7631 |
| ETH_UCY | test | 25901 | 1 | 1 | 6422 | 5433 | 20861 | 6214 | 10474 |
| ETH_UCY | unused | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TrajNet | train | 63650 | 8 | 3 | 14652 | 8768 | 46014 | 21535 | 20951 |
| TrajNet | val | 37153 | 1 | 1 | 9113 | 5608 | 26922 | 9065 | 14502 |
| TrajNet | test | 20087 | 1 | 1 | 4927 | 3032 | 13491 | 8546 | 6638 |
| TrajNet | unused | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| UCY | train | 47223 | 1 | 1 | 11583 | 7128 | 31139 | 16840 | 14287 |
| UCY | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| UCY | test | 9540 | 1 | 1 | 2340 | 1440 | 7389 | 1979 | 4302 |
| UCY | unused | 9540 | 1 | 1 | 2340 | 1440 | 7389 | 1979 | 4302 |

- heldout test sources: `['TrajNet/Train/crowds/crowds_zara02.txt', 'UCY/zara02/obsmat.txt', 'UCY/zara03/crowds_zara03.txt']`
- overlap audit: `{'row_overlap': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'source_file_overlap': {'train_val': [], 'train_test': [], 'val_test': []}, 'scene_overlap': {'train_val': ['TrajNet_crowds'], 'train_test': ['TrajNet_crowds'], 'val_test': ['TrajNet_crowds']}, 'row_overlap_pass': True, 'source_file_overlap_pass': True, 'scene_overlap_note': 'TrajNet scene_id is coarse, so scene overlap can remain even with source-file held-out rotation; source-file overlap is the strict confirmation no-leakage check.'}`
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'source_file_overlap_pass': True, 'unused_duplicate_source': 'TrajNet/Train/crowds/crowds_zara03.txt is unused to avoid UCY zara03 duplicate leakage.'}`
